#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <sqlite3.h>
#include <Eigen/Dense>

class CopulaEngine {
public:
    CopulaEngine(const Eigen::MatrixXd& corrMatrix) : corr_(corrMatrix) {}

    Eigen::MatrixXd choleskyDecompose() const {
        Eigen::LLT<Eigen::MatrixXd> llt(corr_);
        if (llt.info() != Eigen::Success) {
            std::cerr << "Cholesky failed" << std::endl;
        }
        return llt.matrixL();
    }

private:
    Eigen::MatrixXd corr_;
};

Eigen::MatrixXd loadReturnsMatrix(const std::string& dbPath, int nAssets) {
    sqlite3* db;
    sqlite3_open(dbPath.c_str(), &db);
    std::vector<std::vector<double>> allReturns(nAssets);
    size_t minLength = SIZE_MAX;
    for (int assetId = 1; assetId <= nAssets; assetId++) {
        std::string sql = "SELECT log_return FROM prices WHERE asset_id = ? AND log_return IS NOT NULL ORDER BY date ASC;";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, assetId);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            allReturns[assetId - 1].push_back(sqlite3_column_double(stmt, 0));
        }
        sqlite3_finalize(stmt);
        minLength = std::min(minLength, allReturns[assetId - 1].size());
    }
    sqlite3_close(db);
    Eigen::MatrixXd returns(minLength, nAssets);
    for (int col = 0; col < nAssets; col++)
        for (size_t row = 0; row < minLength; row++)
            returns(row, col) = allReturns[col][row];
    return returns;
}

Eigen::MatrixXd computeCorrelation(const Eigen::MatrixXd& returns) {
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / (returns.rows() - 1);
    Eigen::VectorXd stdDevs = cov.diagonal().array().sqrt();
    Eigen::MatrixXd corr = cov;
    for (int i = 0; i < corr.rows(); i++)
        for (int j = 0; j < corr.cols(); j++)
            corr(i, j) /= (stdDevs(i) * stdDevs(j));
    return corr;
}

class MonteCarloEngine {
public:
    MonteCarloEngine(const Eigen::MatrixXd& corr, const Eigen::VectorXd& means,
                      const Eigen::VectorXd& stdDevs, const Eigen::VectorXd& weights,
                      double portfolioValue)
        : copula_(corr), means_(means), stdDevs_(stdDevs),
          weights_(weights), portfolioValue_(portfolioValue) {}

    // Simulates nSims one-day portfolio P&L outcomes. antithetic=true pairs
    // every draw z with its mirror -z, cutting estimator variance for the
    // same amount of "real" randomness.
    std::vector<double> simulatePortfolioPnL(int nSims, bool antithetic = true) const {
        Eigen::MatrixXd L = copula_.choleskyDecompose();
        int nAssets = means_.size();
        std::mt19937 gen(42);
        std::normal_distribution<double> stdNormal(0.0, 1.0);

        std::vector<double> pnl;
        pnl.reserve(nSims);

        int drawsNeeded = antithetic ? nSims / 2 : nSims;
        for (int i = 0; i < drawsNeeded; i++) {
            Eigen::VectorXd z(nAssets);
            for (int j = 0; j < nAssets; j++) z(j) = stdNormal(gen);

            pnl.push_back(pnlFromZ(L, z));
            if (antithetic) pnl.push_back(pnlFromZ(L, -z));
        }
        return pnl;
    }

private:
    double pnlFromZ(const Eigen::MatrixXd& L, const Eigen::VectorXd& z) const {
        Eigen::VectorXd zCorr = L * z;
        int nAssets = means_.size();
        double totalPnl = 0.0;
        for (int i = 0; i < nAssets; i++) {
            double simReturn = means_(i) + stdDevs_(i) * zCorr(i);
            totalPnl += weights_(i) * (std::exp(simReturn) - 1.0) * portfolioValue_;
        }
        return totalPnl;
    }

    CopulaEngine copula_;
    Eigen::VectorXd means_, stdDevs_, weights_;
    double portfolioValue_;
};

// 95% VaR: sort outcomes, find the 5th percentile loss, report as a
// positive number (VaR is conventionally quoted as "how much you could
// lose," not a negative P&L figure).
double computeVaR95(std::vector<double> pnl) {
    std::sort(pnl.begin(), pnl.end());
    int idx = static_cast<int>(0.05 * pnl.size());
    return -pnl[idx];
}

int main() {
    Eigen::MatrixXd returns = loadReturnsMatrix("data/risk_engine.db", 5);
    Eigen::MatrixXd corr = computeCorrelation(returns);

    Eigen::VectorXd means = returns.colwise().mean();
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    Eigen::VectorXd variances = (centered.array().square().colwise().sum()) / (returns.rows() - 1);
    Eigen::VectorXd stdDevs = variances.array().sqrt();

    std::cout << "Per-asset daily mean returns:\n" << means.transpose() << "\n";
    std::cout << "Per-asset daily volatility:\n" << stdDevs.transpose() << "\n\n";

    Eigen::VectorXd weights(5);
    weights << 0.2, 0.2, 0.2, 0.2, 0.2;
    double portfolioValue = 1000000.0;

    MonteCarloEngine mc(corr, means, stdDevs, weights, portfolioValue);

    std::cout << "--- Convergence check: 1-day 95% VaR estimate vs. simulation count ---\n";
    for (int n : {100, 1000, 10000, 100000}) {
        std::vector<double> pnl = mc.simulatePortfolioPnL(n, true);
        double var95 = computeVaR95(pnl);
        std::cout << "n=" << n << ": VaR_95 = $" << var95 << "\n";
    }

    return 0;
}