#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <sqlite3.h>
#include <Eigen/Dense>

class CopulaEngine {
public:
    CopulaEngine(const Eigen::MatrixXd& corrMatrix) : corr_(corrMatrix) {}

    Eigen::MatrixXd choleskyDecompose() const {
        Eigen::LLT<Eigen::MatrixXd> llt(corr_);
        if (llt.info() != Eigen::Success) {
            std::cerr << "Cholesky failed -- matrix may not be positive semi-definite" << std::endl;
        }
        return llt.matrixL();
    }

    Eigen::MatrixXd simulateGaussian(int nSims) const {
        Eigen::MatrixXd L = choleskyDecompose();
        int nAssets = corr_.rows();

        std::mt19937 gen(42);
        std::normal_distribution<double> stdNormal(0.0, 1.0);

        Eigen::MatrixXd simulations(nSims, nAssets);
        for (int i = 0; i < nSims; i++) {
            Eigen::VectorXd z(nAssets);
            for (int j = 0; j < nAssets; j++) {
                z(j) = stdNormal(gen);
            }
            simulations.row(i) = (L * z).transpose();
        }
        return simulations;
    }

private:
    Eigen::MatrixXd corr_;
};

// Loads log returns for all 5 assets (asset_id 1-5) and aligns them by
// position (a simplification -- EURUSD=X trades on days equities don't,
// as we saw back in Week 1, so this isn't perfectly date-aligned yet).
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
    for (int col = 0; col < nAssets; col++) {
        for (size_t row = 0; row < minLength; row++) {
            returns(row, col) = allReturns[col][row];
        }
    }
    return returns;
}

Eigen::MatrixXd computeCorrelation(const Eigen::MatrixXd& returns) {
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / (returns.rows() - 1);

    Eigen::VectorXd stdDevs = cov.diagonal().array().sqrt();
    Eigen::MatrixXd corr = cov;
    for (int i = 0; i < corr.rows(); i++) {
        for (int j = 0; j < corr.cols(); j++) {
            corr(i, j) /= (stdDevs(i) * stdDevs(j));
        }
    }
    return corr;
}

int main() {
    Eigen::MatrixXd returns = loadReturnsMatrix("data/risk_engine.db", 5);
    std::cout << "Loaded returns matrix: " << returns.rows() << " rows x " << returns.cols() << " assets\n\n";

    Eigen::MatrixXd corr = computeCorrelation(returns);
    std::cout << "Correlation matrix (SPY, IEF, EURUSD=X, GC=F, NG=F):\n" << corr << "\n\n";

    CopulaEngine engine(corr);
    Eigen::MatrixXd L = engine.choleskyDecompose();
    std::cout << "Cholesky factor:\n" << L << std::endl;

    return 0;
}