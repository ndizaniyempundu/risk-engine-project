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

    // Multivariate Student-t: same correlated normal draws as above, but
    // scaled by sqrt(nu / W), where W ~ chi-squared(nu). This extra random
    // scaling is what creates fat tails AND tail dependence -- occasionally
    // the scale factor is large for everyone at once, dragging all assets
    // to extremes together. Lower nu = fatter tails = stronger co-crash effect.
    Eigen::MatrixXd simulateStudentT(int nSims, double nu) const {
        Eigen::MatrixXd L = choleskyDecompose();
        int nAssets = corr_.rows();

        std::mt19937 gen(42);
        std::normal_distribution<double> stdNormal(0.0, 1.0);
        std::chi_squared_distribution<double> chiSq(nu);

        Eigen::MatrixXd simulations(nSims, nAssets);
        for (int i = 0; i < nSims; i++) {
            Eigen::VectorXd z(nAssets);
            for (int j = 0; j < nAssets; j++) {
                z(j) = stdNormal(gen);
            }
            double w = chiSq(gen);
            double scale = std::sqrt(nu / w);
            simulations.row(i) = (L * z * scale).transpose();
        }
        return simulations;
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

// Finds the empirical value at a given percentile (e.g. 0.05 = 5th
// percentile) by sorting the column and picking the value at that position.
double percentile(Eigen::VectorXd col, double pct) {
    std::vector<double> vals(col.data(), col.data() + col.size());
    std::sort(vals.begin(), vals.end());
    int idx = static_cast<int>(pct * vals.size());
    return vals[idx];
}

int main() {
    Eigen::MatrixXd returns = loadReturnsMatrix("data/risk_engine.db", 5);
    std::cout << "Loaded returns matrix: " << returns.rows() << " rows x " << returns.cols() << " assets\n\n";

    Eigen::MatrixXd corr = computeCorrelation(returns);
    std::cout << "Correlation matrix (SPY, IEF, EURUSD=X, GC=F, NG=F):\n" << corr << "\n\n";

    CopulaEngine engine(corr);
    Eigen::MatrixXd L = engine.choleskyDecompose();
    std::cout << "Cholesky factor:\n" << L << "\n";

    // --- Why the copula choice matters: Gaussian vs Student-t ---
    // Toy 2-asset case with a clean, moderate positive correlation, so the
    // tail-dependence difference is easy to see and reason about directly.
    Eigen::MatrixXd toyCorr(2, 2);
    toyCorr << 1.0, 0.7,
               0.7, 1.0;
    CopulaEngine toyEngine(toyCorr);

    int nTailSims = 200000;
    Eigen::MatrixXd gaussSims = toyEngine.simulateGaussian(nTailSims);
    Eigen::MatrixXd tSims = toyEngine.simulateStudentT(nTailSims, 4.0);

    double gaussThresh0 = percentile(gaussSims.col(0), 0.05);
    double gaussThresh1 = percentile(gaussSims.col(1), 0.05);
    int gaussJointTail = 0;
    for (int i = 0; i < nTailSims; i++) {
        if (gaussSims(i, 0) < gaussThresh0 && gaussSims(i, 1) < gaussThresh1) {
            gaussJointTail++;
        }
    }

    double tThresh0 = percentile(tSims.col(0), 0.05);
    double tThresh1 = percentile(tSims.col(1), 0.05);
    int tJointTail = 0;
    for (int i = 0; i < nTailSims; i++) {
        if (tSims(i, 0) < tThresh0 && tSims(i, 1) < tThresh1) {
            tJointTail++;
        }
    }

    std::cout << "\n--- Tail dependence: Gaussian vs Student-t (toy corr=0.7) ---\n";
    std::cout << "Gaussian copula: P(both assets in bottom 5%) = "
              << (100.0 * gaussJointTail / nTailSims) << "%\n";
    std::cout << "Student-t copula (nu=4): P(both assets in bottom 5%) = "
              << (100.0 * tJointTail / nTailSims) << "%\n";
    std::cout << "Independence baseline: 0.05 * 0.05 = 0.25%\n";

    return 0;
}