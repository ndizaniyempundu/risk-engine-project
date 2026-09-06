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
            for (int j = 0; j < nAssets; j++) z(j) = stdNormal(gen);
            simulations.row(i) = (L * z).transpose();
        }
        return simulations;
    }

    Eigen::MatrixXd simulateStudentT(int nSims, double nu) const {
        Eigen::MatrixXd L = choleskyDecompose();
        int nAssets = corr_.rows();
        std::mt19937 gen(42);
        std::normal_distribution<double> stdNormal(0.0, 1.0);
        std::chi_squared_distribution<double> chiSq(nu);
        Eigen::MatrixXd simulations(nSims, nAssets);
        for (int i = 0; i < nSims; i++) {
            Eigen::VectorXd z(nAssets);
            for (int j = 0; j < nAssets; j++) z(j) = stdNormal(gen);
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

double percentile(Eigen::VectorXd col, double pct) {
    std::vector<double> vals(col.data(), col.data() + col.size());
    std::sort(vals.begin(), vals.end());
    int idx = static_cast<int>(pct * vals.size());
    return vals[idx];
}

// Counts concordant vs discordant pairs of observations. A pair is
// concordant if both assets moved the same direction relative to each
// other; discordant if not. O(n^2), but n~2700 is trivial for a computer.
double kendallTau(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    int n = x.size();
    long concordant = 0, discordant = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double sign = (x(i) - x(j)) * (y(i) - y(j));
            if (sign > 0) concordant++;
            else if (sign < 0) discordant++;
        }
    }
    return static_cast<double>(concordant - discordant) / (concordant + discordant);
}

// Converts raw values to ranks (1 = smallest), averaging ranks for ties.
Eigen::VectorXd rankValues(const Eigen::VectorXd& v) {
    int n = v.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v(a) < v(b); });

    Eigen::VectorXd ranks(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && v(idx[j + 1]) == v(idx[i])) j++;
        double avgRank = (i + j) / 2.0 + 1;
        for (int k = i; k <= j; k++) ranks(idx[k]) = avgRank;
        i = j + 1;
    }
    return ranks;
}

// Pearson correlation computed on ranks instead of raw values. Captures
// monotonic relationships and is robust to outliers/fat tails, unlike
// plain Pearson correlation on raw returns.
double spearmanRho(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    Eigen::VectorXd rx = rankValues(x);
    Eigen::VectorXd ry = rankValues(y);
    double meanX = rx.mean(), meanY = ry.mean();
    double cov = ((rx.array() - meanX) * (ry.array() - meanY)).mean();
    double stdX = std::sqrt((rx.array() - meanX).square().mean());
    double stdY = std::sqrt((ry.array() - meanY).square().mean());
    return cov / (stdX * stdY);
}

int main() {
    Eigen::MatrixXd returns = loadReturnsMatrix("data/risk_engine.db", 5);
    std::cout << "Loaded returns matrix: " << returns.rows() << " rows x " << returns.cols() << " assets\n\n";

    Eigen::MatrixXd corr = computeCorrelation(returns);
    std::cout << "Correlation matrix (SPY, IEF, EURUSD=X, GC=F, NG=F):\n" << corr << "\n\n";

    CopulaEngine engine(corr);
    Eigen::MatrixXd L = engine.choleskyDecompose();
    std::cout << "Cholesky factor:\n" << L << "\n";

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
    for (int i = 0; i < nTailSims; i++)
        if (gaussSims(i, 0) < gaussThresh0 && gaussSims(i, 1) < gaussThresh1) gaussJointTail++;

    double tThresh0 = percentile(tSims.col(0), 0.05);
    double tThresh1 = percentile(tSims.col(1), 0.05);
    int tJointTail = 0;
    for (int i = 0; i < nTailSims; i++)
        if (tSims(i, 0) < tThresh0 && tSims(i, 1) < tThresh1) tJointTail++;

    std::cout << "\n--- Tail dependence: Gaussian vs Student-t (toy corr=0.7) ---\n";
    std::cout << "Gaussian copula: P(both assets in bottom 5%) = " << (100.0 * gaussJointTail / nTailSims) << "%\n";
    std::cout << "Student-t copula (nu=4): P(both assets in bottom 5%) = " << (100.0 * tJointTail / nTailSims) << "%\n";
    std::cout << "Independence baseline: 0.05 * 0.05 = 0.25%\n";

    double tau = kendallTau(returns.col(0), returns.col(1));
    double rho = spearmanRho(returns.col(0), returns.col(1));
    std::cout << "\n--- Robust correlation checks: SPY vs IEF ---\n";
    std::cout << "Pearson correlation: " << corr(0, 1) << "\n";
    std::cout << "Kendall's tau: " << tau << "\n";
    std::cout << "Spearman's rho: " << rho << "\n";

    return 0;
}