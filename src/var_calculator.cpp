#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <sqlite3.h>
#include <Eigen/Dense>

const double PI = 3.14159265358979323846;

class CopulaEngine {
public:
    CopulaEngine(const Eigen::MatrixXd& corrMatrix) : corr_(corrMatrix) {}
    Eigen::MatrixXd choleskyDecompose() const {
        Eigen::LLT<Eigen::MatrixXd> llt(corr_);
        if (llt.info() != Eigen::Success) std::cerr << "Cholesky failed" << std::endl;
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

struct RiskResult { double var; double cvar; };

// Historical Simulation: no distributional assumption -- reads the loss
// directly off the sorted empirical return distribution.
RiskResult historicalVaRCVaR(const Eigen::VectorXd& portfolioReturns, double confidence, double portfolioValue) {
    std::vector<double> sorted(portfolioReturns.data(), portfolioReturns.data() + portfolioReturns.size());
    std::sort(sorted.begin(), sorted.end());
    int idx = static_cast<int>((1.0 - confidence) * sorted.size());

    double var = -sorted[idx] * portfolioValue;

    double sumTail = 0.0;
    for (int i = 0; i <= idx; i++) sumTail += sorted[i];
    double cvar = -(sumTail / (idx + 1)) * portfolioValue;

    return {var, cvar};
}

// Parametric (Normal): closed-form VaR and CVaR under a Gaussian assumption.
// z = 1.645 for 95%, 2.326 for 99% (one-tailed normal critical values).
RiskResult parametricVaRCVaR(double mean, double stdDev, double confidence, double portfolioValue) {
    double z = (confidence >= 0.99) ? 2.326 : 1.645;
    double var = (z * stdDev - mean) * portfolioValue;

    double alpha = 1.0 - confidence;
    double phiZ = (1.0 / std::sqrt(2 * PI)) * std::exp(-0.5 * z * z);
    double esReturn = mean - stdDev * (phiZ / alpha);
    double cvar = -esReturn * portfolioValue;

    return {var, cvar};
}

// Monte Carlo: same percentile logic as historical, but on simulated
// rather than actual past outcomes.
RiskResult monteCarloVaRCVaR(std::vector<double> pnl, double confidence) {
    std::sort(pnl.begin(), pnl.end());
    int idx = static_cast<int>((1.0 - confidence) * pnl.size());
    double var = -pnl[idx];

    double sumTail = 0.0;
    for (int i = 0; i <= idx; i++) sumTail += pnl[i];
    double cvar = -(sumTail / (idx + 1));

    return {var, cvar};
}

std::vector<double> simulatePortfolioPnL(const Eigen::MatrixXd& corr, const Eigen::VectorXd& means,
                                          const Eigen::VectorXd& stdDevs, const Eigen::VectorXd& weights,
                                          double portfolioValue, int nSims) {
    CopulaEngine copula(corr);
    Eigen::MatrixXd L = copula.choleskyDecompose();
    int nAssets = means.size();
    std::mt19937 gen(42);
    std::normal_distribution<double> stdNormal(0.0, 1.0);

    std::vector<double> pnl;
    pnl.reserve(nSims);
    for (int i = 0; i < nSims / 2; i++) {
        Eigen::VectorXd z(nAssets);
        for (int j = 0; j < nAssets; j++) z(j) = stdNormal(gen);
        for (int sign : {1, -1}) {
            Eigen::VectorXd zCorr = L * (z * sign);
            double totalPnl = 0.0;
            for (int a = 0; a < nAssets; a++) {
                double simReturn = means(a) + stdDevs(a) * zCorr(a);
                totalPnl += weights(a) * (std::exp(simReturn) - 1.0) * portfolioValue;
            }
            pnl.push_back(totalPnl);
        }
    }
    return pnl;
}

struct RiskDecomposition { Eigen::VectorXd marginalVaR; Eigen::VectorXd componentVaR; };

// Marginal VaR = dVaR/dw_i. Component VaR = w_i * MarginalVaR_i. By Euler's
// theorem, these components sum EXACTLY to total parametric VaR -- that's
// not a coincidence, it falls out of VaR being a homogeneous function of
// the weights.
RiskDecomposition computeVaRDecomposition(const Eigen::VectorXd& weights, const Eigen::VectorXd& means,
                                           const Eigen::MatrixXd& covMatrix, double confidence, double portfolioValue) {
    double z = (confidence >= 0.99) ? 2.326 : 1.645;
    Eigen::VectorXd sigmaW = covMatrix * weights;
    double portfolioVol = std::sqrt(weights.dot(sigmaW));

    int n = weights.size();
    Eigen::VectorXd marginalVaR(n);
    for (int i = 0; i < n; i++) {
        marginalVaR(i) = portfolioValue * (z * sigmaW(i) / portfolioVol - means(i));
    }
    Eigen::VectorXd componentVaR = weights.array() * marginalVaR.array();
    return {marginalVaR, componentVaR};
}

int main() {
    Eigen::MatrixXd returns = loadReturnsMatrix("data/risk_engine.db", 5);
    Eigen::MatrixXd corr = computeCorrelation(returns);

    Eigen::VectorXd means = returns.colwise().mean();
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    Eigen::VectorXd variances = (centered.array().square().colwise().sum()) / (returns.rows() - 1);
    Eigen::VectorXd stdDevs = variances.array().sqrt();
    Eigen::MatrixXd covMatrix = stdDevs.asDiagonal() * corr * stdDevs.asDiagonal();

    Eigen::VectorXd weights(5);
    weights << 0.2, 0.2, 0.2, 0.2, 0.2;
    double portfolioValue = 1000000.0;

    Eigen::VectorXd portfolioReturns = returns * weights;
    double portfolioMean = weights.dot(means);
    double portfolioVol = std::sqrt(weights.dot(covMatrix * weights));

    const char* names[] = {"SPY", "IEF", "EURUSD=X", "GC=F", "NG=F"};

    std::cout << "=== Historical Simulation ===\n";
    for (double conf : {0.95, 0.99}) {
        RiskResult r = historicalVaRCVaR(portfolioReturns, conf, portfolioValue);
        std::cout << "  " << (int)(conf * 100) << "%: VaR=$" << r.var << "  CVaR=$" << r.cvar << "\n";
    }

    std::cout << "\n=== Parametric (Normal) ===\n";
    for (double conf : {0.95, 0.99}) {
        RiskResult r = parametricVaRCVaR(portfolioMean, portfolioVol, conf, portfolioValue);
        std::cout << "  " << (int)(conf * 100) << "%: VaR=$" << r.var << "  CVaR=$" << r.cvar << "\n";
    }

    std::cout << "\n=== Monte Carlo ===\n";
    std::vector<double> pnl = simulatePortfolioPnL(corr, means, stdDevs, weights, portfolioValue, 100000);
    for (double conf : {0.95, 0.99}) {
        RiskResult r = monteCarloVaRCVaR(pnl, conf);
        std::cout << "  " << (int)(conf * 100) << "%: VaR=$" << r.var << "  CVaR=$" << r.cvar << "\n";
    }

    std::cout << "\n=== Marginal & Component VaR (95%, parametric) ===\n";
    RiskDecomposition decomp = computeVaRDecomposition(weights, means, covMatrix, 0.95, portfolioValue);
    double sumComponent = 0.0;
    for (int i = 0; i < 5; i++) {
        std::cout << "  " << names[i] << ": Marginal=$" << decomp.marginalVaR(i)
                  << "  Component=$" << decomp.componentVaR(i) << "\n";
        sumComponent += decomp.componentVaR(i);
    }
    std::cout << "  Sum of components: $" << sumComponent << " (compare to Parametric 95% VaR above)\n";

    return 0;
}