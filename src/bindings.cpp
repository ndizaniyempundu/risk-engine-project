#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <sqlite3.h>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

namespace py = pybind11;
const double PI = 3.14159265358979323846;

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

Eigen::MatrixXd computeCorrelationMatrix(const Eigen::MatrixXd& returns) {
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / (returns.rows() - 1);
    Eigen::VectorXd stdDevs = cov.diagonal().array().sqrt();
    Eigen::MatrixXd corr = cov;
    for (int i = 0; i < corr.rows(); i++)
        for (int j = 0; j < corr.cols(); j++)
            corr(i, j) /= (stdDevs(i) * stdDevs(j));
    return corr;
}

class RiskEngine {
public:
    RiskEngine(const std::string& dbPath, int nAssets, std::vector<double> weights, double portfolioValue)
        : weights_(Eigen::Map<Eigen::VectorXd>(weights.data(), weights.size())),
          portfolioValue_(portfolioValue) {
        returns_ = loadReturnsMatrix(dbPath, nAssets);
        corr_ = computeCorrelationMatrix(returns_);
        means_ = returns_.colwise().mean();
        Eigen::MatrixXd centered = returns_.rowwise() - returns_.colwise().mean();
        Eigen::VectorXd variances = (centered.array().square().colwise().sum()) / (returns_.rows() - 1);
        stdDevs_ = variances.array().sqrt();
        covMatrix_ = stdDevs_.asDiagonal() * corr_ * stdDevs_.asDiagonal();
        portfolioReturns_ = returns_ * weights_;
        portfolioMean_ = weights_.dot(means_);
        portfolioVol_ = std::sqrt(weights_.dot(covMatrix_ * weights_));
    }

    Eigen::MatrixXd getCorrelationMatrix() const { return corr_; }
    Eigen::VectorXd getPortfolioReturns() const { return portfolioReturns_; }
    int numObservations() const { return returns_.rows(); }

    double historicalVaR(double confidence) const { return historicalRisk(confidence).first; }
    double historicalCVaR(double confidence) const { return historicalRisk(confidence).second; }
    double parametricVaR(double confidence) const { return parametricRisk(confidence).first; }
    double parametricCVaR(double confidence) const { return parametricRisk(confidence).second; }
    double monteCarloVaR(double confidence, int nSims) const { return monteCarloRisk(confidence, nSims).first; }
    double monteCarloCVaR(double confidence, int nSims) const { return monteCarloRisk(confidence, nSims).second; }

    Eigen::VectorXd componentVaR(double confidence) const {
        double z = (confidence >= 0.99) ? 2.326 : 1.645;
        Eigen::VectorXd sigmaW = covMatrix_ * weights_;
        double portfolioVol = std::sqrt(weights_.dot(sigmaW));
        int n = weights_.size();
        Eigen::VectorXd marginalVaR(n);
        for (int i = 0; i < n; i++) {
            marginalVaR(i) = portfolioValue_ * (z * sigmaW(i) / portfolioVol - means_(i));
        }
        return (weights_.array() * marginalVaR.array()).matrix();
    }

    std::pair<double, std::vector<double>> stressTest(std::vector<double> shocks) const {
        std::vector<double> perAssetPnL(weights_.size());
        double total = 0.0;
        for (int i = 0; i < weights_.size(); i++) {
            double pnl = weights_(i) * shocks[i] * portfolioValue_;
            perAssetPnL[i] = pnl;
            total += pnl;
        }
        return {total, perAssetPnL};
    }

private:
    std::pair<double, double> historicalRisk(double confidence) const {
        std::vector<double> sorted(portfolioReturns_.data(), portfolioReturns_.data() + portfolioReturns_.size());
        std::sort(sorted.begin(), sorted.end());
        int idx = static_cast<int>((1.0 - confidence) * sorted.size());
        double var = -sorted[idx] * portfolioValue_;
        double sumTail = 0.0;
        for (int i = 0; i <= idx; i++) sumTail += sorted[i];
        double cvar = -(sumTail / (idx + 1)) * portfolioValue_;
        return {var, cvar};
    }

    std::pair<double, double> parametricRisk(double confidence) const {
        double z = (confidence >= 0.99) ? 2.326 : 1.645;
        double var = (z * portfolioVol_ - portfolioMean_) * portfolioValue_;
        double alpha = 1.0 - confidence;
        double phiZ = (1.0 / std::sqrt(2 * PI)) * std::exp(-0.5 * z * z);
        double esReturn = portfolioMean_ - portfolioVol_ * (phiZ / alpha);
        double cvar = -esReturn * portfolioValue_;
        return {var, cvar};
    }

    std::pair<double, double> monteCarloRisk(double confidence, int nSims) const {
        Eigen::LLT<Eigen::MatrixXd> llt(corr_);
        Eigen::MatrixXd L = llt.matrixL();
        int nAssets = means_.size();
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
                    double simReturn = means_(a) + stdDevs_(a) * zCorr(a);
                    totalPnl += weights_(a) * (std::exp(simReturn) - 1.0) * portfolioValue_;
                }
                pnl.push_back(totalPnl);
            }
        }
        std::sort(pnl.begin(), pnl.end());
        int idx = static_cast<int>((1.0 - confidence) * pnl.size());
        double var = -pnl[idx];
        double sumTail = 0.0;
        for (int i = 0; i <= idx; i++) sumTail += pnl[i];
        double cvar = -(sumTail / (idx + 1));
        return {var, cvar};
    }

    Eigen::MatrixXd returns_, corr_, covMatrix_;
    Eigen::VectorXd weights_, means_, stdDevs_, portfolioReturns_;
    double portfolioValue_, portfolioMean_, portfolioVol_;
};

PYBIND11_MODULE(risk_engine, m) {
    m.doc() = "Multi-asset portfolio risk engine (C++ core via pybind11)";
    py::class_<RiskEngine>(m, "RiskEngine")
        .def(py::init<const std::string&, int, std::vector<double>, double>(),
             py::arg("db_path"), py::arg("n_assets"), py::arg("weights"), py::arg("portfolio_value"))
        .def("get_correlation_matrix", &RiskEngine::getCorrelationMatrix)
        .def("get_portfolio_returns", &RiskEngine::getPortfolioReturns)
        .def("num_observations", &RiskEngine::numObservations)
        .def("historical_var", &RiskEngine::historicalVaR)
        .def("historical_cvar", &RiskEngine::historicalCVaR)
        .def("parametric_var", &RiskEngine::parametricVaR)
        .def("parametric_cvar", &RiskEngine::parametricCVaR)
        .def("monte_carlo_var", &RiskEngine::monteCarloVaR)
        .def("monte_carlo_cvar", &RiskEngine::monteCarloCVaR)
        .def("component_var", &RiskEngine::componentVaR)
        .def("stress_test", &RiskEngine::stressTest);
}