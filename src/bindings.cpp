// bindings.cpp
// Exposes the C++ risk engine (return stats, correlation, copula-based
// simulation, VaR/CVaR, stress testing) to Python via pybind11.
// Compiled into a .so file that Python imports directly as `risk_engine`.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // lets pybind11 auto-convert std::vector/std::pair <-> Python list/tuple
#include <pybind11/eigen.h>   // lets pybind11 auto-convert Eigen matrices/vectors <-> NumPy arrays
#include <Eigen/Dense>
#include <sqlite3.h>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

namespace py = pybind11;
const double PI = 3.14159265358979323846;

// --- Data loading -----------------------------------------------------

// Pulls every non-null log_return for assets 1..nAssets from SQLite, in
// date order, and packs them into one matrix (rows = days, cols = assets).
// Truncates all series to the shortest one so the matrix is rectangular --
// a simplification, since EURUSD=X trades on days equities don't (see
// Week 1 notes), so this isn't perfectly date-aligned across assets.
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

// Pearson correlation matrix, computed manually (mean-center, then
// covariance, then normalize by each asset's std dev) rather than via a
// library shortcut -- this is the same formula validated against Python
// back in Module 3.
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

// --- Stress scenarios (Module 6) ---------------------------------------
// Per-asset simple % return shocks, in order: SPY, IEF, EURUSD=X, GC=F, NG=F.
// These are informed estimates translating the guide's broad-strokes
// scenario descriptions (e.g. "equity -40%") into concrete per-ticker
// numbers -- not sourced to one single citation. See docs/concepts_to_review.md
// for the reasoning behind each scenario's specific shock values.
std::vector<std::pair<std::string, std::vector<double>>> getStressScenarios() {
    return {
        {"2008 Global Financial Crisis", {-0.40, 0.10, -0.05, -0.05, -0.30}},
        {"2020 COVID Crash", {-0.34, 0.05, -0.02, -0.05, -0.20}},
        {"2021 Texas Winter Freeze", {-0.01, 0.00, 0.00, 0.00, 0.40}},
        {"2022 Fed Rate Hike Cycle", {-0.25, -0.20, -0.15, -0.05, 0.50}},
        {"Custom: Taiwan Strait Conflict", {-0.20, 0.08, -0.05, 0.15, 0.25}}
    };
}

// --- Main engine --------------------------------------------------------
// RiskEngine bundles everything: on construction, it loads real price
// history from SQLite and precomputes every stat (means, vols, correlation,
// covariance) ONCE. Every VaR/CVaR/stress method below then reuses those
// cached values instead of recomputing from raw data on every call -- this
// matters a lot in the Streamlit dashboard, where methods get called
// repeatedly as the user moves sliders.
class RiskEngine {
public:
    // weights: portfolio allocation per asset (e.g. 0.2 each for equal-weight).
    // portfolioValue: total dollars, e.g. $1,000,000.
    RiskEngine(const std::string& dbPath, int nAssets, std::vector<double> weights, double portfolioValue)
        : weights_(Eigen::Map<Eigen::VectorXd>(weights.data(), weights.size())),
          portfolioValue_(portfolioValue) {
        returns_ = loadReturnsMatrix(dbPath, nAssets);
        corr_ = computeCorrelationMatrix(returns_);
        means_ = returns_.colwise().mean();

        Eigen::MatrixXd centered = returns_.rowwise() - returns_.colwise().mean();
        Eigen::VectorXd variances = (centered.array().square().colwise().sum()) / (returns_.rows() - 1);
        stdDevs_ = variances.array().sqrt();

        // Covariance = correlation scaled by each asset's std dev (row AND
        // column), i.e. Cov_ij = Corr_ij * stdDev_i * stdDev_j. Needed for
        // the parametric VaR and marginal/component VaR formulas below.
        covMatrix_ = stdDevs_.asDiagonal() * corr_ * stdDevs_.asDiagonal();

        portfolioReturns_ = returns_ * weights_;               // historical daily portfolio returns
        portfolioMean_ = weights_.dot(means_);                 // weighted average daily return
        portfolioVol_ = std::sqrt(weights_.dot(covMatrix_ * weights_)); // portfolio std dev, accounting for correlation
    }

    // --- Simple accessors, mostly for dashboard display ---
    Eigen::MatrixXd getCorrelationMatrix() const { return corr_; }
    Eigen::VectorXd getPortfolioReturns() const { return portfolioReturns_; }
    int numObservations() const { return returns_.rows(); }

    // --- The 3 VaR/CVaR methodologies (Module 5) ---
    // Historical: no distributional assumption, reads the loss straight off
    // the sorted REAL past returns. Best at capturing fat tails, since it
    // uses whatever actually happened (e.g. the COVID crash), not an assumed
    // shape.
    double historicalVaR(double confidence) const { return historicalRisk(confidence).first; }
    double historicalCVaR(double confidence) const { return historicalRisk(confidence).second; }

    // Parametric: assumes returns are Normal. Fast closed-form, but
    // structurally CANNOT see fat tails/excess kurtosis -- this is why it
    // underestimates 99% CVaR relative to Historical (proven in Module 5).
    double parametricVaR(double confidence) const { return parametricRisk(confidence).first; }
    double parametricCVaR(double confidence) const { return parametricRisk(confidence).second; }

    // Monte Carlo: simulates thousands of correlated outcomes (via
    // Cholesky-decomposed correlation, same trick as CopulaEngine) and reads
    // the percentile off the simulated distribution.
    double monteCarloVaR(double confidence, int nSims) const { return monteCarloRisk(confidence, nSims).first; }
    double monteCarloCVaR(double confidence, int nSims) const { return monteCarloRisk(confidence, nSims).second; }

    // Component VaR: how much each asset contributes to TOTAL portfolio VaR.
    // By Euler's theorem, these sum EXACTLY to total parametric VaR -- not a
    // coincidence, verified numerically in Module 5 ($13,275.0 both ways).
    // NG=F dominating this (~89% of total risk despite 20% weight) is the
    // single most quotable finding in the whole project.
    Eigen::VectorXd componentVaR(double confidence) const {
        double z = (confidence >= 0.99) ? 2.326 : 1.645;  // one-tailed normal critical values
        Eigen::VectorXd sigmaW = covMatrix_ * weights_;
        double portfolioVol = std::sqrt(weights_.dot(sigmaW));
        int n = weights_.size();
        Eigen::VectorXd marginalVaR(n);
        for (int i = 0; i < n; i++) {
            marginalVaR(i) = portfolioValue_ * (z * sigmaW(i) / portfolioVol - means_(i));
        }
        // Component_i = weight_i * Marginal_i (this weighting is what makes
        // the components sum to the whole -- Euler's theorem for
        // homogeneous-degree-1 functions).
        return (weights_.array() * marginalVaR.array()).matrix();
    }

    // --- Stress testing (Module 6) ---
    // Applies a scenario's simple % shocks directly to each position's
    // dollar value (NOT log returns -- stress scenarios are conventionally
    // quoted as simple moves, e.g. "equity -40%").
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

    // Runs all 5 named historical/custom scenarios and returns just the
    // (name, total P&L) pairs -- convenient for the dashboard's stress
    // waterfall chart, which just needs one number per scenario.
    std::vector<std::pair<std::string, double>> runAllScenarios() const {
        std::vector<std::pair<std::string, double>> results;
        for (const auto& scenario : getStressScenarios()) {
            auto result = stressTest(scenario.second);
            results.push_back({scenario.first, result.first});
        }
        return results;
    }

    // Returns the RAW simulated portfolio P&L outcomes (not just a
    // percentile like monteCarloVaR does) -- needed so the dashboard can
    // plot the full P&L distribution as a histogram, with VaR/CVaR lines
    // marked on top of it.
    std::vector<double> getPortfolioPnLDistribution(int nSims) const {
        Eigen::LLT<Eigen::MatrixXd> llt(corr_);
        Eigen::MatrixXd L = llt.matrixL();  // Cholesky factor: L * L^T = corr_
        int nAssets = means_.size();
        std::mt19937 gen(42);  // fixed seed -- reproducible across dashboard reruns
        std::normal_distribution<double> stdNormal(0.0, 1.0);

        std::vector<double> pnl;
        pnl.reserve(nSims);
        // Antithetic variates: for every random draw z, also use -z. Their
        // simulation errors tend to cancel when averaged, tightening the
        // estimate without needing more truly-random draws (Module 4).
        for (int i = 0; i < nSims / 2; i++) {
            Eigen::VectorXd z(nAssets);
            for (int j = 0; j < nAssets; j++) z(j) = stdNormal(gen);
            for (int sign : {1, -1}) {
                Eigen::VectorXd zCorr = L * (z * sign);  // correlate the independent draws
                double totalPnl = 0.0;
                for (int a = 0; a < nAssets; a++) {
                    // Simplified 1-day GBM step: return = mean + vol * correlated_shock
                    double simReturn = means_(a) + stdDevs_(a) * zCorr(a);
                    totalPnl += weights_(a) * (std::exp(simReturn) - 1.0) * portfolioValue_;
                }
                pnl.push_back(totalPnl);
            }
        }
        return pnl;
    }

private:
    // Historical Simulation: sort actual past returns, read off the
    // (1-confidence) percentile as VaR, average everything worse than that
    // as CVaR.
    std::pair<double, double> historicalRisk(double confidence) const {
        std::vector<double> sorted(portfolioReturns_.data(), portfolioReturns_.data() + portfolioReturns_.size());
        std::sort(sorted.begin(), sorted.end());
        int idx = static_cast<int>((1.0 - confidence) * sorted.size());
        double var = -sorted[idx] * portfolioValue_;  // negate: VaR quoted as a positive "loss" number
        double sumTail = 0.0;
        for (int i = 0; i <= idx; i++) sumTail += sorted[i];
        double cvar = -(sumTail / (idx + 1)) * portfolioValue_;
        return {var, cvar};
    }

    // Parametric (Normal): closed-form using the z-score for the confidence
    // level and the normal PDF for CVaR's tail-average formula. No
    // simulation needed -- but this speed comes at the cost of assuming
    // normality, which Module 2 already proved is false for this data.
    std::pair<double, double> parametricRisk(double confidence) const {
        double z = (confidence >= 0.99) ? 2.326 : 1.645;
        double var = (z * portfolioVol_ - portfolioMean_) * portfolioValue_;
        double alpha = 1.0 - confidence;
        double phiZ = (1.0 / std::sqrt(2 * PI)) * std::exp(-0.5 * z * z);  // standard normal density at z
        double esReturn = portfolioMean_ - portfolioVol_ * (phiZ / alpha);
        double cvar = -esReturn * portfolioValue_;
        return {var, cvar};
    }

    // Monte Carlo: same correlated-simulation logic as
    // getPortfolioPnLDistribution above, but only returns the VaR/CVaR
    // percentile rather than the full distribution (cheaper when you just
    // need the summary numbers, e.g. for the VaR comparison bar chart).
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

// --- Python module definition --------------------------------------------
// PYBIND11_MODULE(risk_engine, m) is the macro that actually creates the
// Python-importable module. "risk_engine" here MUST match the .so filename
// (risk_engine.cpython-...so) -- Python's import system uses this name to
// find the module's entry point inside the compiled binary.
PYBIND11_MODULE(risk_engine, m) {
    m.doc() = "Multi-asset portfolio risk engine (C++ core via pybind11)";
    py::class_<RiskEngine>(m, "RiskEngine")
        // py::arg(...) names let Python callers use keyword arguments
        // (e.g. RiskEngine(db_path="...", n_assets=5, ...)) instead of only
        // positional -- nicer API, purely a usability addition.
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
        .def("stress_test", &RiskEngine::stressTest)
        .def("run_all_scenarios", &RiskEngine::runAllScenarios)
        .def("get_portfolio_pnl_distribution", &RiskEngine::getPortfolioPnLDistribution);
}