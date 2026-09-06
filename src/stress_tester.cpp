#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <sqlite3.h>

// Per-asset simple % return shocks, in order: SPY, IEF, EURUSD=X, GC=F, NG=F
struct Scenario {
    std::string name;
    std::vector<double> shocks;
};

std::vector<Scenario> getScenarios() {
    return {
        // Equities crash, Treasuries rally (flight to safety), dollar
        // strengthens (EUR falls), gold sells off in the initial
        // deleveraging panic before its later rally, energy demand collapses.
        {"2008 Global Financial Crisis", {-0.40, 0.10, -0.05, -0.05, -0.30}},

        // Fastest crash in history, but Fed cut rates to zero (bonds rally
        // hard), commodities broadly fell on demand destruction fears.
        {"2020 COVID Crash", {-0.34, 0.05, -0.02, -0.05, -0.20}},

        // Highly localized energy event -- broad market barely moved, but
        // futures-level nat gas shock is large (spot prices moved far more
        // extremely than this, but that's not representative of NG=F).
        {"2021 Texas Winter Freeze", {-0.01, 0.00, 0.00, 0.00, 0.40}},

        // Aggressive rate hikes hit bonds hard, strong dollar, equities
        // down, but energy spiked on the Ukraine war supply shock.
        {"2022 Fed Rate Hike Cycle", {-0.25, -0.20, -0.15, -0.05, 0.50}},

        // Custom forward-looking scenario: a Taiwan Strait conflict --
        // equity selloff, flight to bonds and gold, energy spike on
        // supply-route disruption fears.
        {"Custom: Taiwan Strait Conflict", {-0.20, 0.08, -0.05, 0.15, 0.25}}
    };
}

class StressTester {
public:
    StressTester(const std::vector<double>& weights, double portfolioValue)
        : weights_(weights), portfolioValue_(portfolioValue) {}

    struct ScenarioResult {
        double totalPnL;
        std::vector<double> perAssetPnL;
    };

    // Applies a shock (simple % move per asset, e.g. -0.40 = -40%) directly
    // to each position's dollar value. Stress scenarios are conventionally
    // quoted as simple percentage moves, not log returns, so we match that.
    ScenarioResult applyScenario(const std::vector<double>& shocks) const {
        std::vector<double> perAssetPnL(weights_.size());
        double total = 0.0;
        for (size_t i = 0; i < weights_.size(); i++) {
            double pnl = weights_[i] * shocks[i] * portfolioValue_;
            perAssetPnL[i] = pnl;
            total += pnl;
        }
        return {total, perAssetPnL};
    }

private:
    std::vector<double> weights_;
    double portfolioValue_;
};

// Reverse stress test: instead of asking "what does scenario X do to my
// portfolio," ask "what SIZE of a 2008-shaped shock would it take to lose
// exactly $targetLoss?" We binary-search a scaling factor applied to a real
// scenario's shape, rather than searching an arbitrary/unconstrained
// direction -- more interpretable, and a fair simplification of ICAAP-style
// reverse stress testing for this project's scope.
double reverseStressTest(const StressTester& tester, const std::vector<double>& baseShocks,
                          double targetLoss, double tolerance = 1.0) {
    double lo = 0.0, hi = 10.0;
    for (int iter = 0; iter < 100; iter++) {
        double mid = (lo + hi) / 2.0;
        std::vector<double> scaled(baseShocks.size());
        for (size_t i = 0; i < baseShocks.size(); i++) scaled[i] = baseShocks[i] * mid;
        double loss = -tester.applyScenario(scaled).totalPnL;
        if (std::abs(loss - targetLoss) < tolerance) return mid;
        if (loss < targetLoss) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2.0;
}

void storeScenarioResults(sqlite3* db, const std::string& scenarioName,
                          const std::vector<double>& shocks, const std::vector<double>& perAssetPnL) {
    std::string sql = "INSERT INTO stress_scenarios (scenario_name, asset_id, shocked_return, pnl_impact) VALUES (?, ?, ?, ?);";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    for (size_t i = 0; i < shocks.size(); i++) {
        sqlite3_bind_text(stmt, 1, scenarioName.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 2, static_cast<int>(i) + 1);
        sqlite3_bind_double(stmt, 3, shocks[i]);
        sqlite3_bind_double(stmt, 4, perAssetPnL[i]);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
}

int main() {
    std::vector<double> weights = {0.2, 0.2, 0.2, 0.2, 0.2};
    double portfolioValue = 1000000.0;
    StressTester tester(weights, portfolioValue);
    const char* names[] = {"SPY", "IEF", "EURUSD=X", "GC=F", "NG=F"};

    sqlite3* db;
    sqlite3_open("data/risk_engine.db", &db);
    sqlite3_exec(db, "DELETE FROM stress_scenarios;", nullptr, nullptr, nullptr); // keep reruns clean

    auto scenarios = getScenarios();
    double worstLoss = -1e18;
    std::string worstName;

    std::cout << "=== Stress Test Results (Portfolio Value: $" << portfolioValue << ") ===\n\n";
    for (const auto& scenario : scenarios) {
        auto result = tester.applyScenario(scenario.shocks);
        std::cout << scenario.name << ": Total P&L = $" << result.totalPnL << "\n";
        for (size_t i = 0; i < 5; i++) {
            std::cout << "  " << names[i] << ": shock=" << (scenario.shocks[i] * 100)
                      << "%  P&L=$" << result.perAssetPnL[i] << "\n";
        }
        std::cout << "\n";

        storeScenarioResults(db, scenario.name, scenario.shocks, result.perAssetPnL);

        if (-result.totalPnL > worstLoss) {
            worstLoss = -result.totalPnL;
            worstName = scenario.name;
        }
    }
    sqlite3_close(db);

    std::cout << "Worst-case scenario: " << worstName << " (Loss = $" << worstLoss << ")\n\n";

    std::cout << "=== Reverse Stress Test ===\n";
    double targetLoss = 100000.0;
    double scaleFactor = reverseStressTest(tester, scenarios[0].shocks, targetLoss);
    std::vector<double> scaledShocks(scenarios[0].shocks.size());
    for (size_t i = 0; i < scaledShocks.size(); i++) scaledShocks[i] = scenarios[0].shocks[i] * scaleFactor;
    double checkPnL = tester.applyScenario(scaledShocks).totalPnL;

    std::cout << "A " << scaleFactor << "x scaling of the 2008 GFC shock shape produces a $" << targetLoss << " loss.\n";
    std::cout << "Verification: that scaled shock gives P&L = $" << checkPnL << " (should be close to -$" << targetLoss << ")\n";

    return 0;
}