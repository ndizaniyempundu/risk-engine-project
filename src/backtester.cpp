#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <sqlite3.h>
#include <Eigen/Dense>

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

// Pulls the first `count` dates for one asset -- used as the reference
// "calendar" for the whole matrix. Approximate across assets (see Week 1
// notes on EURUSD=X trading extra days), fine for a diagnostic printout.
std::vector<std::string> loadDatesForAsset(const std::string& dbPath, int assetId, size_t count) {
    sqlite3* db;
    sqlite3_open(dbPath.c_str(), &db);
    std::vector<std::string> dates;
    std::string sql = "SELECT date FROM prices WHERE asset_id = ? AND log_return IS NOT NULL ORDER BY date ASC;";
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_int(stmt, 1, assetId);
    while (sqlite3_step(stmt) == SQLITE_ROW && dates.size() < count) {
        const unsigned char* text = sqlite3_column_text(stmt, 0);
        dates.push_back(reinterpret_cast<const char*>(text));
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return dates;
}

double historicalVaRFromWindow(std::vector<double> windowReturns, double confidence, double portfolioValue) {
    std::sort(windowReturns.begin(), windowReturns.end());
    int idx = static_cast<int>((1.0 - confidence) * windowReturns.size());
    return -windowReturns[idx] * portfolioValue;
}

int main() {
    const char* names[] = {"SPY", "IEF", "EURUSD=X", "GC=F", "NG=F"};
    Eigen::MatrixXd returns = loadReturnsMatrix("data/risk_engine.db", 5);
    Eigen::VectorXd weights(5);
    weights << 0.2, 0.2, 0.2, 0.2, 0.2;
    double portfolioValue = 1000000.0;

    Eigen::VectorXd portfolioReturns = returns * weights;
    int n = portfolioReturns.size();
    std::vector<std::string> dates = loadDatesForAsset("data/risk_engine.db", 1, n);

    int window = 252;
    double confidence = 0.95;

    // Now storing ABSOLUTE row index (not offset from window start), so we
    // can look up real dates/returns for each breach directly.
    std::vector<int> breachIndices;
    int totalTestedDays = 0;

    for (int t = window; t < n; t++) {
        std::vector<double> windowReturns(portfolioReturns.data() + (t - window), portfolioReturns.data() + t);
        double var95 = historicalVaRFromWindow(windowReturns, confidence, portfolioValue);
        double actualLoss = -portfolioReturns(t) * portfolioValue;
        totalTestedDays++;
        if (actualLoss > var95) {
            breachIndices.push_back(t);
        }
    }

    double breachRate = 100.0 * breachIndices.size() / totalTestedDays;
    std::cout << "=== VaR Backtest (95% Historical, rolling 252-day re-estimation) ===\n";
    std::cout << "Total days tested: " << totalTestedDays << "\n";
    std::cout << "Total breaches: " << breachIndices.size() << "\n";
    std::cout << "Breach rate: " << breachRate << "%\n\n";

    int baselWindow = 250;
    int cutoffIndex = n - baselWindow;
    int recentBreaches = 0;
    for (int idx : breachIndices) {
        if (idx >= cutoffIndex) recentBreaches++;
    }
    std::string zone;
    if (recentBreaches <= 4) zone = "GREEN";
    else if (recentBreaches <= 9) zone = "YELLOW";
    else zone = "RED";

    std::cout << "=== Basel Traffic-Light Test (most recent 250 trading days) ===\n";
    std::cout << "Breaches in window: " << recentBreaches << "\n";
    std::cout << "Zone: " << zone << "\n\n";

    std::cout << "=== Recent Breach Diagnostic ===\n";
    for (int idx : breachIndices) {
        if (idx < cutoffIndex) continue;
        double actualLoss = -portfolioReturns(idx) * portfolioValue;

        // Find which asset contributed the largest single-day dollar loss
        // that day -- this is what actually tests the "NG=F volatility
        // drives this" hypothesis, rather than just asserting it.
        int worstAsset = 0;
        double worstContribution = 0.0;
        for (int a = 0; a < 5; a++) {
            double contribution = -weights(a) * returns(idx, a) * portfolioValue;
            if (contribution > worstContribution) {
                worstContribution = contribution;
                worstAsset = a;
            }
        }
        std::cout << "  " << dates[idx] << ": Loss=$" << actualLoss
                  << "  |  Largest single-asset contributor: " << names[worstAsset]
                  << " ($" << worstContribution << ")\n";
    }

    return 0;
}