#include <iostream>
#include <vector>
#include <cmath>
#include <sqlite3.h>
#include <string>

class ReturnSeries {
public:
    ReturnSeries(const std::vector<double>& prices) : prices_(prices) {}

    std::vector<double> logReturns() const {
        std::vector<double> returns;
        for (size_t i = 1; i < prices_.size(); i++) {
            returns.push_back(std::log(prices_[i] / prices_[i - 1]));
        }
        return returns;
    }

    double meanReturn() const {
        std::vector<double> returns = logReturns();
        double sum = 0.0;
        for (double r : returns) {
            sum += r;
        }
        return sum / returns.size();
    }

    // Annualized volatility: daily std dev * sqrt(252). Uses n-1 (sample
    // variance) since we're estimating the true variance from a sample.
    double volatility() const {
        std::vector<double> returns = logReturns();
        double mean = meanReturn();
        double sumSq = 0.0;
        for (double r : returns) {
            sumSq += (r - mean) * (r - mean);
        }
        double variance = sumSq / (returns.size() - 1);
        return std::sqrt(variance) * std::sqrt(252.0);
    }

    // RiskMetrics EWMA volatility. Weights recent returns more heavily
    // than old ones, so it reacts faster to new market shocks than the
    // plain volatility() above, which treats every day equally.
    double ewmaVol(double lambda = 0.94) const {
        std::vector<double> returns = logReturns();
        double variance = returns[0] * returns[0];
        for (size_t i = 1; i < returns.size(); i++) {
            variance = lambda * variance + (1 - lambda) * returns[i - 1] * returns[i - 1];
        }
        return std::sqrt(variance) * std::sqrt(252.0);
    }

    // Asymmetry of the return distribution. 0 = symmetric. Negative =
    // long left tail (big losses more likely than big gains) -- typical
    // for equities.
    double skewness() const {
        std::vector<double> returns = logReturns();
        double mean = meanReturn();
        double n = static_cast<double>(returns.size());

        double sumSq = 0.0, sumCube = 0.0;
        for (double r : returns) {
            double diff = r - mean;
            sumSq += diff * diff;
            sumCube += diff * diff * diff;
        }
        double variance = sumSq / n;
        double stdDev = std::sqrt(variance);
        double thirdMoment = sumCube / n;
        return thirdMoment / (stdDev * stdDev * stdDev);
    }

    // Excess kurtosis: tail fatness relative to a normal distribution
    // (which has kurtosis = 3 exactly, hence "excess" = kurtosis - 3).
    // Positive = fatter tails than normal predicts -- your empirical
    // proof that returns aren't normally distributed.
    double kurtosis() const {
        std::vector<double> returns = logReturns();
        double mean = meanReturn();
        double n = static_cast<double>(returns.size());

        double sumSq = 0.0, sumQuad = 0.0;
        for (double r : returns) {
            double diff = r - mean;
            sumSq += diff * diff;
            sumQuad += diff * diff * diff * diff;
        }
        double variance = sumSq / n;
        double fourthMoment = sumQuad / n;
        return (fourthMoment / (variance * variance)) - 3.0;
    }

    // Jarque-Bera: combines skewness + kurtosis into one statistic
    // testing "are these returns normally distributed?" Larger = more
    // evidence against normality.
    double jarqueBera() const {
        double n = static_cast<double>(logReturns().size());
        double s = skewness();
        double k = kurtosis();
        return (n / 6.0) * (s * s + (k * k) / 4.0);
    }

private:
    std::vector<double> prices_;
};

#include <sqlite3.h>

// Connects to your database and pulls every close_price for one asset,
// ordered by date. This replaces the hardcoded vector you've been using.
std::vector<double> loadPricesFromDB(const std::string& dbPath, int assetId) {
    std::vector<double> prices;
    sqlite3* db;

    if (sqlite3_open(dbPath.c_str(), &db)) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return prices;
    }

    // The "?" is a placeholder -- we bind assetId to it below instead of
    // pasting the number directly into the SQL string. This is standard
    // practice (prevents SQL injection, and it's what production code does).
    std::string sql = "SELECT close_price FROM prices WHERE asset_id = ? ORDER BY date ASC;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return prices;
    }

    sqlite3_bind_int(stmt, 1, assetId);

    // sqlite3_step() fetches one row at a time -- this loop runs once
    // per row in the result set, same idea as iterating a cursor in Python.
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        prices.push_back(sqlite3_column_double(stmt, 0));
    }

    // C-style APIs like this one don't have automatic cleanup (no garbage
    // collector) -- you must explicitly release what you opened.
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return prices;
}

int main() {
    std::vector<double> prices = loadPricesFromDB("data/risk_engine.db", 1); // asset_id 1 = SPY
    std::cout << "Loaded " << prices.size() << " prices from database" << std::endl;

    ReturnSeries series(prices);

    std::cout << "Mean return: " << series.meanReturn() << std::endl;
    std::cout << "Volatility (annualized): " << series.volatility() << std::endl;
    std::cout << "EWMA volatility (annualized): " << series.ewmaVol() << std::endl;
    std::cout << "Skewness: " << series.skewness() << std::endl;
    std::cout << "Excess kurtosis: " << series.kurtosis() << std::endl;
    std::cout << "Jarque-Bera statistic: " << series.jarqueBera() << std::endl;

    return 0;
}