#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Same 5 SPY closing prices already verified in SQLite
    std::vector<double> prices = {
        148.198516845703,
        148.174163818359,
        147.744781494141,
        148.652191162109,
        148.6845703125
    };

    // We can't compute a return for the very first price (nothing before it),
    // so the loop starts at index 1, not 0 — same reason your SQL log_return
    // column had a blank/NULL first row.
    for (size_t i = 1; i < prices.size(); i++) {
        double log_return = std::log(prices[i] / prices[i - 1]);
        std::cout << "Day " << i << ": " << log_return << std::endl;
    }

    return 0;
}