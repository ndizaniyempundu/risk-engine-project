#include <iostream>
#include <cmath>
#include <random>
#include <Eigen/Dense>

class CopulaEngine {
public:
    CopulaEngine(const Eigen::MatrixXd& corrMatrix) : corr_(corrMatrix) {}

    // Breaks corr_ into L such that L * L^T = corr_. Eigen::LLT does the
    // actual math -- we're just calling it and checking it succeeded.
    Eigen::MatrixXd choleskyDecompose() const {
        Eigen::LLT<Eigen::MatrixXd> llt(corr_);
        if (llt.info() != Eigen::Success) {
            std::cerr << "Cholesky failed -- matrix may not be positive semi-definite" << std::endl;
        }
        return llt.matrixL();
    }

    // Generates nSims draws of correlated standard normals. Each row = one
    // simulation, each column = one asset. This is the building block for
    // every simulation-based method later (Monte Carlo VaR, stress paths, etc.)
    Eigen::MatrixXd simulateGaussian(int nSims) const {
        Eigen::MatrixXd L = choleskyDecompose();
        int nAssets = corr_.rows();

        std::mt19937 gen(42);  // fixed seed -- reproducible while we're testing
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

int main() {
    // Simple 2-asset test case with a known 0.5 correlation. We'll swap
    // this for your real 5-asset correlation matrix next.
    Eigen::MatrixXd corr(2, 2);
    corr << 1.0, 0.5,
            0.5, 1.0;

    CopulaEngine engine(corr);

    Eigen::MatrixXd L = engine.choleskyDecompose();
    std::cout << "Cholesky factor L:\n" << L << "\n\n";

    int nSims = 100000;
    Eigen::MatrixXd sims = engine.simulateGaussian(nSims);

    // Compute the empirical correlation of the simulated output -- this is
    // the actual proof the simulation is correct, not just "it ran."
    Eigen::VectorXd col0 = sims.col(0);
    Eigen::VectorXd col1 = sims.col(1);
    double mean0 = col0.mean();
    double mean1 = col1.mean();

    double cov = ((col0.array() - mean0) * (col1.array() - mean1)).mean();
    double std0 = std::sqrt((col0.array() - mean0).square().mean());
    double std1 = std::sqrt((col1.array() - mean1).square().mean());
    double empiricalCorr = cov / (std0 * std1);

    std::cout << "Target correlation: 0.5" << std::endl;
    std::cout << "Empirical correlation from " << nSims << " sims: " << empiricalCorr << std::endl;

    return 0;
}