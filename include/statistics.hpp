#pragma once
#include <random>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace ppl {
    using RandomGenerator = std::mt19937;
    extern RandomGenerator rgen;
    void seed(int);
    unsigned rand8();
    int randrel();
    unsigned rand(int);
    double urand();
}

#include "statistics.inc"
