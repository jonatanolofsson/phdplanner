#pragma once
#ifndef STATISTICS_INC
#define STATISTICS_INC

#include <Eigen/Eigenvalues>
#include "statistics.hpp"

namespace ppl {
    std::random_device rd{};
    RandomGenerator rgen{rd()};
    static std::uniform_real_distribution<> udist;

    void seed(int s) { rgen.seed(s); }

    double urand() { return udist(rgen); }

    unsigned rand8() {
        static std::uniform_int_distribution<> idist8(0, 7);
        return idist8(rgen);
    }

    int randrel() {
        static std::uniform_int_distribution<> idist(-2, 2);
        return idist(rgen);
    }

    unsigned rand(int randmax) {
        std::uniform_int_distribution<> idist(0, randmax);
        return static_cast<unsigned>(idist(rgen));
    }
}

#endif // STATISTICS_INC
