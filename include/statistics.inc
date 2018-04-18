#pragma once
#ifndef STATISTICS_INC
#define STATISTICS_INC

#include <Eigen/Eigenvalues>
#include "statistics.hpp"
#include "omp.hpp"

namespace ppl {
    std::random_device rd{};
    RandomGenerator rgen{rd()};
    static std::uniform_real_distribution<> udist;

    void seed(int s) { rgen.seed(s); }

    double urand() { return udist(rgen); }

    unsigned rand8() {
        static std::uniform_int_distribution<> idist8(0, 7);
        unsigned res;
        CRITICAL(rand8)
        {
            res = idist8(rgen);
        }
        return res;
    }

    int randrel() {
        static std::uniform_int_distribution<> idist(-2, 2);
        int res;
        CRITICAL(randrel)
        {
            res = idist(rgen);
        }
        return res;
    }

    unsigned rand(int randmax) {
        std::uniform_int_distribution<> idist(0, randmax);
        unsigned res;
        CRITICAL(rand)
        {
            res = idist(rgen);
        }
        return res;
    }
}

#endif // STATISTICS_INC
