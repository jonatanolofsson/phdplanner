#pragma once
#include <Eigen/Core>

#define DPRINT(var) std::cout << #var ": " << var << std::endl;

namespace ppl {
    static Eigen::IOFormat eigenformat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",", "[", "]", "[", "]");
}
