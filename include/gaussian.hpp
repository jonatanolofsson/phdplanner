// Copyright 2018 Jonatan Olofsson
#pragma once
#include <Eigen/Core>
#include <vector>
#include "statistics.hpp"
#include "constants.hpp"

namespace ppl {

template<int S>
struct alignas(16) Gaussian_ {
    static const int STATES = S;
    using Self = Gaussian_<S>;
    using State = Eigen::Matrix<double, STATES, 1>;
    using Covariance = Eigen::Matrix<double, STATES, STATES>;
    State x;
    Covariance P;
    double w;

    Gaussian_() {}

    Gaussian_(const State& x_, const Covariance& P_, double w_ = 1.0)
    : x(x_),
      P(P_),
      w(w_)
    {}

    bool operator<(const Gaussian_& b) const {
        return w < b.w;
    }

    State mean() const {
        return x;
    }

    Covariance cov() const {
        return P;
    }

    Eigen::Vector2d pos() const {
        return x.template head<2>();
    }

    Eigen::Matrix2d poscov() const {
        return P.template topLeftCorner<2, 2>();
    }

    template<typename RES>
    void sample(RES& res) const {
        nrand(res, x, P);
    }

    template<typename RES>
    void sampled_pos_pdf(const Eigen::Array<double, 2, Eigen::Dynamic>& points,
                         RES& res,
                         const double scale = 1) const {
        const double logSqrt2Pi = 0.5*std::log(2*M_PI);
        typedef Eigen::LLT<Eigen::Matrix2d> Chol;
        Chol chol(poscov());
        if (chol.info() != Eigen::Success) {
            throw "decomposition failed!";
        }
        const Chol::Traits::MatrixL& L = chol.matrixL();
        auto diff = (points.matrix().colwise() - pos()).matrix().eval();
        auto quadform = L.solve(diff).colwise().squaredNorm().array();
        auto pdf = ((-0.5*quadform - points.rows()*logSqrt2Pi).exp()
            / L.determinant()).eval();
        res.array() += pdf * scale / pdf.sum();
    }

    Eigen::Array<double, 1, Eigen::Dynamic>
    sampled_pos_pdf_h(const Eigen::Array<double, 2, Eigen::Dynamic>& points, const double scale = 1.0) {
        Eigen::Array<double, 1, Eigen::Dynamic> res(1, points.cols()); res.setZero();
        sampled_pos_pdf(points, res, scale);
        return res;
    }

    void repr(std::ostream& os) const {
        os << "{\"type\":\"G\","
            << "\"w\":" << w
            << ",\"x\":" << x.format(eigenformat)
            << ",\"P\":" << P.format(eigenformat)
            << "}";
    }

};

template<int S>
auto& operator<<(std::ostream& os, const Gaussian_<S> c) {
    c.repr(os);
    return os;
}
}  // namespace lmb
