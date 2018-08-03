// Copyright 2018 Jonatan Olofsson
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <vector>

#include "phdplanner.hpp"
#include "gaussian.hpp"

using namespace ppl;
using Gaussian = Gaussian_<2>;


TEST(PHDPlannerTests, ConstructPlanner) {
    unsigned nx = 50; unsigned ny = 40; unsigned T = 10;
    unsigned popsize = 1e4;
    Planner::PHD phd(nx, ny); phd.setZero();
    Eigen::Array<double, 2, Eigen::Dynamic> points(2, 50 * 40);
    unsigned i = 0;
    for (unsigned x = 0; x < nx; ++x) {
        for (unsigned y = 0; y < ny; ++y) {
            points.col(i) << x, y;
        }
    }
    Gaussian target1((Gaussian::State() << 10, 10).finished(),
                     (Gaussian::Covariance() << 10, 0, 0, 10).finished());
    Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> res(
        phd.data(), 1, phd.size());
    target1.sampled_pos_pdf(points, res);
    Planner::Position start; start << 5, 5;
    Planner::Params params;
    Planner plan(phd, params, start, T, popsize);
    plan.evolve();
    EXPECT_EQ(plan.population.size(), popsize);
    Planner::Path path;
    plan.best_path(path);
    EXPECT_EQ(path.cols(), T + 1);
    EXPECT_EQ(path.rows(), 2);
}


TEST(PHDPlannerTests, Evolve) {
    unsigned nx = 50; unsigned ny = 40; unsigned T = 20;
    Planner::PHD phd(nx, ny); phd.setZero();
    Eigen::Array<double, 2, Eigen::Dynamic> points(2, nx * ny);
    unsigned i = 0;
    for (unsigned x = 0; x < nx; ++x) {
        for (unsigned y = 0; y < ny; ++y) {
            points.col(i) << x, y;
        }
    }
    Gaussian target1((Gaussian::State() << 10, 10).finished(),
                     (Gaussian::Covariance() << 10, 0, 0, 10).finished());
    Eigen::Map<Eigen::Array<double, 1, Eigen::Dynamic>> res(
        phd.data(), 1, phd.size());
    target1.sampled_pos_pdf(points, res);
    Planner::Position start; start << 5, 5;
    Planner::Params params;
    Planner plan(phd, params, start, T);
    Planner::Path path;
    for (unsigned k = 0; k < 200; ++k) {
        plan.evolve();
        plan.best_path(path);
        EXPECT_EQ(path.cols(), T + 1);
        EXPECT_EQ(path.rows(), 2);
    }
}

TEST(PHDPlannerTests, random_action) {
    typename Planner::Action action = Planner::random_action();
    for (unsigned n = 0; n < 1000; ++n) {
        action = Planner::random_action(action);
        EXPECT_LT(action, 8);
        EXPECT_GE(action, 0);
    }
}
