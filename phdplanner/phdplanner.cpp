// Copyright 2018 Jonatan Olofsson
#include <signal.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <string>
#include <stdlib.h>
#include "phdplanner.hpp"

namespace py = pybind11;
using namespace py::literals;

struct HaltException : public std::exception {};


PYBIND11_MODULE(phdplanner, m) {
    using ppl::Params;
    m.def("srand", &std::srand);
    py::class_<Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("value_factor", &Params::value_factor)
        .def_readwrite("straight_reward", &Params::straight_reward)
        .def_readwrite("diagf_reward", &Params::diagf_reward)
        .def_readwrite("side_reward", &Params::side_reward)
        .def_readwrite("max_streak", &Params::max_streak)
        .def_readwrite("streak_reward", &Params::streak_reward)
        .def_readwrite("score_power", &Params::score_power)
        .def_readwrite("pD", &Params::pD)
        .def_readwrite("memory", &Params::memory)
        .def_readwrite("pathgen", &Params::pathgen);

    using ppl::Planner;
    py::class_<Planner>(m, "Planner")
        .def(py::init<typename Planner::PHD,
                      Params,
                      typename Planner::Position,
                      unsigned,
                      unsigned,
                      uint8_t>(),
             "phd"_a, "params"_a, "start"_a, "T"_a, "N"_a = 10000,
             "prior_action"_a = 8)
        .def_readwrite("params", &Planner::params)
        .def_readonly("w", &Planner::w)
        .def("update_values", &Planner::update_values, "values"_a)
        .def_readonly("population", &Planner::population)
        .def("evolve", &Planner::evolve, "mutation"_a = 0, "N"_a = -1)
        .def_readonly("best_index", &Planner::best_index)
        .def_readonly("best_score", &Planner::best_score)
        .def("get_path", [](const Planner& planner, const unsigned i) {
            typename Planner::Path path;
            planner.get_path(i, path);
            return path;
        })
        .def("best_path", [](const Planner& planner) {
            typename Planner::Path path;
            planner.best_path(path);
            return path;
        });
}
