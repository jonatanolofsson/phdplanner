// Copyright 2018 Jonatan Olofsson
#pragma once

namespace ppl {
struct Params {
    double value_factor = 1.0;
    double straight_reward = 1e-4;
    double diagf_reward = 5e-5;
    double side_reward = 5e-5;
    unsigned max_streak = 10;
    double streak_reward = 1e-4;
    double score_power = 2;
    double pD = 0.99;
    int memory = 1e9;
    unsigned pathgen = 3;
};
}
