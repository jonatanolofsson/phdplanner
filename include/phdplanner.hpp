// Copyright 2018 Jonatan Olofsson
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "statistics.hpp"
#include "omp.hpp"
#include "constants.hpp"
#include "params.hpp"

namespace ppl {
    static const unsigned E = 0;
    static const unsigned NE = 1;
    static const unsigned N = 2;
    static const unsigned NW = 3;
    static const unsigned W = 4;
    static const unsigned SW = 5;
    static const unsigned S = 6;
    static const unsigned SE = 7;
    // E, NE, N, NW, W, SW, S, SE
    const Eigen::Matrix<int, 2, 8> moves = (Eigen::Matrix<int, 2, 8>() <<
        0, 1, 1,  1,  0, -1, -1, -1,
        1, 1, 0, -1, -1, -1,  0,  1).finished();

struct Planner {
    using PHD = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Position = Eigen::Vector2i;

    using Action = uint8_t;
    using Actions = std::vector<Action>;
    using Path = Eigen::Array<int, 2, Eigen::Dynamic>;
    using Population = std::vector<Actions>;
    using Weights = Eigen::Array<double, 1, Eigen::Dynamic>;
    using Params = Params;

    PHD values;
    double eta;
    Weights w;
    std::size_t T;
    Position start;
    Action prior_action;
    Position maxpos;
    Population population;
    int best_index;
    double best_score;
    Params params;

    Planner(const PHD& values_, Params params_, Position start_, unsigned T_, unsigned N = 1e1, Action prior_action_ = 8)
    : values(values_),
      T(T_),
      start(start_),
      prior_action(prior_action_),
      maxpos(values.rows() - 1, values.cols() - 1),
      population(N),
      params(params_)
    {
        //std::cout << "Values shape: " << values.rows() << ", " << values.cols() << std::endl;
        //std::cout << "c 200, 2: " << values(200, 2) << std::endl;
        switch (params.pathgen) {
            case 1:
                PARFOR for (unsigned i = 0; i < N; ++i) { generate_path1(population[i]); } break;
            case 2:
                PARFOR for (unsigned i = 0; i < N; ++i) { generate_path2(population[i]); } break;
            case 3:
                PARFOR for (unsigned i = 0; i < N; ++i) { generate_path3(population[i]); } break;
            case 4:
                PARFOR for (unsigned i = 0; i < N; ++i) { generate_path4(population[i]); } break;
        };
        calculate_score();
    }

    void update_values(const PHD& new_values) {
        values = new_values;
        calculate_score();
    }

    void generate_path1(Actions& actions) {
        Action action;
        Action prev_action = (prior_action > 7 ? random_action() : prior_action);
        Position pos = start;
        Position new_pos;
        actions.resize(T);
        for (unsigned t = 0; t < T; ++t) {
            unsigned c = 40;
            do {
                action = random_action(prev_action);
                new_pos = pos + moves.col(action);
                if (!(--c)) { prev_action = prior_action; t = 0; }
            } while (out_of_bounds(new_pos));
            actions[t] = prev_action = action;
            pos = new_pos;
        }
    }

    bool is_valid(const Action action, const Action prev_action) {
        int d = static_cast<int>(action) - static_cast<int>(prev_action);
        return ((d % 8 + 8) % 8) < 3;
    }

    unsigned follow_line(Position& pos, const Position& new_pos, unsigned T, Actions::iterator actions) {
        unsigned t = 0;
        Action action;
        Position delta = new_pos - pos;
        //std::cout << "Follow from " << pos.transpose() << " to " << new_pos.transpose() << std::endl;
        if (delta.y() != 0) {
            double deltaerr = std::abs(static_cast<double>(delta.x()) / static_cast<double>(delta.y()));
            double error = 0;
            while (!(delta.x() == 0 && delta.y() == 0) && t < T) {
                action = (delta.y() > 0
                            ? (delta.x() > 0 ? NE : (delta.x() < 0 ? SE : E))            // Right
                            : (delta.y() == 0
                                ? (delta.x() > 0 ? N : S)                               // Up/down
                                : (delta.x() > 0 ? NW : (delta.x() < 0 ? SW : W))));     // Left
                if (action != 0 && action != 4) { error -= 1.0; }
                while (error < 0.5 && delta.y() != 0 && t < T) {
                    //std::cout << "\t\t\t\t\t\t\t\tDelta: " << delta.transpose() << ", la: " << static_cast<int>(action) << std::endl;
                    pos += moves.col(action);
                    delta -= moves.col(action);
                    //std::cout << "Pos0: " << pos.transpose() << ", Delta: " << delta.transpose() << ", la: " << static_cast<int>(action) << std::endl;
                    *(actions++) = action;
                    ++t;
                    error += deltaerr;
                    action = (delta.y() > 0 ? 0 : 4);
                }
                action = (delta.y() > 0
                            ? (delta.x() > 0 ? NE : (delta.x() < 0 ? SE : E))            // Right
                            : (delta.y() == 0
                                ? (delta.x() > 0 ? N : S)                               // Up/down
                                : (delta.x() > 0 ? NW : (delta.x() < 0 ? SW : W))));     // Left
                if (action != 2 && action != 6) { error += deltaerr; }
                while (error > 0.5 && delta.x() != 0 && t < T) {
                    //std::cout << "\t\t\t\t\t\t\t\tDelta: " << delta.transpose() << ", la: " << static_cast<int>(action) << std::endl;
                    pos += moves.col(action);
                    delta -= moves.col(action);
                    //std::cout << "Pos1: " << pos.transpose() << ", Delta: " << delta.transpose() << ", la: " << static_cast<int>(action) << std::endl;
                    *(actions++) = action;
                    ++t;
                    error -= 1.0;
                    action = (delta.x() > 0 ? 2 : 6);
                }

                if (delta.y() == 0) {
                    action = delta.x() > 0 ? 2 : 6;
                    while (delta.x() != 0 && t < T) {
                        //std::cout << "\t\t\t\t\t\t\t\tDelta: " << delta.transpose() << ", la: " << static_cast<int>(action) << std::endl;
                        pos += moves.col(action);
                        delta -= moves.col(action);
                        //std::cout << "Pos2: " << pos.transpose() << ", Delta: " << delta.transpose() << ", la: " << static_cast<int>(action) << std::endl;
                        *(actions++) = action;
                        ++t;
                    }
                }
            }
        } else {
            action = delta.x() > 0 ? 2 : 6;
            while (delta.x() != 0 && t < T) {
                pos += moves.col(action);
                delta -= moves.col(action);
                *(actions++) = action;
                ++t;
            }
        }
        return t;
    }

    void generate_path2(Actions& actions) {
        Position pos = start;
        Position new_pos;
        actions.resize(T);
        unsigned t = 0;
        while (t < T) {
            new_pos << rand(maxpos(0)), rand(maxpos(1));
            t += follow_line(pos, new_pos, T - t, std::begin(actions) + t);
        }
    }

    void generate_path3(Actions& actions) {
        Position pos = start;
        Position new_pos, old_pos;
        actions.resize(T);
        unsigned t = 0;
        auto vals = values;
        while (t < T) {
            double value = vals.sum() * urand();
            unsigned i = 0;
            double cumsum = 0;
            while (cumsum < value) { cumsum += vals(i++); }
            --i;

            old_pos = pos;
            new_pos << i / vals.cols(), i % vals.cols();
            //std::cout << "value " << t << ": " << value << " : go from " << pos.format(eigenformat) << " to " << new_pos.format(eigenformat) << " (" << i << ") | " << values(new_pos.x(), new_pos.y()) << " / " << vals(i) << std::endl;
            auto a = follow_line(pos, new_pos, T - t, std::begin(actions) + t);
            observe(vals, old_pos, std::begin(actions) + t, std::begin(actions) + t + a);
            t += a;
        }
    }

    void generate_path4(Actions& actions) {
        Position pos = start;
        Position new_pos, old_pos;
        actions.resize(T);
        unsigned t = 0;
        auto vals = values;
        while (t < T) {
            unsigned x, y;
            vals.maxCoeff(&x, &y);

            old_pos = pos;
            new_pos << x, y;
            auto a = follow_line(pos, new_pos, T - t, std::begin(actions) + t);
            //std::cout << "value " << t << "/" << T << " :: " << a << " : " << " : go from " << pos.format(eigenformat) << " to " << new_pos.format(eigenformat) << " | " << values(new_pos.x(), new_pos.y()) << std::endl;
            observe(vals, old_pos, std::begin(actions) + t, std::begin(actions) + t + a);
            t += a;
        }
    }

    void observe(PHD& vals, Position pos, Actions::iterator start, Actions::iterator end) {
        for(auto action = start; action != end; ++action) {
            pos += moves.col(*action);
            if (pos.x() < 0 || pos.y() < 0 || pos.x() > maxpos.x() || pos.y() > maxpos.y()) {
                std::cout << "Invalid pos (observe): (" << pos.x() << ", " << pos.y() << ") > (" << maxpos.x() << ", " << maxpos.y() << ")"<< std::endl;
                exit(1);
            }
            vals(pos.x(), pos.y()) *= 1 - params.pD;
        }
    }

    void normalize() {
        eta = w.sum();
        if (eta > 1e-8) {
            w /= eta;
        }
    }

    inline bool out_of_bounds(const Position pos) const {
        return (   pos.x() < 0
                || pos.y() < 0
                || pos.x() > maxpos.x()
                || pos.y() > maxpos.y());
    }

    void validify(Actions& actions) const {
        Action action;
        Action prev_action = (prior_action > 7 ? random_action() : prior_action);
        Position pos = start;
        Position new_pos;
        if (out_of_bounds(pos)) { throw "Start position out of bounds."; }

        for (unsigned t = 0; t < T; ++t) {
            unsigned c = 0;
            action = actions[t];
            new_pos = pos + moves.col(action);
            while (out_of_bounds(new_pos)) {
                if (++c == 40) {
                    c = 0;
                    if (t == 0) {
                        throw "No good start..";
                    } else {
                        --t;
                        pos -= moves.col(prev_action);
                        prev_action = (t > 0
                                        ? actions[t - 1]
                                        : (prior_action > 7 ? random_action() : prior_action));
                    }
                }
                action = random_action(prev_action);
                new_pos = pos + moves.col(action);
            }
            actions[t] = prev_action = action;
            pos = new_pos;
        }
    }

    static Action relaction(const Action action, int d) {
        return static_cast<Action>((8 + (action + d) % 8) % 8);
    }

    void calculate_score() {
        unsigned N = population.size();
        w.resize(1, N);
        std::vector<double> revisit_factor(T);
        revisit_factor[0] = 0;
        for (unsigned t = 1; t < T; ++t) { revisit_factor[t] = (1 - revisit_factor[t-1]) * params.pD; }
        for (unsigned t = 1; t < T; ++t) { revisit_factor[t] *= params.value_factor; }

        //std::cout << "Calculate score: " << std::endl;
        //std::cout << "\tstraight: " << params.straight_reward << std::endl;
        //std::cout << "\tdiagf: " << params.diagf_reward << std::endl;
        //std::cout << "\tside: " << params.side_reward << std::endl;
        //std::cout << "\tstreak: " << params.streak_reward << std::endl;
        //std::cout << "\trevisits: " << revisit_factor[0] << ", " << revisit_factor[1] << ", " << revisit_factor[2] << std::endl;

        PARFOR
        for (unsigned i = 0; i < N; ++i) {
            w[i] = 0;
            auto& actions = population[i];
            Action action;
            Position pos = start;
            std::vector<Position> visited(T);

            for (unsigned t = 0; t < T; ++t) {
                visited[t] = pos;
                action = actions[t];
                pos += moves.col(action);
                int visits = std::count_if(std::begin(visited) + std::max(0, static_cast<int>(t) - params.memory),
                                           std::begin(visited) + t,
                                           [pos](Position& p) { return p == pos; });
                if (pos.x() < 0 || pos.y() < 0 || pos.x() > maxpos.x() || pos.y() > maxpos.y()) {
                    std::cout << "Invalid pos: (" << pos.x() << ", " << pos.y() << ") > (" << maxpos.x() << ", " << maxpos.y() << ")"<< std::endl;
                    exit(1);
                }
                w[i] += revisit_factor[visits] * values(pos.x(), pos.y());
                if (t > 0) {
                         if (action == actions[t-1])                { w[i] += params.straight_reward; }
                    else if (action == relaction(actions[t-1], 1))  { w[i] += params.diagf_reward; }
                    else if (action == relaction(actions[t-1], -1)) { w[i] += params.diagf_reward; }
                    else if (action == relaction(actions[t-1], 2))  { w[i] += params.side_reward; }
                    else if (action == relaction(actions[t-1], -2)) { w[i] += params.side_reward; }

                    if (action == actions[t-1] ||
                        (action == relaction(actions[t-1], 1)) ||
                        (action == relaction(actions[t-1], -1))) {
                        for (unsigned tt = t-1; tt > 0 && tt > t - params.max_streak; --tt) {
                            if (action == actions[tt]) { w[i] += params.streak_reward; }
                            else { break; }
                        }
                    }
                }
            }
        }
        w.pow(params.score_power);
        int _;
        best_score = w.maxCoeff(&_, &best_index);
        normalize();
    }

    static Action random_action() {
        return static_cast<Action>(rand8());
    }

    static Action random_action(const Action prev_action) {
        if (prev_action == 8)  {
            return random_action();
        }
        return relaction(prev_action, randrel());
    }

    void breed(Actions& res1, Actions& res2, const Actions a, const Actions b, const double mutation) const {
        res1.resize(T);
        res2.resize(T);
        unsigned split = rand(T - 1);
        std::copy(std::begin(a), std::begin(a) + split, std::begin(res1));
        std::copy(std::begin(b) + split, std::end(b), std::begin(res1) + split);
        std::copy(std::begin(b), std::begin(b) + split, std::begin(res2));
        std::copy(std::begin(a) + split, std::end(a), std::begin(res2) + split);
        unsigned mutations = mutation * T;
        PARFOR
        for (unsigned i = 0; i < mutations; ++i) {
            unsigned gene;
            gene = rand(T - 1);
            res1[gene] = gene > 1 ? random_action(res1[gene - 1]) : random_action();
            gene = rand(T - 1);
            res2[gene] = gene > 1 ? random_action(res2[gene - 1]) : random_action();
        }
        PARSECS
        {
            PARSEC
            { validify(res1); }
            PARSEC
            { validify(res2); }
        }
    }

    void evolve(const double mutation = 0, const int N_ = -1) {
        unsigned N = (N_ > 0 ? N_ : population.size());
        std::vector<std::size_t> samples(N);
        std::vector<double> cdf(w.size());
        std::partial_sum(w.data(), w.data() + w.size(), std::begin(cdf));
        auto w0 = 1.0 / N;
        double u = urand() * w0;

        int j = 0;
        for (unsigned i = 0; i < N; ++i) {
            while (cdf[j] < u && j < w.size()) { ++j; }
            samples[i] = j;
            u += w0;
        }
        std::shuffle(samples.begin(), samples.end(), rgen);

        Population new_population(N);
        PARFOR
        for (unsigned i = 0; i < N / 2; ++i) {
            breed(new_population[2 * i],
                  new_population[2 * i + 1],
                  population[samples[2 * i]],
                  population[samples[2 * i + 1]],
                  mutation);
        }

        population = new_population;
        calculate_score();
    }

    void best_path(Path& path) const {
        get_path(best_index, path);
    }

    void get_path(const unsigned i, Path& path) const {
        path.resize(2, T + 1);
        path.col(0) = start;
        assert(i < population.size());
        for (unsigned t = 0; t < T; ++t) {
            path.col(t + 1) = path.col(t) + moves.array().col(population[i][t]);
        }
    }
};

auto& operator<<(std::ostream& os, const Planner::Actions& actions) {
    os << "{";
    bool first = true;
    for (auto action : actions) {
        if (!first) { os << ","; } else { first = false; }
        switch (action) {
            case E: os << "E"; break;
            case NE: os << "NE"; break;
            case N: os << "N"; break;
            case NW: os << "NW"; break;
            case W: os << "W"; break;
            case SW: os << "SW"; break;
            case S: os << "S"; break;
            case SE: os << "SE"; break;
        }
    }
    os << "}";
    return os;
}

template<typename T>
std::string print(const T& o) {
    std::stringstream s;
    o.repr(s);
    return s.str();
}
}
