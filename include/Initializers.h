//27-11-26: He Initialization


#pragma once

#include "Tensor.h"

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace initializers {

inline std::pair<Tensor, Tensor> he_initialize(
    std::size_t fan_in,
    std::size_t fan_out,
    std::mt19937& generator
) {
    if (fan_in == 0 || fan_out == 0) {
        throw std::invalid_argument(
            "fan_in and fan_out must be greater than zero"
        );
    }

    // He-initialisatie:
    // standaardafwijking = sqrt(2 / fan_in)
    const float standard_deviation =
        std::sqrt(2.0f / static_cast<float>(fan_in));

    std::normal_distribution<float> distribution(
        0.0f,
        standard_deviation
    );

    std::vector<float> weights(fan_in * fan_out);

    for (float& weight : weights) {
        weight = distribution(generator);
    }

    // Biases beginnen op nul.
    std::vector<float> biases(fan_out, 0.0f);

    Tensor weight_tensor(
        std::move(weights),
        {fan_in, fan_out}
    );

    Tensor bias_tensor(
        std::move(biases),
        {fan_out}
    );

    return {
        std::move(weight_tensor),
        std::move(bias_tensor)
    };
}

} // namespace initializers
