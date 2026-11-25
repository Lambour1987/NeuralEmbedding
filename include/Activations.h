//28-10-2026
//25-11-26 aangepast


#pragma once

#include "Tensor.h"

#include <vector>

inline Tensor relu(const Tensor& input) {
    std::vector<float> result = input.data();

    for (float& value : result) {
        if (value < 0.0f) {
            value = 0.0f;
        }
    }

    return Tensor(std::move(result), input.shape());
}

inline Tensor leaky_relu(
    const Tensor& input,
    float negative_slope = 0.01f
) {
    std::vector<float> result = input.data();

    for (float& value : result) {
        if (value < 0.0f) {
            value *= negative_slope;
        }
    }

    return Tensor(std::move(result), input.shape());
}
