//28-10-2026


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
