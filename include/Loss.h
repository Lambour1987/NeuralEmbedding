// 28-10-2026


#pragma once

#include "Tensor.h"

#include <stdexcept>

inline float mean_squared_error(
    const Tensor& predictions,
    const Tensor& targets
) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument(
            "Predictions and targets must have the same shape"
        );
    }

    if (predictions.size() == 0) {
        throw std::invalid_argument(
            "Cannot calculate MSE for empty tensors"
        );
    }

    float sum = 0.0f;

    for (std::size_t i = 0; i < predictions.size(); ++i) {
        const float difference =
            predictions.data()[i] - targets.data()[i];

        sum += difference * difference;
    }

    return sum / static_cast<float>(predictions.size());
}
