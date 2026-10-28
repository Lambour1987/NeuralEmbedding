//28-10-2026
#include <utility>

#pragma once

#include "Linear.h"
#include "Activations.h"

class LinearReLU {
public:
    LinearReLU(Tensor weights, Tensor bias)
        : linear_(std::move(weights), std::move(bias)) {
    }

    Tensor forward(const Tensor& input) const {
        Tensor linear_output = linear_.forward(input);
        return relu(linear_output);
    }

private:
    Linear linear_;
};
