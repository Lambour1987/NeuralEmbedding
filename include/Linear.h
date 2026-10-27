//27-10-2026


#pragma once

#include "TensorOps.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

class Linear {
public:
    Linear(Tensor weights, Tensor bias)
        : weights_(std::move(weights)),
          bias_(std::move(bias))
    {
        if (weights_.shape().size() != 2) {
            throw std::invalid_argument(
                "Linear weights must be a 2D tensor"
            );
        }

        if (bias_.shape().size() != 1) {
            throw std::invalid_argument(
                "Linear bias must be a 1D tensor"
            );
        }

        if (weights_.shape()[1] != bias_.shape()[0]) {
            throw std::invalid_argument(
                "Bias size must match output features"
            );
        }
    }

    Tensor forward(const Tensor& input) const
    {
        if (input.shape().size() != 2) {
            throw std::invalid_argument(
                "Linear input must be a 2D tensor"
            );
        }

        if (input.shape()[1] != weights_.shape()[0]) {
            throw std::invalid_argument(
                "Input features do not match weights"
            );
        }

        const std::size_t batch_size = input.shape()[0];
        const std::size_t output_features = weights_.shape()[1];

        Tensor output = matmul(input, weights_);
        std::vector<float> result = output.data();

        for (std::size_t row = 0; row < batch_size; ++row) {
            for (std::size_t col = 0;
                 col < output_features;
                 ++col) {
                result[row * output_features + col] +=
                    bias_.data()[col];
            }
        }

        return Tensor(
            std::move(result),
            {batch_size, output_features}
        );
    }

    const Tensor& weights() const
    {
        return weights_;
    }

    const Tensor& bias() const
    {
        return bias_;
    }

private:
    Tensor weights_;
    Tensor bias_;
};
