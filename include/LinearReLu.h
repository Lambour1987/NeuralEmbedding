//28-10-2026
//24-11-2026 Inhoudt volledig aangepast

#pragma once

#include "Linear.h"
#include "Activations.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

class LinearReLU {
public:
    LinearReLU(Tensor weights, Tensor bias)
        : linear_(std::move(weights), std::move(bias)) {
    }

    Tensor forward(const Tensor& input) const {
        input_cache_ = input;

        Tensor linear_output = linear_.forward(input);
        linear_output_cache_ = linear_output;

        return relu(linear_output);
    }

    Tensor backward(const Tensor& output_gradients) {
        if (!input_cache_.has_value() ||
            !linear_output_cache_.has_value()) {
            throw std::logic_error(
                "Forward must be called before backward"
            );
        }

        const Tensor& linear_output = linear_output_cache_.value();

        if (output_gradients.shape() != linear_output.shape()) {
            throw std::invalid_argument(
                "Output gradients have incorrect shape"
            );
        }

        std::vector<float> relu_gradients =
            output_gradients.data();

        for (std::size_t i = 0; i < relu_gradients.size(); ++i) {
            if (linear_output.data()[i] <= 0.0f) {
                relu_gradients[i] = 0.0f;
            }
        }

        Tensor linear_gradients(
            std::move(relu_gradients),
            linear_output.shape()
        );

        return linear_.backward(
            input_cache_.value(),
            linear_gradients
        );
    }
        const Tensor& weight_gradients() const {
        return linear_.weight_gradients();
    }

    const Tensor& bias_gradients() const {
        return linear_.bias_gradients();
    }

    void update_parameters(float learning_rate) {
    linear_.update_parameters(learning_rate);
}

private:
    Linear linear_;

    mutable std::optional<Tensor> input_cache_;
    mutable std::optional<Tensor> linear_output_cache_;
};
