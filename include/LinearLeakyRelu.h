//25-11-26: Nieuw aangemaakt omdat we na 10 epoch al direct nullen enzo als waarden krgen
// of dezelfde waarden. 


#pragma once

#include "Linear.h"
#include "Activations.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

class LinearLeakyReLU {
public:
    explicit LinearLeakyReLU(
        Tensor weights,
        Tensor bias,
        float negative_slope = 0.01f
    )
        : linear_(std::move(weights), std::move(bias)),
          negative_slope_(negative_slope)
    {
        if (!std::isfinite(negative_slope_) ||
            negative_slope_ < 0.0f) {
            throw std::invalid_argument(
                "Negative slope must be finite and non-negative"
            );
        }
    }

    Tensor forward(const Tensor& input) const {
        input_cache_ = input;

        Tensor output = linear_.forward(input);
        linear_output_cache_ = output;

        return leaky_relu(output, negative_slope_);
    }

    const Tensor& linear_output() const {
        if (!linear_output_cache_.has_value()) {
            throw std::logic_error(
                "Forward must be called before accessing linear output"
            );
        }

        return linear_output_cache_.value();
    }

    Tensor activation_output() const {
        return leaky_relu(linear_output(), negative_slope_);
    }

    Tensor backward(
        const Tensor& output_gradients,
        bool accumulate = false
    ) {
        if (!input_cache_.has_value() ||
            !linear_output_cache_.has_value()) {
            throw std::logic_error(
                "Forward must be called before backward"
            );
        }

        const Tensor& output = linear_output_cache_.value();

        if (output_gradients.shape() != output.shape()) {
            throw std::invalid_argument(
                "Output gradients have incorrect shape"
            );
        }

        std::vector<float> gradients = output_gradients.data();

        for (std::size_t i = 0; i < gradients.size(); ++i) {
            if (output.data()[i] < 0.0f) {
                gradients[i] *= negative_slope_;
            }
        }

        Tensor linear_gradients(
            std::move(gradients),
            output.shape()
        );

        return linear_.backward(
            input_cache_.value(),
            linear_gradients,
            accumulate
        );
    }

    const Tensor& weight_gradients() const {
        return linear_.weight_gradients();
    }

    const Tensor& bias_gradients() const {
        return linear_.bias_gradients();
    }

    const Tensor& weights() const {
        return linear_.weights();
    }

    const Tensor& bias() const {
        return linear_.bias();
    }
    void set_parameters(const Tensor& new_weights,const Tensor& new_bias)
     {
    linear_.set_parameters(new_weights, new_bias);
    }

    void zero_grad() {
        linear_.zero_grad();
    }

    void scale_gradients(float factor) {
        linear_.scale_gradients(factor);
    }

    void update_parameters(float learning_rate) {
        linear_.update_parameters(learning_rate);
    }

    float negative_slope() const {
    return negative_slope_;
}


private:
    Linear linear_;
    float negative_slope_;

    mutable std::optional<Tensor> input_cache_;
    mutable std::optional<Tensor> linear_output_cache_;
};
