//27-10-2026


#pragma once

#include "TensorOps.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

//24-11-26
#include <optional>
#include <utility>
#include <cmath>

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

    //24-11-2026: Uitbreiding Backward
    Tensor backward(
    const Tensor& input,
    const Tensor& output_gradients
    );

    const Tensor& weight_gradients() const;

    const Tensor& bias_gradients() const;

    void update_parameters(float learning_rate);

private:
    Tensor weights_;
    Tensor bias_;

    //24-11-2026: Nieuw
    std::optional<Tensor> weight_gradients_;
    std::optional<Tensor> bias_gradients_;
};

//24-11-26: onderstaande hier toevoegen? buiten de class

inline Tensor Linear::backward(
    const Tensor& input,
    const Tensor& output_gradients
)
{
    // Controleer de input
    if (input.shape().size() != 2 ||
        input.shape()[1] != weights_.shape()[0]) {
        throw std::invalid_argument(
            "Backward input shape does not match weights"
        );
    }

    // Controleer de output gradients
    if (output_gradients.shape().size() != 2 ||
        output_gradients.shape()[0] != input.shape()[0] ||
        output_gradients.shape()[1] != weights_.shape()[1]) {
        throw std::invalid_argument(
            "Output gradients have incorrect shape"
        );
    }
    if (input.shape()[0] == 0) {
    throw std::invalid_argument(
        "Backward requires a non-empty batch"
    );
}

    // dW = transpose(input) * output_gradients
    weight_gradients_ = matmul(
        transpose(input),
        output_gradients
    );

    // db = som van output_gradients over de batch
    const std::size_t batch_size = input.shape()[0];
    const std::size_t output_features = weights_.shape()[1];

    std::vector<float> bias_gradient(output_features, 0.0f);

    for (std::size_t row = 0; row < batch_size; ++row) {
        for (std::size_t col = 0; col < output_features; ++col) {
            bias_gradient[col] +=
                output_gradients.data()[row * output_features + col];
        }
    }

    bias_gradients_ = Tensor(
        std::move(bias_gradient),
        {output_features}
    );

    // dX = output_gradients * transpose(weights)
    return matmul(
        output_gradients,
        transpose(weights_)
    );
}

inline const Tensor& Linear::weight_gradients() const
{
    if (!weight_gradients_.has_value()) {
        throw std::logic_error(
            "Weight gradients have not been calculated"
        );
    }

    return weight_gradients_.value();
}

inline const Tensor& Linear::bias_gradients() const
{
    if (!bias_gradients_.has_value()) {
        throw std::logic_error(
            "Bias gradients have not been calculated"
        );
    }

    return bias_gradients_.value();
}

inline void Linear::update_parameters(float learning_rate)
{
    // Controleer de learning rate
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
        throw std::invalid_argument(
            "Learning rate must be a positive finite number"
        );
    }

    // Haal de berekende gradients op
    const Tensor& weight_gradient = weight_gradients();
    const Tensor& bias_gradient = bias_gradients();

    // Controleer of de gradientvormen overeenkomen
    if (weight_gradient.shape() != weights_.shape() ||
        bias_gradient.shape() != bias_.shape()) {
        throw std::logic_error(
            "Gradient shapes do not match parameter shapes"
        );
    }

    // Bereken de nieuwe weights
    std::vector<float> updated_weights = weights_.data();

    for (std::size_t i = 0; i < updated_weights.size(); ++i) {
        updated_weights[i] -=
            learning_rate * weight_gradient.data()[i];
    }

    // Bereken de nieuwe biases
    std::vector<float> updated_bias = bias_.data();

    for (std::size_t i = 0; i < updated_bias.size(); ++i) {
        updated_bias[i] -=
            learning_rate * bias_gradient.data()[i];
    }

    // Bouw nieuwe Tensor-objecten
    Tensor new_weights(
        std::move(updated_weights),
        weights_.shape()
    );

    Tensor new_bias(
        std::move(updated_bias),
        bias_.shape()
    );

    // Vervang de oude parameters
    weights_ = std::move(new_weights);
    bias_ = std::move(new_bias);
}

