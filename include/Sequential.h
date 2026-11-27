//28-10-2026
//24-11-2026: volledig aangepast.
//25-11-2026: volledig aangepast


#pragma once

#include "LinearReLu.h"
#include "LinearLeakyReLU.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

//27-11-26: nieuwe class
enum class LayerType {
    Linear,
    ReLU,
    LeakyReLU
};

struct LayerParameters {
    LayerType type;
    Tensor weights;
    Tensor bias;
    float negative_slope = 0.0f;
};

class Sequential {
public:
    void add(LinearReLU layer) {
        layers_.push_back(std::move(layer));
    }

    void add(LinearLeakyReLU layer) {
        layers_.push_back(std::move(layer));
    }

    void add(Linear layer) {
        layers_.push_back(std::move(layer));
    }

    Tensor forward(const Tensor& input) const {
        input_cache_.clear();

        Tensor output = input;

        for (const auto& layer : layers_) {
            input_cache_.push_back(output);

            output = std::visit(
                [&output](const auto& current_layer) {
                    return current_layer.forward(output);
                },
                layer
            );
        }

        return output;
    }

    Tensor backward(
        const Tensor& output_gradients,
        bool accumulate = false
    ) {
        if (input_cache_.size() != layers_.size()) {
            throw std::logic_error(
                "Forward must be called before backward"
            );
        }

        Tensor gradients = output_gradients;

        for (std::size_t i = layers_.size(); i-- > 0;) {
            gradients = std::visit(
                [&](auto& current_layer) -> Tensor {
                    using LayerType =
                        std::decay_t<decltype(current_layer)>;

                    if constexpr (
                        std::is_same_v<LayerType, Linear>
                    ) {
                        return current_layer.backward(
                            input_cache_[i],
                            gradients,
                            accumulate
                        );
                    }
                    else {
                        return current_layer.backward(
                            gradients,
                            accumulate
                        );
                    }
                },
                layers_[i]
            );
        }

        return gradients;
    }

    void zero_grad() {
        for (auto& layer : layers_) {
            std::visit(
                [](auto& current_layer) {
                    current_layer.zero_grad();
                },
                layer
            );
        }
    }

    void scale_gradients(float factor) {
        if (!std::isfinite(factor) || factor < 0.0f) {
            throw std::invalid_argument(
                "Gradient scale must be a non-negative finite number"
            );
        }

        for (auto& layer : layers_) {
            std::visit(
                [factor](auto& current_layer) {
                    current_layer.scale_gradients(factor);
                },
                layer
            );
        }
    }

    void update_parameters(float learning_rate) {
        if (!std::isfinite(learning_rate) ||
            learning_rate <= 0.0f) {
            throw std::invalid_argument(
                "Learning rate must be a positive finite number"
            );
        }

        for (const auto& layer : layers_) {
            std::visit(
                [](const auto& current_layer) {
                    current_layer.weight_gradients();
                    current_layer.bias_gradients();
                },
                layer
            );
        }

        for (auto& layer : layers_) {
            std::visit(
                [learning_rate](auto& current_layer) {
                    current_layer.update_parameters(learning_rate);
                },
                layer
            );
        }
    }

    const Tensor& weight_gradients(
        std::size_t layer_index
    ) const {
        if (layer_index >= layers_.size()) {
            throw std::out_of_range(
                "Layer index is out of range"
            );
        }

        return std::visit(
            [](const auto& current_layer) -> const Tensor& {
                return current_layer.weight_gradients();
            },
            layers_[layer_index]
        );
    }

    const Tensor& bias_gradients(
        std::size_t layer_index
    ) const {
        if (layer_index >= layers_.size()) {
            throw std::out_of_range(
                "Layer index is out of range"
            );
        }

        return std::visit(
            [](const auto& current_layer) -> const Tensor& {
                return current_layer.bias_gradients();
            },
            layers_[layer_index]
        );
    }

    const Tensor& linear_output(
        std::size_t layer_index
    ) const {
        if (layer_index >= layers_.size()) {
            throw std::out_of_range(
                "Layer index is out of range"
            );
        }

        return std::visit(
            [](const auto& current_layer) -> const Tensor& {
                using LayerType =
                    std::decay_t<decltype(current_layer)>;

                if constexpr (
                    std::is_same_v<LayerType, Linear>
                ) {
                    throw std::invalid_argument(
                        "Linear layer has no separate activation output"
                    );
                }
                else {
                    return current_layer.linear_output();
                }
            },
            layers_[layer_index]
        );
    }

    Tensor activation_output(
        std::size_t layer_index
    ) const {
        if (layer_index >= layers_.size()) {
            throw std::out_of_range(
                "Layer index is out of range"
            );
        }

        return std::visit(
            [](const auto& current_layer) -> Tensor {
                using LayerType =
                    std::decay_t<decltype(current_layer)>;

                if constexpr (
                    std::is_same_v<LayerType, Linear>
                ) {
                    throw std::invalid_argument(
                        "Linear layer has no separate activation function"
                    );
                }
                else {
                    return current_layer.activation_output();
                }
            },
            layers_[layer_index]
        );
    }

    const Tensor& weights(std::size_t layer_index) const {
        if (layer_index >= layers_.size()) {
            throw std::out_of_range(
                "Layer index is out of range"
            );
        }

        return std::visit(
            [](const auto& current_layer) -> const Tensor& {
                return current_layer.weights();
            },
            layers_[layer_index]
        );
    }

    const Tensor& bias(std::size_t layer_index) const {
        if (layer_index >= layers_.size()) {
            throw std::out_of_range(
                "Layer index is out of range"
            );
        }

        return std::visit(
            [](const auto& current_layer) -> const Tensor& {
                return current_layer.bias();
            },
            layers_[layer_index]
        );
    }

    std::size_t size() const {
        return layers_.size();
    }

    //27-11-26
    LayerType layer_type(std::size_t layer_index) const {
    if (layer_index >= layers_.size()) {
        throw std::out_of_range(
            "Layer index is out of range"
        );
    }

    
    return std::visit(
        [](const auto& current_layer) -> LayerType {
            using Type = std::decay_t<decltype(current_layer)>;

            if constexpr (std::is_same_v<Type, Linear>) {
                return LayerType::Linear;
            }
            else if constexpr (std::is_same_v<Type, LinearReLU>) {
                return LayerType::ReLU;
            }
            else {
                return LayerType::LeakyReLU;
            }
        },
        layers_[layer_index]
    );
}
float negative_slope(std::size_t layer_index) const {
    if (layer_index >= layers_.size()) {
        throw std::out_of_range(
            "Layer index is out of range"
        );
    }

    return std::visit(
        [](const auto& current_layer) -> float {
            using Type = std::decay_t<decltype(current_layer)>;

            if constexpr (
                std::is_same_v<Type, LinearLeakyReLU>
            ) {
                return current_layer.negative_slope();
            }
            else {
                return 0.0f;
            }
        },
        layers_[layer_index]
    );
}

void set_parameters(
    const std::vector<LayerParameters>& parameters
) {
    // Controleer eerst het aantal lagen.
    if (parameters.size() != layers_.size()) {
        throw std::invalid_argument(
            "Checkpoint has an incorrect number of layers"
        );
    }

    // Fase 1: valideer alle lagen voordat we iets wijzigen.
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const LayerParameters& current = parameters[i];

        // Controleer het laagtype.
        if (current.type != layer_type(i)) {
            throw std::invalid_argument(
                "Checkpoint layer type does not match network"
            );
        }

        // Controleer de tensorvormen.
        if (current.weights.shape() != weights(i).shape()) {
            throw std::invalid_argument(
                "Checkpoint weights have incorrect shape"
            );
        }

        if (current.bias.shape() != bias(i).shape()) {
            throw std::invalid_argument(
                "Checkpoint bias has incorrect shape"
            );
        }

        // Controleer alle gewichten.
        for (float value : current.weights.data()) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Checkpoint weights contain non-finite values"
                );
            }
        }

        // Controleer alle biases.
        for (float value : current.bias.data()) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument(
                    "Checkpoint biases contain non-finite values"
                );
            }
        }

        // Controleer de LeakyReLU-instelling.
        if (current.type == LayerType::LeakyReLU) {
            if (!std::isfinite(current.negative_slope) ||
                current.negative_slope < 0.0f ||
                current.negative_slope != negative_slope(i)) {
                throw std::invalid_argument(
                    "Checkpoint negative slope does not match network"
                );
            }
        }
    }

    // Fase 2: alle controles zijn geslaagd.
    // Nu mogen de parameters daadwerkelijk worden vervangen.
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        std::visit(
            [&parameters, i](auto& current_layer) {
                current_layer.set_parameters(
                    parameters[i].weights,
                    parameters[i].bias
                );
            },
            layers_[i]
        );
    }
}


private:
    std::vector<
        std::variant<LinearReLU, LinearLeakyReLU, Linear>
    > layers_;

    mutable std::vector<Tensor> input_cache_;
};
