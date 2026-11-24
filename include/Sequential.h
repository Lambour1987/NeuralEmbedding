//28-10-2026
//24-11-2026: volledig aangepast.

#pragma once

#include "LinearReLu.h"

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

class Sequential {
public:
    void add(LinearReLU layer) {
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

    Tensor backward(const Tensor& output_gradients) {
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
                        std::is_same_v<LayerType, LinearReLU>
                    ) {
                        return current_layer.backward(gradients);
                    }
                    else {
                        return current_layer.backward(
                            input_cache_[i],
                            gradients
                        );
                    }
                },
                layers_[i]
            );
        }

        return gradients;
    }


    const Tensor& weight_gradients(std::size_t layer_index) const {
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

    const Tensor& bias_gradients(std::size_t layer_index) const {
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

    std::size_t size() const {
        return layers_.size();
    }

private:
    std::vector<std::variant<LinearReLU, Linear>> layers_;

    mutable std::vector<Tensor> input_cache_;
};
