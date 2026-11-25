//25-11-26


#pragma once

#include "GradientDescent.h"
#include "Loss.h"
#include "Sequential.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

class Trainer {
public:
    // Voert één volledige trainingsstap uit.
    // Geeft de loss van vóór de parameterupdate terug.
    float train_step(
        Sequential& network,
        const GradientDescent& optimizer,
        const Tensor& input,
        const Tensor& target
    ) const {
        if (network.size() == 0) {
            throw std::logic_error(
                "Cannot train an empty network"
            );
        }

        // 1. Forward pass
        Tensor predictions = network.forward(input);

        // 2. Loss berekenen
        const float loss =
            mean_squared_error(predictions, target);

        // 3. Gradient van de loss berekenen
        Tensor loss_gradients =
            mean_squared_error_backward(predictions, target);

        // 4. Backpropagation
        network.backward(loss_gradients);

        // 5. Parameters bijwerken
        optimizer.step(network);

        return loss;
    }

    // Trained het netwerk gedurende meerdere epochs.
    // Iedere input/target-combinatie is één datapunt.
    std::vector<float> train(
        Sequential& network,
        const GradientDescent& optimizer,
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& targets,
        std::size_t epochs
    ) const {
        if (inputs.empty()) {
            throw std::invalid_argument(
                "Training dataset cannot be empty"
            );
        }

        if (inputs.size() != targets.size()) {
            throw std::invalid_argument(
                "Inputs and targets must have the same size"
            );
        }

        if (epochs == 0) {
            throw std::invalid_argument(
                "Number of epochs must be greater than zero"
            );
        }

        std::vector<float> loss_history;
        loss_history.reserve(epochs);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;

            for (std::size_t i = 0; i < inputs.size(); ++i) {
                total_loss += train_step(
                    network,
                    optimizer,
                    inputs[i],
                    targets[i]
                );
            }

            const float average_loss =
                total_loss / static_cast<float>(inputs.size());

            loss_history.push_back(average_loss);
        }

        return loss_history;
    }
};
