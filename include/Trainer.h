//25-11-26


#pragma once

#include "GradientDescent.h"
#include "Loss.h"
#include "Sequential.h"

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <algorithm>

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

    // Trained het netwerk met mini-batches.
    // Per batch worden gradients verzameld en gemiddeld.
    // Daarna vindt precies één parameterupdate plaats.
    std::vector<float> train_mini_batch(
        Sequential& network,
        const GradientDescent& optimizer,
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& targets,
        std::size_t epochs,
        std::size_t batch_size
    ) const {
        if (network.size() == 0) {
            throw std::logic_error(
                "Cannot train an empty network"
            );
        }

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

        if (batch_size == 0) {
            throw std::invalid_argument(
                "Batch size must be greater than zero"
            );
        }

        // Controleer of alle voorbeelden dezelfde vorm hebben.
        const auto input_shape = inputs[0].shape();
        const auto target_shape = targets[0].shape();

        if (input_shape.size() != 2 ||
            input_shape[0] != 1 ||
            target_shape.size() != 2 ||
            target_shape[0] != 1) {
            throw std::invalid_argument(
                "Each training example must have batch dimension 1"
            );
        }

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].shape() != input_shape ||
                targets[i].shape() != target_shape) {
                throw std::invalid_argument(
                    "All training examples must have matching shapes"
                );
            }
        }

        std::vector<float> loss_history;
        loss_history.reserve(epochs);

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            float total_epoch_loss = 0.0f;

            for (
                std::size_t batch_start = 0;
                batch_start < inputs.size();
                batch_start += batch_size
            ) {
                const std::size_t current_batch_size =
                    std::min(
                        batch_size,
                        inputs.size() - batch_start
                    );

                // 1. Reset gradients voor deze batch.
                network.zero_grad();

                float batch_loss = 0.0f;

                // 2. Verwerk alle voorbeelden in deze batch.
                for (
                    std::size_t j = 0;
                    j < current_batch_size;
                    ++j
                ) {
                    const std::size_t index = batch_start + j;

                    Tensor predictions =
                        network.forward(inputs[index]);

                    const float loss =
                        mean_squared_error(
                            predictions,
                            targets[index]
                        );

                    batch_loss += loss;

                    Tensor loss_gradients =
                        mean_squared_error_backward(
                            predictions,
                            targets[index]
                        );

                    // true betekent: gradients optellen.
                    network.backward(loss_gradients, true);
                }

                // 3. Gemiddelde gradient berekenen.
                const float gradient_scale =
                    1.0f /
                    static_cast<float>(current_batch_size);

                network.scale_gradients(gradient_scale);

                // 4. Eén update voor de volledige batch.
                optimizer.step(network);

                total_epoch_loss += batch_loss;
            }

            // Gemiddelde loss over alle voorbeelden in deze epoch.
            const float average_epoch_loss =
                total_epoch_loss /
                static_cast<float>(inputs.size());

            loss_history.push_back(average_epoch_loss);
        }

        return loss_history;
    }

};
