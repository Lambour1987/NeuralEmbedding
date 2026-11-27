//27-11-26: Het doel is dat je Neural Network Engine zelfstandig 
// kan beoordelen hoe goed een getraind netwerk presteert, zonder dat de evaluatiecode onderdeel is van de Trainer.


#pragma once

#include "Loss.h"
#include "Sequential.h"

#include <stdexcept>
#include <vector>

class Evaluator {
public:
    float evaluate_mse(
        Sequential& network,
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& targets
    ) const {
        if (network.size() == 0) {
            throw std::logic_error(
                "Cannot evaluate an empty network"
            );
        }

        if (inputs.empty()) {
            throw std::invalid_argument(
                "Evaluation dataset cannot be empty"
            );
        }

        if (inputs.size() != targets.size()) {
            throw std::invalid_argument(
                "Inputs and targets must have the same size"
            );
        }

        float total_loss = 0.0f;

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            Tensor predictions = network.forward(inputs[i]);

            total_loss += mean_squared_error(
                predictions,
                targets[i]
            );
        }

        return total_loss /
            static_cast<float>(inputs.size());
    }
};
