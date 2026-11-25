//25-11-2026


#pragma once

#include "Sequential.h"

#include <cmath>
#include <stdexcept>

class GradientDescent {
public:
    explicit GradientDescent(float learning_rate)
        : learning_rate_(learning_rate)
    {
        if (!std::isfinite(learning_rate_) ||
            learning_rate_ <= 0.0f) {
            throw std::invalid_argument(
                "Learning rate must be a positive finite number"
            );
        }
    }

    void step(Sequential& network) const {
        network.update_parameters(learning_rate_);
    }

    float learning_rate() const {
        return learning_rate_;
    }

private:
    float learning_rate_;
};
