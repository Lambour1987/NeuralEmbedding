//28-10-2026


#pragma once

#include "LinearReLU.h"

#include <utility>
#include <vector>

class Sequential {
public:
    void add(LinearReLU layer) {
        layers_.push_back(std::move(layer));
    }

    Tensor forward(const Tensor& input) const {
        Tensor output = input;

        for (const LinearReLU& layer : layers_) {
            output = layer.forward(output);
        }

        return output;
    }

    std::size_t size() const {
        return layers_.size();
    }

private:
    std::vector<LinearReLU> layers_;
};
