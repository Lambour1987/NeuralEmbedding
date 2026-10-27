//19-10-2026


#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "Tensor.h"

inline Tensor add(const Tensor& a, const Tensor& b)
{
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(
            "Cannot add tensors with different shapes"
        );
    }

    std::vector<float> result;
    result.reserve(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result.push_back(a.data()[i] + b.data()[i]);
    }

    return Tensor(std::move(result), a.shape());
}


inline Tensor subtract(const Tensor& a, const Tensor& b)
{
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(
            "Cannot subtract tensors with different shapes"
        );
    }

    std::vector<float> result;
    result.reserve(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result.push_back(a.data()[i] - b.data()[i]);
    }

    return Tensor(std::move(result), a.shape());
}

inline Tensor multiply(const Tensor& a, const Tensor& b)
{
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(
            "Cannot multiply tensors with different shapes"
        );
    }

    std::vector<float> result;
    result.reserve(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result.push_back(a.data()[i] * b.data()[i]);
    }

    return Tensor(std::move(result), a.shape());
}

