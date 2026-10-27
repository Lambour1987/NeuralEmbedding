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

//27-10-2026: Transponeer matrix
inline Tensor transpose(const Tensor& a)
{
    if (a.shape().size() != 2) {
        throw std::invalid_argument(
            "transpose requires a 2D tensor"
        );
    }

    const std::size_t rows = a.shape()[0];
    const std::size_t cols = a.shape()[1];

    std::vector<float> result(rows * cols);

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result[j * rows + i] =
                a.data()[i * cols + j];
        }
    }

    return Tensor(std::move(result), {cols, rows});
}
