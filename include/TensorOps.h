//19-10-2026


#pragma once
#include "Tensor.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>


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


inline Tensor matmul(const Tensor& a, const Tensor& b)
{
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw std::invalid_argument("matmul requires two 2D tensors");

    const std::size_t rowsA = a.shape()[0];
    const std::size_t colsA = a.shape()[1];
    const std::size_t rowsB = b.shape()[0];
    const std::size_t colsB = b.shape()[1];

    if (colsA != rowsB)
        throw std::invalid_argument(
            "Incompatible matrix dimensions for matmul"
        );

    if (colsB != 0 &&
        rowsA > std::numeric_limits<std::size_t>::max() / colsB) {
        throw std::overflow_error(
            "matmul result dimensions exceed maximum size"
        );
    }

    std::vector<float> result(rowsA * colsB, 0.0f);

    const auto& dataA = a.data();
    const auto& dataB = b.data();

    for (std::size_t i = 0; i < rowsA; ++i) {
        for (std::size_t k = 0; k < colsA; ++k) {
            const float valueA = dataA[i * colsA + k];

            for (std::size_t j = 0; j < colsB; ++j) {
                result[i * colsB + j] +=
                    valueA * dataB[k * colsB + j];
            }
        }
    }

    return Tensor(std::move(result), {rowsA, colsB});
}

