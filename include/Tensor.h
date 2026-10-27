//16-10-2026

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

class Tensor {
public:
    Tensor(std::vector<float> data,
           std::vector<std::size_t> shape)
        : data_(std::move(data)),
          shape_(std::move(shape))
    {
        validate();
        calculate_strides();
    }

    float at(const std::vector<std::size_t>& indices) const
    {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument(
                "Number of indices does not match tensor dimensions"
            );
        }

        std::size_t offset = 0;

        for (std::size_t d = 0; d < shape_.size(); ++d) {
            if (indices[d] >= shape_[d]) {
                throw std::out_of_range(
                    "Tensor index out of range"
                );
            }

            offset += indices[d] * strides_[d];
        }

        return data_[offset];
    }

    const std::vector<std::size_t>& shape() const
    {
        return shape_;
    }

    const std::vector<std::size_t>& strides() const
    {
        return strides_;
    }
    
    const std::vector<float>& data() const
    {
        return data_;
    }

    std::size_t size() const
    {
        return data_.size();
    }

private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

    void validate() const
    {
        std::size_t expected = 1;

        for (const std::size_t dim : shape_) {
            if (dim != 0 &&
                expected >
                    std::numeric_limits<std::size_t>::max() / dim) {
                throw std::overflow_error(
                    "Tensor shape exceeds maximum size"
                );
            }

            expected *= dim;
        }

        if (expected != data_.size()) {
            throw std::invalid_argument(
                "Tensor data size does not match shape"
            );
        }
    }

    void calculate_strides()
    {
        strides_.resize(shape_.size());

        std::size_t stride = 1;

        for (std::size_t d = shape_.size(); d > 0; --d) {
            const std::size_t i = d - 1;

            strides_[i] = stride;

            if (shape_[i] != 0 &&
                stride >
                    std::numeric_limits<std::size_t>::max()
                        / shape_[i]) {
                throw std::overflow_error(
                    "Tensor stride exceeds maximum size"
                );
            }

            stride *= shape_[i];
        }
    }
};

//27-10-2026: Matrix Multiplication: 
// Kijk nog even naar float: dit heeft te maken dat een computer geen 1/3 kan doen maar wel 0.333

inline Tensor matmul(const Tensor& a, const Tensor& b)
{
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::invalid_argument(
            "matmul requires two 2D tensors"
        );
    }

    const std::size_t rowsA = a.shape()[0];
    const std::size_t colsA = a.shape()[1];
    const std::size_t rowsB = b.shape()[0];
    const std::size_t colsB = b.shape()[1];

    if (colsA != rowsB) {
        throw std::invalid_argument(
            "Incompatible matrix dimensions for matmul"
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

