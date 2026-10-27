//27-10-2026: Tensortests: Shape, strides, elementaantal en indexgrenzen

#include "Tensor.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    int testsPassed = 0;
    int testsFailed = 0;

    void check(bool condition, const std::string& name)
    {
        if (condition)
        {
            ++testsPassed;
            std::cout << "[PASS] " << name << '\n';
        }
        else
        {
            ++testsFailed;
            std::cout << "[FAIL] " << name << '\n';
        }
    }

    void testTensorShapeAndStrides()
    {
        Tensor tensor({1, 2, 3, 4, 5, 6}, {2, 3});

        check(tensor.shape() == std::vector<std::size_t>{2, 3},
              "Correct tensor shape");

        check(tensor.strides() == std::vector<std::size_t>{3, 1},
              "Correct tensor strides");

        check(tensor.size() == 6, "Correct element count");
    }

    void testTensorIndexing()
    {
        Tensor tensor({1, 2, 3, 4, 5, 6}, {2, 3});

        check(tensor.at({0, 0}) == 1, "First element");
        check(tensor.at({1, 2}) == 6, "Last element");

        bool caught = false;

        try
        {
            (void)tensor.at({2, 0});
        }
        catch (const std::out_of_range&)
        {
            caught = true;
        }

        check(caught, "Out-of-range index rejected");
    }
        
    void testInvalidTensorConstruction()
    {
        bool caught = false;

        try {
            Tensor invalid({1, 2, 3}, {2, 2});
        }
        catch (const std::invalid_argument&) {
            caught = true;
        }

        check(caught, "Mismatched data and shape are rejected");
    }

    void testInvalidIndexCount()
    {
        Tensor tensor({1, 2, 3, 4}, {2, 2});

        bool caught = false;

        try {
            (void)tensor.at({0});
        }
        catch (const std::invalid_argument&) {
            caught = true;
        }

        check(caught, "Incorrect number of indices is rejected");
    }

    void testZeroSizedTensor()
    {
        Tensor tensor({}, {2, 0, 3});

        check(tensor.size() == 0, "Zero-sized tensor has no elements");

        check(
            tensor.shape() == std::vector<std::size_t>{2, 0, 3},
            "Zero-sized tensor preserves its shape"
        );
    }
}

int main()
{
    std::cout << "=== Tensor tests ===\n";

    testTensorShapeAndStrides();
    testTensorIndexing();

    std::cout << "\nPassed: " << testsPassed
              << " | Failed: " << testsFailed << '\n';

    testInvalidTensorConstruction();
    testInvalidIndexCount();
    testZeroSizedTensor();

    return testsFailed == 0 ? 0 : 1;
}
