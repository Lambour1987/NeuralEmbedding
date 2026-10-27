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
}

int main()
{
    std::cout << "=== Tensor tests ===\n";

    testTensorShapeAndStrides();
    testTensorIndexing();

    std::cout << "\nPassed: " << testsPassed
              << " | Failed: " << testsFailed << '\n';

    return testsFailed == 0 ? 0 : 1;
}
