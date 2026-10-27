//27-10-20: Gemaakt: Testen vooradd/ substract/ multiply/ matmul en transpose


#include "Tensor.h"
#include "TensorOps.h"

#include <cmath>
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

    bool approximatelyEqual(float a, float b)
    {
        return std::abs(a - b) < 0.0001f;
    }

    void testAdd()
    {
        Tensor a({1, 2, 3, 4}, {2, 2});
        Tensor b({10, 20, 30, 40}, {2, 2});

        Tensor result = add(a, b);

        check(result.data() == std::vector<float>{11, 22, 33, 44},
              "add calculates correct values");

        bool caught = false;

        try
        {
            Tensor wrongShape({1, 2}, {2});
            (void)add(a, wrongShape);
        }
        catch (const std::invalid_argument&)
        {
            caught = true;
        }

        check(caught, "add rejects different shapes");
    }

    void testSubtract()
    {
        Tensor a({10, 20, 30, 40}, {2, 2});
        Tensor b({1, 2, 3, 4}, {2, 2});

        Tensor result = subtract(a, b);

        check(result.data() == std::vector<float>{9, 18, 27, 36},
              "subtract calculates correct values");
    }

    void testMultiply()
    {
        Tensor a({1, 2, 3, 4}, {2, 2});
        Tensor b({5, 6, 7, 8}, {2, 2});

        Tensor result = multiply(a, b);

        check(result.data() == std::vector<float>{5, 12, 21, 32},
              "multiply calculates element-wise products");
    }

    void testMatmul()
    {
        Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});
        Tensor b({7, 8, 9, 10, 11, 12}, {3, 2});

        Tensor result = matmul(a, b);

        check(result.shape() == std::vector<std::size_t>{2, 2},
              "matmul returns correct shape");

        check(result.data() == std::vector<float>{58, 64, 139, 154},
              "matmul calculates correct values");

        bool caught = false;

        try
        {
            Tensor wrongShape({1, 2, 3, 4}, {2, 2});
            (void)matmul(a, wrongShape);
        }
        catch (const std::invalid_argument&)
        {
            caught = true;
        }

        check(caught, "matmul rejects incompatible dimensions");
    }

    void testTranspose()
    {
        Tensor a({1, 2, 3, 4, 5, 6}, {2, 3});

        Tensor result = transpose(a);

        check(result.shape() == std::vector<std::size_t>{3, 2},
              "transpose swaps dimensions");

        check(result.data() == std::vector<float>{1, 4, 2, 5, 3, 6},
              "transpose rearranges values correctly");

        bool caught = false;

        try
        {
            Tensor oneDimensional({1, 2, 3}, {3});
            (void)transpose(oneDimensional);
        }
        catch (const std::invalid_argument&)
        {
            caught = true;
        }

        check(caught, "transpose rejects non-2D tensors");
    }
}

int main()
{
    std::cout << "=== Tensor operation tests ===\n";

    testAdd();
    testSubtract();
    testMultiply();
    testMatmul();
    testTranspose();

    std::cout << "\nPassed: " << testsPassed
              << " | Failed: " << testsFailed << '\n';

    return testsFailed == 0 ? 0 : 1;
}
