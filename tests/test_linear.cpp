//27-10-2026

#include "..\include\Linear.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main()
{
    // Twee invoervoorbeelden, elk met twee kenmerken.
    Tensor input(
        {1.0f, 2.0f,
         3.0f, 4.0f},
        {2, 2}
    );

    // Gewichten: 2 invoerkenmerken -> 2 uitvoerkenmerken.
    Tensor weights(
        {1.0f, 2.0f,
         3.0f, 4.0f},
        {2, 2}
    );

    Tensor bias({0.5f, -0.5f}, {2});

    Linear layer(weights, bias);
    Tensor output = layer.forward(input);

    check(
        output.shape() == std::vector<std::size_t>{2, 2},
        "Unexpected output shape"
    );

    const auto& values = output.data();

    check(std::fabs(values[0] - 7.5f) < 1e-5f,
          "Incorrect output [0,0]");
    check(std::fabs(values[1] - 9.5f) < 1e-5f,
          "Incorrect output [0,1]");
    check(std::fabs(values[2] - 15.5f) < 1e-5f,
          "Incorrect output [1,0]");
    check(std::fabs(values[3] - 21.5f) < 1e-5f,
          "Incorrect output [1,1]");

    std::cout << "All Linear tests passed!\n";
    return 0;
}
