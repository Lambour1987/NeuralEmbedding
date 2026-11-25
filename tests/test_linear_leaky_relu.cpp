//25-11-26: nieuw aangemaakt


#include "../include/LinearLeakyReLU.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_near(
    float actual,
    float expected,
    const char* message
)
{
    constexpr float tolerance = 0.0001f;

    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

int main()
{
    // Input: [1, 2]
    Tensor input({1, 2}, {1, 2});

    // Identiteitsmatrix als gewichten.
    Tensor weights({1, 0, 0, 1}, {2, 2});

    Tensor bias({0, -3}, {2});

    LinearLeakyReLU layer(weights, bias);

    // Linear output: [1, -1]
    // Leaky ReLU output: [1, -0.01]
    Tensor output = layer.forward(input);

    check_near(output.data()[0], 1, "Forward output 0 incorrect");
    check_near(output.data()[1], -0.01f, "Forward output 1 incorrect");

    // Upstream gradients: [2, 3]
    Tensor output_gradients({2, 3}, {1, 2});

    Tensor input_gradients = layer.backward(output_gradients);

    // Leaky ReLU derivative: [1, 0.01]
    // dZ = [2, 0.03]
    // dX = [2, 0.03]
    check(
        input_gradients.shape() == std::vector<std::size_t>{1, 2},
        "Input gradient shape incorrect"
    );

    check_near(input_gradients.data()[0], 2, "dX[0] incorrect");
    check_near(input_gradients.data()[1], 0.03f, "dX[1] incorrect");

    // dW = [[2, 0.03], [4, 0.06]]
    const Tensor& weight_gradients = layer.weight_gradients();

    check_near(weight_gradients.data()[0], 2, "dW[0] incorrect");
    check_near(weight_gradients.data()[1], 0.03f, "dW[1] incorrect");
    check_near(weight_gradients.data()[2], 4, "dW[2] incorrect");
    check_near(weight_gradients.data()[3], 0.06f, "dW[3] incorrect");

    const Tensor& bias_gradients = layer.bias_gradients();

    check_near(bias_gradients.data()[0], 2, "db[0] incorrect");
    check_near(bias_gradients.data()[1], 0.03f, "db[1] incorrect");

    std::cout << "All LinearLeakyReLU backward tests passed!\n";

    return 0;
}
