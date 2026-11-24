//24-11-26
//We gaan daarin afzonderlijk testen:
// 1. De vorm en waarden van dW.
// 2. De vorm en waarden van db.
// 3. De vorm en waarden van dX.
// 4. Een ongeldige invoer.
// 5. Het opvragen van gradients voordat backward() is uitgevoerd.


#include "../include/Linear.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

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
    // Testdata
    Tensor input(
        {1, 2, 3, 4},
        {2, 2}
    );

    Tensor weights(
        {1, 2, 3, 4},
        {2, 2}
    );

    Tensor bias(
        {0, 0},
        {2}
    );

    Tensor output_gradients(
        {1, 2, 3, 4},
        {2, 2}
    );

    Linear layer(weights, bias);

    // Backward uitvoeren
    Tensor input_gradients =
        layer.backward(input, output_gradients);

    // Test dW
    const Tensor& weight_gradients =
        layer.weight_gradients();

    check(
        weight_gradients.shape() == std::vector<std::size_t>{2, 2},
        "Weight gradient shape is incorrect"
    );

    check_near(weight_gradients.data()[0], 10, "dW[0] incorrect");
    check_near(weight_gradients.data()[1], 14, "dW[1] incorrect");
    check_near(weight_gradients.data()[2], 14, "dW[2] incorrect");
    check_near(weight_gradients.data()[3], 20, "dW[3] incorrect");

    // Test db
    const Tensor& bias_gradients =
        layer.bias_gradients();

    check(
        bias_gradients.shape() == std::vector<std::size_t>{2},
        "Bias gradient shape is incorrect"
    );

    check_near(bias_gradients.data()[0], 4, "db[0] incorrect");
    check_near(bias_gradients.data()[1], 6, "db[1] incorrect");

    // Test dX
    check(
        input_gradients.shape() == std::vector<std::size_t>{2, 2},
        "Input gradient shape is incorrect"
    );

    check_near(input_gradients.data()[0], 5, "dX[0] incorrect");
    check_near(input_gradients.data()[1], 11, "dX[1] incorrect");
    check_near(input_gradients.data()[2], 11, "dX[2] incorrect");
    check_near(input_gradients.data()[3], 25, "dX[3] incorrect");

    std::cout << "All Linear backward tests passed!\n";

    return 0;
}
