//28-10-2026


#include "Sequential.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    Tensor input(
        {1.0f, 2.0f,
         3.0f, 4.0f},
        {2, 2}
    );

    Tensor weights1(
        {1.0f,  0.0f,  1.0f,
         0.0f,  1.0f, -1.0f},
        {2, 3}
    );

    Tensor bias1({0.0f, 0.0f, 0.0f}, {3});

    Tensor weights2(
        { 1.0f,  1.0f,
          1.0f, -1.0f,
         -1.0f,  1.0f},
        {3, 2}
    );

    Tensor bias2({0.0f, 0.0f}, {2});

    Sequential network;

    network.add(LinearReLU(weights1, bias1));
    network.add(LinearReLU(weights2, bias2));

    Tensor output = network.forward(input);

    const std::vector<float> expected = {
        3.0f, 0.0f,
        7.0f, 0.0f
    };

    check(network.size() == 2, "Network should contain two layers");

    check(
        output.shape() == std::vector<std::size_t>{2, 2},
        "Output shape is incorrect"
    );

    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(
            std::fabs(output.data()[i] - expected[i]) < 1e-6f,
            "Sequential network produced an incorrect value"
        );
    }

    std::cout << "Sequential network test passed!\n";
    return 0;
}
