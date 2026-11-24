//28-10-2026
//24-11-2026


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
    // Test 1: bestaande Sequential forward-test

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

    // Test 2: Sequential met LinearReLU en Linear

    Tensor backward_input(
        {1.0f, 2.0f},
        {1, 2}
    );

    Tensor hidden_weights(
        {1.0f, 0.0f,
         0.0f, 1.0f},
        {2, 2}
    );

    Tensor hidden_bias(
        {0.0f, -3.0f},
        {2}
    );

    Tensor output_weights(
        {2.0f, 3.0f},
        {2, 1}
    );

    Tensor output_bias(
        {0.0f},
        {1}
    );

    Sequential backward_network;

    backward_network.add(
        LinearReLU(hidden_weights, hidden_bias)
    );

    backward_network.add(
        Linear(output_weights, output_bias)
    );

    Tensor backward_output =
        backward_network.forward(backward_input);

    check(
        backward_network.size() == 2,
        "Backward network should contain two layers"
    );

    check(
        backward_output.shape() ==
            std::vector<std::size_t>{1, 1},
        "Backward network output shape is incorrect"
    );

    check(
        std::fabs(backward_output.data()[0] - 2.0f) < 1e-6f,
        "Backward network forward output is incorrect"
    );

    // Gradient afkomstig van de loss-functie
    Tensor output_gradients(
        {1.0f},
        {1, 1}
    );

    Tensor input_gradients =
        backward_network.backward(output_gradients);

    check(
        input_gradients.shape() ==
            std::vector<std::size_t>{1, 2},
        "Input gradients have incorrect shape"
    );

    // Verwachte input gradients: [2, 0]
    check(
        std::fabs(input_gradients.data()[0] - 2.0f) < 1e-6f,
        "First input gradient is incorrect"
    );

    check(
        std::fabs(input_gradients.data()[1] - 0.0f) < 1e-6f,
        "Second input gradient is incorrect"
    );

    std::cout << "Sequential forward test passed!\n";
    std::cout << "Sequential backward test passed!\n";

    return 0;
}
