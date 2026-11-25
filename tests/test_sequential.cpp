
// 28-10-2026
// 24-11-2026

#include "Sequential.h"
#include "Loss.h"

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
    try {
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

    check(
        std::fabs(input_gradients.data()[0] - 2.0f) < 1e-6f,
        "First input gradient is incorrect"
    );

    check(
        std::fabs(input_gradients.data()[1] - 0.0f) < 1e-6f,
        "Second input gradient is incorrect"
    );

    // Test 3: volledige MSE -> Sequential backward integratie

    Tensor integration_input(
        {1.0f, 2.0f},
        {1, 2}
    );

    Sequential integration_network;

    integration_network.add(
        LinearReLU(
            Tensor(
                {1.0f, 0.0f,
                 0.0f, 1.0f},
                {2, 2}
            ),
            Tensor({0.0f, -3.0f}, {2})
        )
    );

    integration_network.add(
        Linear(
            Tensor({2.0f, 3.0f}, {2, 1}),
            Tensor({0.0f}, {1})
        )
    );

    // Forward pass
    Tensor predictions =
        integration_network.forward(integration_input);

    check(
        std::fabs(predictions.data()[0] - 2.0f) < 1e-6f,
        "Integration prediction is incorrect"
    );

    // Loss berekenen
    Tensor targets({0.0f}, {1,1});

    float loss = mean_squared_error(predictions, targets);

    check(
        std::fabs(loss - 4.0f) < 1e-6f,
        "Integration MSE loss is incorrect"
    );

    // Gradient van de loss berekenen
    Tensor loss_gradients =
        mean_squared_error_backward(predictions, targets);

    check(
        std::fabs(loss_gradients.data()[0] - 4.0f) < 1e-6f,
        "Integration loss gradient is incorrect"
    );

    // Backpropagation door het volledige netwerk
    Tensor integration_input_gradients =
        integration_network.backward(loss_gradients);

    check(
        std::fabs(integration_input_gradients.data()[0] - 8.0f) < 1e-6f,
        "Integration first input gradient is incorrect"
    );

    check(
        std::fabs(integration_input_gradients.data()[1] - 0.0f) < 1e-6f,
        "Integration second input gradient is incorrect"
    );

    // Gradients van verborgen laag controleren
    const Tensor& hidden_weight_gradients =
        integration_network.weight_gradients(0);

    const Tensor& hidden_bias_gradients =
        integration_network.bias_gradients(0);

    check(
        hidden_weight_gradients.shape() ==
            std::vector<std::size_t>{2, 2},
        "Hidden weight gradients have incorrect shape"
    );

    check(
    std::fabs(hidden_weight_gradients.data()[0] - 8.0f) < 1e-6f &&
    std::fabs(hidden_weight_gradients.data()[1] - 0.0f) < 1e-6f &&
    std::fabs(hidden_weight_gradients.data()[2] - 16.0f) < 1e-6f &&
    std::fabs(hidden_weight_gradients.data()[3] - 0.0f) < 1e-6f,
    "Hidden weight gradients are incorrect"
    );

    check(
        std::fabs(hidden_bias_gradients.data()[0] - 8.0f) < 1e-6f &&
        std::fabs(hidden_bias_gradients.data()[1] - 0.0f) < 1e-6f,
        "Hidden bias gradients are incorrect"
    );

    // Gradients van de outputlaag controleren
    const Tensor& output_weight_gradients =
        integration_network.weight_gradients(1);

    const Tensor& output_bias_gradients =
        integration_network.bias_gradients(1);

    check(
        output_weight_gradients.shape() ==
            std::vector<std::size_t>{2, 1},
        "Output weight gradients have incorrect shape"
    );

    check(
        std::fabs(output_weight_gradients.data()[0] - 4.0f) < 1e-6f &&
        std::fabs(output_weight_gradients.data()[1] - 0.0f) < 1e-6f,
        "Output weight gradients are incorrect"
    );

    check(
        std::fabs(output_bias_gradients.data()[0] - 4.0f) < 1e-6f,
        "Output bias gradient is incorrect"
    );

    // ==========================================
    // Test 4: Sequential Gradient Descent update
    // ==========================================

    integration_network.update_parameters(0.1f);

    // Forward pass na de parameterupdate
    Tensor updated_predictions =
        integration_network.forward(integration_input);

    check(
        std::fabs(updated_predictions.data()[0] - (-0.4f)) < 1e-5f,
        "Sequential Gradient Descent update produced an incorrect prediction"
    );


    // ==========================================
    // Test 5: Sequential met LinearLeakyReLU
    // ==========================================

    Tensor leaky_input(
        {1.0f, 2.0f},
        {1, 2}
    );

    Sequential leaky_network;

    leaky_network.add(
        LinearLeakyReLU(
            Tensor(
                {1.0f, 0.0f,
                 0.0f, 1.0f},
                {2, 2}
            ),
            Tensor({0.0f, -3.0f}, {2})
        )
    );

    leaky_network.add(
        Linear(
            Tensor({2.0f, 3.0f}, {2, 1}),
            Tensor({0.0f}, {1})
        )
    );

    // Forward: hidden output wordt [1.0, -0.01].
    Tensor leaky_predictions =
        leaky_network.forward(leaky_input);

    check(
        leaky_network.size() == 2,
        "Leaky network should contain two layers"
    );

    check(
        std::fabs(leaky_predictions.data()[0] - 1.97f) < 1e-6f,
        "Leaky network forward output is incorrect"
    );

    // Backpropagation
    Tensor leaky_output_gradients(
        {1.0f},
        {1, 1}
    );

    Tensor leaky_input_gradients =
        leaky_network.backward(leaky_output_gradients);

    check(
        std::fabs(leaky_input_gradients.data()[0] - 2.0f) < 1e-6f,
        "Leaky input gradient 0 is incorrect"
    );

    check(
        std::fabs(leaky_input_gradients.data()[1] - 0.03f) < 1e-6f,
        "Leaky input gradient 1 is incorrect"
    );

    // Hidden layer gradients controleren.
    const Tensor& leaky_weight_gradients =
        leaky_network.weight_gradients(0);

    const Tensor& leaky_bias_gradients =
        leaky_network.bias_gradients(0);

    check(
        std::fabs(leaky_weight_gradients.data()[0] - 2.0f) < 1e-6f &&
        std::fabs(leaky_weight_gradients.data()[1] - 0.03f) < 1e-6f &&
        std::fabs(leaky_weight_gradients.data()[2] - 4.0f) < 1e-6f &&
        std::fabs(leaky_weight_gradients.data()[3] - 0.06f) < 1e-6f,
        "Leaky hidden weight gradients are incorrect"
    );

    check(
        std::fabs(leaky_bias_gradients.data()[0] - 2.0f) < 1e-6f &&
        std::fabs(leaky_bias_gradients.data()[1] - 0.03f) < 1e-6f,
        "Leaky hidden bias gradients are incorrect"
    );

    std::cout << "Sequential LinearLeakyReLU test passed!\n";



    std::cout << "Sequential Gradient Descent update test passed!\n";

    std::cout << "Sequential forward test passed!\n";
    std::cout << "Sequential backward test passed!\n";
    std::cout << "MSE and Sequential integration test passed!\n";

            return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "TEST FAILED: " << error.what() << '\n';
        return 1;
    }
}