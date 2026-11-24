//27-10-2026

//27-10-2026

#include "..\include\Linear.h"

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

int main()
{
    try {
        // ==========================================
        // TEST 1: Linear forward
        // ==========================================

        Tensor input(
            {1.0f, 2.0f,
             3.0f, 4.0f},
            {2, 2}
        );

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

        std::cout << "Linear forward test passed!\n";


        // ==========================================
        // TEST 2: Update zonder backward moet falen
        // ==========================================

        Linear layer_without_gradients(weights, bias);

        bool missing_gradients_thrown = false;

        try {
            layer_without_gradients.update_parameters(0.1f);
        }
        catch (const std::logic_error&) {
            missing_gradients_thrown = true;
        }

        check(
            missing_gradients_thrown,
            "Update should fail without gradients"
        );

        std::cout << "Missing gradients test passed!\n";


        // ==========================================
        // TEST 3: Gradient Descent parameter update
        // ==========================================

        Tensor update_input(
            {1.0f, 2.0f},
            {1, 2}
        );

        Tensor output_gradients(
            {0.5f, 1.0f},
            {1, 2}
        );

        Linear trainable_layer(weights, bias);

        trainable_layer.backward(
            update_input,
            output_gradients
        );

        // Ongeldige learning rate moet worden geweigerd.
        bool invalid_learning_rate_thrown = false;

        try {
            trainable_layer.update_parameters(0.0f);
        }
        catch (const std::invalid_argument&) {
            invalid_learning_rate_thrown = true;
        }

        check(
            invalid_learning_rate_thrown,
            "Zero learning rate should be rejected"
        );

        // Learning rate = 0.1
        trainable_layer.update_parameters(0.1f);

        const auto& updated_weights =
            trainable_layer.weights().data();

        const auto& updated_bias =
            trainable_layer.bias().data();

        // Verwachte weights:
        // [0.95, 1.90, 2.90, 3.80]

        check(std::fabs(updated_weights[0] - 0.95f) < 1e-5f,
              "Incorrect updated weight [0]");
        check(std::fabs(updated_weights[1] - 1.90f) < 1e-5f,
              "Incorrect updated weight [1]");
        check(std::fabs(updated_weights[2] - 2.90f) < 1e-5f,
              "Incorrect updated weight [2]");
        check(std::fabs(updated_weights[3] - 3.80f) < 1e-5f,
              "Incorrect updated weight [3]");

        // Verwachte biases:
        // [0.45, -0.60]

        check(std::fabs(updated_bias[0] - 0.45f) < 1e-5f,
              "Incorrect updated bias [0]");
        check(std::fabs(updated_bias[1] - (-0.60f)) < 1e-5f,
              "Incorrect updated bias [1]");

        std::cout << "Gradient Descent update test passed!\n";

        std::cout << "\nAll Linear tests passed!\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "TEST FAILED: "
                  << error.what() << '\n';

        return 1;
    }
}
