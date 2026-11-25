//25-11-26


#include "GradientDescent.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        // Test 1: Optimizer aanmaken met learning rate 0.1
        GradientDescent optimizer(0.1f);

        check(
            std::fabs(optimizer.learning_rate() - 0.1f) < 1e-6f,
            "Incorrect learning rate"
        );

        std::cout << "GradientDescent construction test passed!\n";

        // Test 2: Optimizer voert een parameterupdate uit
        Sequential network;

        LinearReLU hidden_layer(
            Tensor({1.0f, 0.0f, 0.0f, 1.0f}, {2, 2}),
            Tensor({0.0f, -3.0f}, {2})
        );

        Linear output_layer(
            Tensor({2.0f, 3.0f}, {2, 1}),
            Tensor({0.0f}, {1})
        );

        network.add(std::move(hidden_layer));
        network.add(std::move(output_layer));

        Tensor input({1.0f, 2.0f}, {1, 2});

        Tensor prediction = network.forward(input);

        check(
            std::fabs(prediction.data()[0] - 2.0f) < 1e-5f,
            "Unexpected prediction before update"
        );

        // Gradient van de MSE-loss voor target 0
        Tensor loss_gradients({4.0f}, {1, 1});

        network.backward(loss_gradients);

        optimizer.step(network);

        // Controleer de voorspelling na de parameterupdate
        Tensor updated_prediction = network.forward(input);

        check(
            std::fabs(updated_prediction.data()[0] - (-0.4f)) < 1e-5f,
            "GradientDescent produced an incorrect update"
        );

        std::cout << "GradientDescent update test passed!\n";

        // Test 3: Ongeldige learning rates weigeren
        bool zero_rejected = false;
        bool negative_rejected = false;
        bool infinity_rejected = false;
        bool nan_rejected = false;

        try {
            GradientDescent invalid_optimizer(0.0f);
        }
        catch (const std::invalid_argument&) {
            zero_rejected = true;
        }

        try {
            GradientDescent invalid_optimizer(-0.1f);
        }
        catch (const std::invalid_argument&) {
            negative_rejected = true;
        }

        try {
            GradientDescent invalid_optimizer(
                std::numeric_limits<float>::infinity()
            );
        }
        catch (const std::invalid_argument&) {
            infinity_rejected = true;
        }

        try {
            GradientDescent invalid_optimizer(
                std::numeric_limits<float>::quiet_NaN()
            );
        }
        catch (const std::invalid_argument&) {
            nan_rejected = true;
        }

        check(zero_rejected, "Zero learning rate was accepted");
        check(negative_rejected, "Negative learning rate was accepted");
        check(infinity_rejected, "Infinite learning rate was accepted");
        check(nan_rejected, "NaN learning rate was accepted");

        std::cout << "GradientDescent validation test passed!\n";

        // Test 4: Step zonder berekende gradients
        // Test 4: Step zonder berekende gradients
        Sequential empty_gradient_network;

        Linear layer(
            Tensor({1.0f}, {1, 1}),
            Tensor({0.0f}, {1})
        );

        empty_gradient_network.add(std::move(layer));

        bool missing_gradients_rejected = false;

        try {
            optimizer.step(empty_gradient_network);
        }
        catch (const std::logic_error&) {
            missing_gradients_rejected = true;
        }

        check(
            missing_gradients_rejected,
            "Optimizer accepted missing gradients"
        );

        std::cout << "GradientDescent missing gradients test passed!\n";

        std::cout << "\nAll GradientDescent tests passed!\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
