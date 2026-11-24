//28-10-2026
//24-11-2026 volledig aangepast

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
    // Test 1: MSE berekening

    Tensor predictions({3.0f, 4.0f, 6.0f}, {3});
    Tensor targets({5.0f, 2.0f, 7.0f}, {3});

    const float loss = mean_squared_error(predictions, targets);

    check(
        std::fabs(loss - 3.0f) < 1e-6f,
        "MSE calculation is incorrect"
    );

    // Test 2: Perfect predictions

    Tensor perfect_predictions({1.0f, 2.0f}, {2});
    Tensor perfect_targets({1.0f, 2.0f}, {2});

    check(
        mean_squared_error(
            perfect_predictions,
            perfect_targets
        ) == 0.0f,
        "Perfect predictions should have zero loss"
    );

    // Test 3: Ongelijke shapes afwijzen

    bool threw = false;

    try {
        Tensor wrong_shape({1.0f, 2.0f}, {2});
        Tensor other_shape({1.0f, 2.0f}, {1, 2});

        mean_squared_error(wrong_shape, other_shape);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    check(threw, "MSE should reject mismatched shapes");

    // Test 4: MSE backward berekening

    Tensor backward_predictions(
        {2.0f, 4.0f},
        {2}
    );

    Tensor backward_targets(
        {1.0f, 2.0f},
        {2}
    );

    Tensor gradients = mean_squared_error_backward(
        backward_predictions,
        backward_targets
    );

    const std::vector<float> expected_gradients = {
        1.0f,
        2.0f
    };

    check(
        gradients.shape() == std::vector<std::size_t>{2},
        "MSE gradients have incorrect shape"
    );

    for (std::size_t i = 0; i < expected_gradients.size(); ++i) {
        check(
            std::fabs(
                gradients.data()[i] - expected_gradients[i]
            ) < 1e-6f,
            "MSE backward calculation is incorrect"
        );
    }

    // Test 5: Perfect predictions moeten nul gradients opleveren

    Tensor perfect_gradients = mean_squared_error_backward(
        perfect_predictions,
        perfect_targets
    );

    for (std::size_t i = 0; i < perfect_gradients.size(); ++i) {
        check(
            std::fabs(perfect_gradients.data()[i]) < 1e-6f,
            "Perfect predictions should have zero gradients"
        );
    }

    std::cout << "Loss tests passed!\n";
    return 0;
}
