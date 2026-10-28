//28-10-2026


#include "Loss.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    Tensor predictions({3.0f, 4.0f, 6.0f}, {3});
    Tensor targets({5.0f, 2.0f, 7.0f}, {3});

    const float loss = mean_squared_error(predictions, targets);

    check(
        std::fabs(loss - 3.0f) < 1e-6f,
        "MSE calculation is incorrect"
    );

    Tensor perfect_predictions({1.0f, 2.0f}, {2});
    Tensor perfect_targets({1.0f, 2.0f}, {2});

    check(
        mean_squared_error(perfect_predictions, perfect_targets) == 0.0f,
        "Perfect predictions should have zero loss"
    );

    bool threw = false;

    try {
        Tensor wrong_shape({1.0f, 2.0f}, {2});
        Tensor other_shape({1.0f, 2.0f}, {1, 2});

        mean_squared_error(wrong_shape, other_shape);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    check(threw, "MSE should reject mismatched shapes");

    std::cout << "Loss tests passed!\n";
    return 0;
}
