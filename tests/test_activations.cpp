//28-10-2026


#include "..\include\activations.h"
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
        {-3.0f, -0.5f, 0.0f, 2.0f, 5.0f},
        {5}
    );

    Tensor output = relu(input);

    const std::vector<float> expected = {
        0.0f, 0.0f, 0.0f, 2.0f, 5.0f
    };

    check(
        output.shape() == input.shape(),
        "ReLU changed the tensor shape"
    );

    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(
            std::fabs(output.data()[i] - expected[i]) < 1e-6f,
            "ReLU produced an incorrect value"
        );
    }

    check(
        input.data()[0] == -3.0f,
        "ReLU modified the original tensor"
    );

    std::cout << "All Activation tests passed!\n";
    return 0;
}
