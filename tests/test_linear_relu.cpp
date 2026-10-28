//28-10-2026


#include "LinearReLu.h"

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
    // Twee inputvoorbeelden, elk met twee features.
    Tensor input(
        {1.0f, 2.0f,
         3.0f, 4.0f},
        {2, 2}
    );

    // Twee inputfeatures en twee outputfeatures.
    Tensor weights(
        {1.0f, -1.0f,
         2.0f,  1.0f},
        {2, 2}
    );

    // Eén bias per outputfeature.
    Tensor bias(
        {0.0f, -1.0f},
        {2}
    );

    LinearReLU layer(weights, bias);
    Tensor output = layer.forward(input);

    // Verwachte Linear-output:
    // Rij 1: [5, -1]
    // Rij 2: [11, -3]
    //
    // Na ReLU:
    // Rij 1: [5, 0]
    // Rij 2: [11, 0]

    const std::vector<float> expected = {
        5.0f, 0.0f,
        11.0f, 0.0f
    };

    check(
        output.shape() == std::vector<std::size_t>{2, 2},
        "Output shape is incorrect"
    );

    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(
            std::fabs(output.data()[i] - expected[i]) < 1e-6f,
            "Linear + ReLU produced an incorrect value"
        );
    }

    std::cout << "Linear + ReLU test passed!\n";
    return 0;
}
