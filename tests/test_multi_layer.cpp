// 28-10-2026


#include "LinearReLU.h"

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
    // Twee voorbeelden met elk twee inputfeatures.
    Tensor input(
        {1.0f, 2.0f,
         3.0f, 4.0f},
        {2, 2}
    );

    // Eerste laag: 2 inputfeatures -> 3 outputfeatures.
    Tensor weights1(
        {1.0f,  0.0f,  1.0f,
         0.0f,  1.0f, -1.0f},
        {2, 3}
    );

    Tensor bias1({0.0f, 0.0f, 0.0f}, {3});

    LinearReLU layer1(weights1, bias1);

    // Tweede laag: 3 inputfeatures -> 2 outputfeatures.
    Tensor weights2(
        { 1.0f,  1.0f,
          1.0f, -1.0f,
         -1.0f,  1.0f},
        {3, 2}
    );

    Tensor bias2({0.0f, 0.0f}, {2});

    LinearReLU layer2(weights2, bias2);

    // Voer de input door beide lagen.
    Tensor hidden = layer1.forward(input);
    Tensor output = layer2.forward(hidden);

    // Verwachte output:
    // Eerste voorbeeld: [1, 2, -1] -> ReLU -> [1, 2, 0]
    // Tweede voorbeeld: [3, 4, -1] -> ReLU -> [3, 4, 0]
    //
    // Tweede laag:
    // [1, 2, 0] -> [3, 0]
    // [3, 4, 0] -> [7, 0]

    const std::vector<float> expected = {
        3.0f, 0.0f,
        7.0f, 0.0f
    };

    check(
        hidden.shape() == std::vector<std::size_t>{2, 3},
        "Hidden layer shape is incorrect"
    );

    check(
        output.shape() == std::vector<std::size_t>{2, 2},
        "Output layer shape is incorrect"
    );

    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(
            std::fabs(output.data()[i] - expected[i]) < 1e-6f,
            "Multi-layer network produced an incorrect value"
        );
    }

    std::cout << "Multi-layer network test passed!\n";
    return 0;
}
