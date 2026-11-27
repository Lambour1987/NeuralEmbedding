//27-11-26


#include "Evaluator.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Maakt een eenvoudig netwerk: y = 2x.
Sequential create_test_network() {
    Sequential network;

    network.add(
        Linear(
            Tensor({2.0f}, {1, 1}),
            Tensor({0.0f}, {1})
        )
    );

    return network;
}

int main() {
    Evaluator evaluator;
    Sequential network = create_test_network();

    // Testdataset:
    // Input 1 -> voorspelling 2, target 1
    // Input 2 -> voorspelling 4, target 2

    const std::vector<Tensor> inputs = {
        Tensor({1.0f}, {1, 1}),
        Tensor({2.0f}, {1, 1})
    };

    const std::vector<Tensor> targets = {
        Tensor({1.0f}, {1, 1}),
        Tensor({2.0f}, {1, 1})
    };

    // Test 1: Correcte gemiddelde MSE.
    // ((2 - 1)^2 + (4 - 2)^2) / 2 = 2.5

    const float loss = evaluator.evaluate_mse(
        network,
        inputs,
        targets
    );

    check(
        std::fabs(loss - 2.5f) < 1e-6f,
        "Evaluator calculated an incorrect MSE"
    );

    // Test 2: Lege dataset afwijzen.

    bool threw = false;

    try {
        const std::vector<Tensor> empty_inputs;
        const std::vector<Tensor> empty_targets;

        evaluator.evaluate_mse(
            network,
            empty_inputs,
            empty_targets
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        "Evaluator should reject an empty dataset"
    );

    // Test 3: Verschillende aantallen inputs en targets afwijzen.

    threw = false;

    try {
        const std::vector<Tensor> one_target = {
            Tensor({1.0f}, {1, 1})
        };

        evaluator.evaluate_mse(
            network,
            inputs,
            one_target
        );
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    check(
        threw,
        "Evaluator should reject mismatched dataset sizes"
    );

    // Test 4: Een leeg netwerk afwijzen.

    threw = false;

    try {
        Sequential empty_network;

        evaluator.evaluate_mse(
            empty_network,
            inputs,
            targets
        );
    } catch (const std::logic_error&) {
        threw = true;
    }

    check(
        threw,
        "Evaluator should reject an empty network"
    );

    // Test 5: Evalueren mag de netwerkparameters niet veranderen.
    // We controleren dit indirect via de voorspelling.

    Tensor prediction_before = network.forward(inputs[0]);

    evaluator.evaluate_mse(
        network,
        inputs,
        targets
    );

    Tensor prediction_after = network.forward(inputs[0]);

    check(
        std::fabs(
            prediction_before.data()[0] -
            prediction_after.data()[0]
        ) < 1e-6f,
        "Evaluation should not change network predictions"
    );

    std::cout << "Evaluator tests passed!\n";

    return 0;
}
