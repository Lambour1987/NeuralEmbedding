//27-11-26


#include "Sequential.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void check_throws(Function function, const char* message) {
    bool thrown = false;

    try {
        function();
    }
    catch (const Exception&) {
        thrown = true;
    }

    check(thrown, message);
}

Sequential create_test_network() {
    Sequential network;

    network.add(LinearReLU(
        Tensor({1.0f, 2.0f}, {1, 2}),
        Tensor({0.1f, 0.2f}, {2})
    ));

    network.add(LinearLeakyReLU(
        Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}),
        Tensor({0.3f, 0.4f}, {2}),
        0.01f
    ));

    network.add(Linear(
        Tensor({1.0f, 2.0f}, {2, 1}),
        Tensor({0.5f}, {1})
    ));

    return network;
}

std::vector<LayerParameters> create_test_parameters() {
    return {
        {
            LayerType::ReLU,
            Tensor({10.0f, 20.0f}, {1, 2}),
            Tensor({1.0f, 2.0f}, {2}),
            0.0f
        },
        {
            LayerType::LeakyReLU,
            Tensor({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2}),
            Tensor({3.0f, 4.0f}, {2}),
            0.01f
        },
        {
            LayerType::Linear,
            Tensor({50.0f, 60.0f}, {2, 1}),
            Tensor({5.0f}, {1}),
            0.0f
        }
    };
}

void test_valid_parameter_restoration() {
    Sequential network = create_test_network();
    const auto parameters = create_test_parameters();

    network.set_parameters(parameters);

    check(
        network.weights(0).data() ==
            std::vector<float>({10.0f, 20.0f}),
        "First layer weights were not restored"
    );

    check(
        network.bias(0).data() ==
            std::vector<float>({1.0f, 2.0f}),
        "First layer bias was not restored"
    );

    check(
        network.weights(1).data() ==
            std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f}),
        "Second layer weights were not restored"
    );

    check(
        network.bias(1).data() ==
            std::vector<float>({3.0f, 4.0f}),
        "Second layer bias was not restored"
    );

    check(
        network.weights(2).data() ==
            std::vector<float>({50.0f, 60.0f}),
        "Output layer weights were not restored"
    );

    check(
        network.bias(2).data() ==
            std::vector<float>({5.0f}),
        "Output layer bias was not restored"
    );
}

void test_wrong_layer_count() {
    Sequential network = create_test_network();
    auto parameters = create_test_parameters();

    parameters.pop_back();

    check_throws<std::invalid_argument>(
        [&]() { network.set_parameters(parameters); },
        "Wrong layer count should be rejected"
    );
}

void test_wrong_layer_type() {
    Sequential network = create_test_network();
    auto parameters = create_test_parameters();

    parameters[2].type = LayerType::ReLU;

    check_throws<std::invalid_argument>(
        [&]() { network.set_parameters(parameters); },
        "Wrong layer type should be rejected"
    );
}

void test_wrong_tensor_shape() {
    Sequential network = create_test_network();
    auto parameters = create_test_parameters();

    parameters[0].weights = Tensor({1.0f}, {1, 1});

    check_throws<std::invalid_argument>(
        [&]() { network.set_parameters(parameters); },
        "Wrong tensor shape should be rejected"
    );
}

void test_nan_rejected() {
    Sequential network = create_test_network();
    auto parameters = create_test_parameters();

    parameters[0].weights = Tensor(
        {std::numeric_limits<float>::quiet_NaN(), 2.0f},
        {1, 2}
    );

    check_throws<std::invalid_argument>(
        [&]() { network.set_parameters(parameters); },
        "NaN weights should be rejected"
    );
}

void test_infinity_rejected() {
    Sequential network = create_test_network();
    auto parameters = create_test_parameters();

    parameters[1].bias = Tensor(
        {std::numeric_limits<float>::infinity(), 4.0f},
        {2}
    );

    check_throws<std::invalid_argument>(
        [&]() { network.set_parameters(parameters); },
        "Infinite bias should be rejected"
    );
}

void test_wrong_negative_slope() {
    Sequential network = create_test_network();
    auto parameters = create_test_parameters();

    parameters[1].negative_slope = 0.1f;

    check_throws<std::invalid_argument>(
        [&]() { network.set_parameters(parameters); },
        "Wrong LeakyReLU slope should be rejected"
    );
}

void test_failed_validation_does_not_change_network() {
    Sequential network = create_test_network();

    const auto original_weights_0 = network.weights(0).data();
    const auto original_weights_1 = network.weights(1).data();
    const auto original_weights_2 = network.weights(2).data();

    auto parameters = create_test_parameters();

    // De eerste twee lagen zijn geldig.
    // De laatste laag heeft een ongeldig type.
    parameters[2].type = LayerType::ReLU;

    check_throws<std::invalid_argument>(
        [&]() { network.set_parameters(parameters); },
        "Invalid parameters should be rejected"
    );

    check(
        network.weights(0).data() == original_weights_0,
        "First layer changed after failed validation"
    );

    check(
        network.weights(1).data() == original_weights_1,
        "Second layer changed after failed validation"
    );

    check(
        network.weights(2).data() == original_weights_2,
        "Third layer changed after failed validation"
    );
}

int main() {
    try {
        test_valid_parameter_restoration();
        test_wrong_layer_count();
        test_wrong_layer_type();
        test_wrong_tensor_shape();
        test_nan_rejected();
        test_infinity_rejected();
        test_wrong_negative_slope();
        test_failed_validation_does_not_change_network();

        std::cout << "Sequential parameter tests passed!\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
