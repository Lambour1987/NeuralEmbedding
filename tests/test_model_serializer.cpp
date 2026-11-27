//27-11-26


#include "ModelSerializer.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cerrno>

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

bool tensors_equal(const Tensor& first, const Tensor& second) {
    return first.shape() == second.shape() &&
           first.data() == second.data();
}

void test_save_and_load() {
    constexpr const char* filename =
        "test_model_checkpoint.nnckpt";

    ModelSerializer serializer;

    Sequential original = create_test_network();
    Sequential restored = create_test_network();

    Tensor input({2.0f}, {1, 1});

    const Tensor expected = original.forward(input);

    serializer.save(original, filename);

    // Verander de parameters van het tweede netwerk.
    auto changed_parameters = std::vector<LayerParameters>{
        {
            LayerType::ReLU,
            Tensor({10.0f, 20.0f}, {1, 2}),
            Tensor({10.0f, 20.0f}, {2}),
            0.0f
        },
        {
            LayerType::LeakyReLU,
            Tensor({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2}),
            Tensor({10.0f, 20.0f}, {2}),
            0.01f
        },
        {
            LayerType::Linear,
            Tensor({50.0f, 60.0f}, {2, 1}),
            Tensor({50.0f}, {1}),
            0.0f
        }
    };

    restored.set_parameters(changed_parameters);

    serializer.load(restored, filename);

    const Tensor actual = restored.forward(input);

    check(
        tensors_equal(expected, actual),
        "Predictions differ after loading checkpoint"
    );

    for (std::size_t i = 0; i < original.size(); ++i) {
        check(
            tensors_equal(original.weights(i), restored.weights(i)),
            "Weights differ after loading checkpoint"
        );

        check(
            tensors_equal(original.bias(i), restored.bias(i)),
            "Bias differs after loading checkpoint"
        );
    }

    check(
        original.layer_type(1) == restored.layer_type(1),
        "Layer type differs after loading checkpoint"
    );

    check(
        original.negative_slope(1) == restored.negative_slope(1),
        "LeakyReLU slope differs after loading checkpoint"
    );

    std::remove(filename);
}

void test_missing_file() {
    ModelSerializer serializer;
    Sequential network = create_test_network();

    check_throws<std::runtime_error>(
        [&]() {
            serializer.load(
                network,
                "this_checkpoint_does_not_exist.nnckpt"
            );
        },
        "Missing checkpoint should be rejected"
    );
}

void test_wrong_architecture() {
    constexpr const char* filename =
        "test_model_checkpoint_architecture.nnckpt";

    ModelSerializer serializer;

    Sequential original = create_test_network();
    Sequential incompatible;

    incompatible.add(Linear(
        Tensor({1.0f}, {1, 1}),
        Tensor({0.0f}, {1})
    ));

    serializer.save(original, filename);

    const auto original_weights =
        incompatible.weights(0).data();

    check_throws<std::invalid_argument>(
        [&]() {
            serializer.load(incompatible, filename);
        },
        "Incompatible architecture should be rejected"
    );

    check(
        incompatible.weights(0).data() == original_weights,
        "Network changed after incompatible checkpoint was rejected"
    );

    std::remove(filename);
}

void test_corrupted_header() {
    constexpr const char* filename =
        "test_model_checkpoint_corrupt.nnckpt";

    {
        FILE* file = nullptr;
        errno_t error = fopen_s(&file, filename, "wb");

        check(error == 0 && file != nullptr,
      "Could not create corrupted test file");

        check(file != nullptr, "Could not create corrupted test file");

        const char invalid_header[] = "INVALID!";

        const std::size_t written = std::fwrite(
            invalid_header,
            1,
            sizeof(invalid_header) - 1,
            file
        );

        std::fclose(file);

        check(
            written == sizeof(invalid_header) - 1,
            "Could not write corrupted test header"
        );
    }

    ModelSerializer serializer;
    Sequential network = create_test_network();

    check_throws<std::runtime_error>(
        [&]() {
            serializer.load(network, filename);
        },
        "Corrupted checkpoint header should be rejected"
    );

    std::remove(filename);
}

int main() {
    try {
        test_save_and_load();
        test_missing_file();
        test_wrong_architecture();
        test_corrupted_header();

        std::cout << "ModelSerializer tests passed!\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Test failed: "
                  << error.what() << '\n';

        return 1;
    }
}
