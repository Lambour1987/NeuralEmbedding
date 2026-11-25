//25-11-26


#include "Trainer.h"

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
    try {
        // Test 1: Eén trainingsstap
        Sequential single_step_network;

        single_step_network.add(
            LinearReLU(
                Tensor(
                    {1.0f, 0.0f,
                     0.0f, 1.0f},
                    {2, 2}
                ),
                Tensor({0.0f, -3.0f}, {2})
            )
        );

        single_step_network.add(
            Linear(
                Tensor({2.0f, 3.0f}, {2, 1}),
                Tensor({0.0f}, {1})
            )
        );

        GradientDescent optimizer(0.1f);
        Trainer trainer;

        Tensor input({1.0f, 2.0f}, {1, 2});
        Tensor target({0.0f}, {1, 1});

        const float initial_loss = trainer.train_step(
            single_step_network,
            optimizer,
            input,
            target
        );

        check(
            std::fabs(initial_loss - 4.0f) < 1e-5f,
            "Single training step produced an incorrect loss"
        );

        Tensor updated_prediction =
            single_step_network.forward(input);

        check(
            std::fabs(updated_prediction.data()[0] - (-0.4f)) < 1e-5f,
            "Single training step produced an incorrect update"
        );

        std::cout << "Trainer single step test passed!\n";

        // Test 2: Training dataset
        std::vector<Tensor> inputs = {
            Tensor({-2.0f}, {1, 1}),
            Tensor({-1.0f}, {1, 1}),
            Tensor({0.0f}, {1, 1}),
            Tensor({1.0f}, {1, 1}),
            Tensor({3.0f}, {1, 1})
        };

        std::vector<Tensor> targets = {
            Tensor({4.0f}, {1, 1}),
            Tensor({1.0f}, {1, 1}),
            Tensor({0.0f}, {1, 1}),
            Tensor({1.0f}, {1, 1}),
            Tensor({9.0f}, {1, 1})
        };

        // Vaste netwerkarchitectuur:
        // 1 input -> 2 ReLU -> 2 ReLU -> 1 output
        Sequential training_network;

        training_network.add(
            LinearReLU(
                Tensor(
                    {0.5f, -0.5f},
                    {1, 2}
                ),
                Tensor({0.0f, 0.0f}, {2})
            )
        );

        training_network.add(
            LinearReLU(
                Tensor(
                    {0.5f, 0.0f,
                     0.0f, 0.5f},
                    {2, 2}
                ),
                Tensor({0.0f, 0.0f}, {2})
            )
        );

        training_network.add(
            Linear(
                Tensor({1.0f, 1.0f}, {2, 1}),
                Tensor({0.0f}, {1})
            )
        );

        const std::vector<float> loss_history = trainer.train(
            training_network,
            optimizer,
            inputs,
            targets,
            100
        );

        check(
            loss_history.size() == 100,
            "Loss history should contain one value per epoch"
        );

        for (const float loss : loss_history) {
            check(
                std::isfinite(loss),
                "Training produced a non-finite loss"
            );
        }

        check(
            loss_history.back() < loss_history.front(),
            "Training loss did not decrease"
        );

        std::cout << "Trainer dataset training test passed!\n";
        std::cout << "Initial epoch loss: "
                  << loss_history.front() << '\n';
        std::cout << "Final epoch loss: "
                  << loss_history.back() << '\n';

        // Test 3: Ongeldige datasets
        bool empty_dataset_rejected = false;
        bool mismatched_dataset_rejected = false;
        bool zero_epochs_rejected = false;

        try {
            trainer.train(
                training_network,
                optimizer,
                {},
                {},
                10
            );
        }
        catch (const std::invalid_argument&) {
            empty_dataset_rejected = true;
        }

        try {
            trainer.train(
                training_network,
                optimizer,
                inputs,
                {},
                10
            );
        }
        catch (const std::invalid_argument&) {
            mismatched_dataset_rejected = true;
        }

        try {
            trainer.train(
                training_network,
                optimizer,
                inputs,
                targets,
                0
            );
        }
        catch (const std::invalid_argument&) {
            zero_epochs_rejected = true;
        }

        check(
            empty_dataset_rejected,
            "Empty dataset was accepted"
        );

        check(
            mismatched_dataset_rejected,
            "Mismatched dataset was accepted"
        );

        check(
            zero_epochs_rejected,
            "Zero epochs were accepted"
        );

        std::cout << "Trainer validation test passed!\n";

        std::cout << "\nAll Trainer tests passed!\n";
    }
    catch (const std::exception& error) {
        std::cerr << "Test failed: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
