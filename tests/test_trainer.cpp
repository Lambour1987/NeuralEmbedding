//25-11-26
//25-11-26: Vervangen


//25-11-26

#include "Trainer.h"

#include <iomanip>
#include <iostream>
#include <vector>

// Maakt steeds hetzelfde netwerk met dezelfde beginparameters.
Sequential create_network()
{
    Sequential network;

    // Hidden layer 1: 1 -> 2 ReLU-neuronen
    network.add(
        LinearReLU(
            Tensor(
                {0.5f, -0.5f},
                {1, 2}
            ),
            Tensor({0.0f, 0.0f}, {2})
        )
    );

    // Hidden layer 2: 2 -> 2 ReLU-neuronen
    network.add(
        LinearReLU(
            Tensor(
                {0.5f, 0.0f,
                 0.0f, 0.5f},
                {2, 2}
            ),
            Tensor({0.0f, 0.0f}, {2})
        )
    );

    // Output layer: 2 -> 1 lineaire neuron
    network.add(
        Linear(
            Tensor({1.0f, 1.0f}, {2, 1}),
            Tensor({0.0f}, {1})
        )
    );

    return network;
}

// Print de voorspellingen na training.
void print_predictions(
    const Sequential& network,
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& targets
)
{
    std::cout << "\n"
              << std::setw(10) << "Input"
              << std::setw(15) << "Target"
              << std::setw(15) << "Prediction"
              << '\n';

    // forward is niet const, dus we gebruiken hier een kopie.
    Sequential prediction_network = network;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        Tensor prediction =
            prediction_network.forward(inputs[i]);

        std::cout << std::setw(10)
                  << inputs[i].data()[0]
                  << std::setw(15)
                  << targets[i].data()[0]
                  << std::setw(15)
                  << prediction.data()[0]
                  << '\n';
    }
}

int main()
{
    // Dataset: y = x^2
    const std::vector<Tensor> inputs = {
        Tensor({-2.0f}, {1, 1}),
        Tensor({-1.0f}, {1, 1}),
        Tensor({ 0.0f}, {1, 1}),
        Tensor({ 1.0f}, {1, 1}),
        Tensor({ 3.0f}, {1, 1})
    };

    const std::vector<Tensor> targets = {
        Tensor({4.0f}, {1, 1}),
        Tensor({1.0f}, {1, 1}),
        Tensor({0.0f}, {1, 1}),
        Tensor({1.0f}, {1, 1}),
        Tensor({9.0f}, {1, 1})
    };

    // Twee identieke netwerken met dezelfde beginparameters.
    Sequential online_network = create_network();
    Sequential mini_batch_network = create_network();

    // Learning rate blijft 0.1.
    GradientDescent optimizer(0.1f);

    Trainer trainer;

    constexpr std::size_t epochs = 100;
    constexpr std::size_t batch_size = 2;

    std::cout << std::fixed << std::setprecision(4);

    // ==========================================
    // 1. Online gradient descent
    // ==========================================

    std::cout << "\n====================================\n";
    std::cout << "       ONLINE GRADIENT DESCENT\n";
    std::cout << "====================================\n";

    const std::vector<float> online_loss =
        trainer.train(
            online_network,
            optimizer,
            inputs,
            targets,
            epochs
        );

    std::cout << "Initial loss: "
              << online_loss.front() << '\n';

    std::cout << "Final loss:   "
              << online_loss.back() << '\n';

    std::cout << "\nPredictions after online training:\n";

    print_predictions(
        online_network,
        inputs,
        targets
    );

    // ==========================================
    // 2. Mini-batch gradient descent
    // ==========================================

    std::cout << "\n====================================\n";
    std::cout << "       MINI-BATCH GRADIENT DESCENT\n";
    std::cout << "====================================\n";

    std::cout << "Batch size: " << batch_size << '\n';

    const std::vector<float> mini_batch_loss =
        trainer.train_mini_batch(
            mini_batch_network,
            optimizer,
            inputs,
            targets,
            epochs,
            batch_size
        );

    std::cout << "Initial loss: "
              << mini_batch_loss.front() << '\n';

    std::cout << "Final loss:   "
              << mini_batch_loss.back() << '\n';

    std::cout << "\nPredictions after mini-batch training:\n";

    print_predictions(
        mini_batch_network,
        inputs,
        targets
    );

    // ==========================================
    // 3. Vergelijking
    // ==========================================

    std::cout << "\n====================================\n";
    std::cout << "             COMPARISON\n";
    std::cout << "====================================\n";

    std::cout << "Online final loss:      "
              << online_loss.back() << '\n';

    std::cout << "Mini-batch final loss:  "
              << mini_batch_loss.back() << '\n';

    std::cout << "\nTraining comparison completed!\n";

    return 0;
}
