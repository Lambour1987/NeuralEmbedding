//25-11-26
//25-11-26: Vervangen


#include "Trainer.h"

#include <iomanip>
#include <iostream>
#include <vector>
#include <iomanip>

// Maakt steeds hetzelfde netwerk: 1 -> 2 -> 2 -> 1
Sequential create_network() {
    Sequential network;

    network.add(
        LinearReLU(
            Tensor({0.5f, -0.5f}, {1, 2}),
            Tensor({0.0f, 0.0f}, {2})
        )
    );

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

    network.add(
        Linear(
            Tensor({1.0f, 1.0f}, {2, 1}),
            Tensor({0.0f}, {1})
        )
    );

    return network;
}


Sequential create_leaky_network() {
    Sequential network;

    network.add(
        LinearLeakyReLU(
            Tensor({0.5f, -0.5f}, {1, 2}),
            Tensor({0.0f, 0.0f}, {2})
        )
    );

    network.add(
        LinearLeakyReLU(
            Tensor(
                {0.5f, 0.0f,
                 0.0f, 0.5f},
                {2, 2}
            ),
            Tensor({0.0f, 0.0f}, {2})
        )
    );

    network.add(
        Linear(
            Tensor({1.0f, 1.0f}, {2, 1}),
            Tensor({0.0f}, {1})
        )
    );

    return network;
}


void print_predictions(
    Sequential& network,
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& targets
) {
    std::cout << "\n"
              << std::setw(10) << "Input"
              << std::setw(15) << "Target"
              << std::setw(15) << "Prediction"
              << '\n';

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        Tensor prediction = network.forward(inputs[i]);

        std::cout << std::setw(10) << inputs[i].data()[0]
                  << std::setw(15) << targets[i].data()[0]
                  << std::setw(15) << prediction.data()[0]
                  << '\n';
    }
}

void inspect_activations(
    Sequential& network,
    const std::vector<Tensor>& inputs
) {
    std::cout << "\n=== Hidden layer activations ===\n";

    std::cout << std::setw(8) << "Input"
              << std::setw(12) << "h1"
              << std::setw(12) << "h2"
              << std::setw(12) << "z1"
              << std::setw(12) << "z2"
              << std::setw(12) << "a1"
              << std::setw(12) << "a2"
              << '\n';

    for (const Tensor& input : inputs) {
        network.forward(input);

        const auto& hidden1 = network.linear_output(0);
        const auto& hidden2 = network.linear_output(1);

        const Tensor activation1 = network.activation_output(0);
        const Tensor activation2 = network.activation_output(1);

        std::cout << std::setw(8) << input.data()[0];

        for (float value : hidden1.data()) {
            std::cout << std::setw(12) << value;
        }

        for (float value : hidden2.data()) {
            std::cout << std::setw(12) << value;
        }

        for (float value : activation1.data()) {
            std::cout << std::setw(12) << value;
        }

        for (float value : activation2.data()) {
            std::cout << std::setw(12) << value;
        }

        std::cout << '\n';
    }
}

//25-11-26: tijdelijk vervangen
// void inspect_weights(const Sequential& network) {
//     std::cout << "\n=== Network weights ===\n";

//     for (std::size_t layer = 0; layer < network.size(); ++layer) {
//         std::cout << "\nLayer " << layer << "\n";

//         std::cout << "Weights: ";
//         for (float value : network.weight_gradients(layer).data()) {
//             std::cout << value << " ";
//         }

//         std::cout << '\n';
//     }
// }

void inspect_weights(const Sequential& network) {
    std::cout << "\n=== Network parameters ===\n";

    for (std::size_t layer = 0; layer < network.size(); ++layer) {
        const Tensor& weights = network.weights(layer);
        const Tensor& bias = network.bias(layer);

        std::cout << "\nLayer " << layer << '\n';

        std::cout << "Weight shape: ";
        for (std::size_t dimension : weights.shape()) {
            std::cout << dimension << " ";
        }

        std::cout << "\nWeights: ";
        for (float value : weights.data()) {
            std::cout << value << " ";
        }

        std::cout << "\nBias: ";
        for (float value : bias.data()) {
            std::cout << value << " ";
        }

        std::cout << '\n';
    }
}

//25-11-26: tijdelijke testfunctie
void inspect_initial_gradients(
    Sequential& network,
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& targets
) {
    network.zero_grad();

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        Tensor predictions = network.forward(inputs[i]);

        Tensor loss_gradients =
            mean_squared_error_backward(
                predictions,
                targets[i]
            );

        network.backward(loss_gradients, true);
    }

    network.scale_gradients(
        1.0f / static_cast<float>(inputs.size())
    );

    std::cout << "\n=== Initial averaged gradients ===\n";

    for (std::size_t layer = 0; layer < network.size(); ++layer) {
        std::cout << "\nLayer " << layer << '\n';

        std::cout << "Weights: ";
        for (float value : network.weight_gradients(layer).data()) {
            std::cout << value << " ";
        }

        std::cout << "\nBias: ";
        for (float value : network.bias_gradients(layer).data()) {
            std::cout << value << " ";
        }

        std::cout << '\n';
    }
}

//25-11-26: tijdelijke testfunctie
void inspect_gradients(const Sequential& network) {
    std::cout << "\n=== Gradient inspection ===\n";

    for (std::size_t layer = 0; layer < network.size(); ++layer) {
        std::cout << "\nLayer " << layer << '\n';

        std::cout << "Weight gradients: ";

        for (float value : network.weight_gradients(layer).data()) {
            std::cout << value << " ";
        }

        std::cout << "\nBias gradients: ";

        for (float value : network.bias_gradients(layer).data()) {
            std::cout << value << " ";
        }

        std::cout << '\n';
    }
}


int main() {
    // Dataset: y = x^2
    std::vector<Tensor> inputs = {
        Tensor({-2.0f}, {1, 1}),
        Tensor({-1.0f}, {1, 1}),
        Tensor({ 0.0f}, {1, 1}),
        Tensor({ 1.0f}, {1, 1}),
        Tensor({ 3.0f}, {1, 1})
    };

    std::vector<Tensor> targets = {
        Tensor({4.0f}, {1, 1}),
        Tensor({1.0f}, {1, 1}),
        Tensor({0.0f}, {1, 1}),
        Tensor({1.0f}, {1, 1}),
        Tensor({9.0f}, {1, 1})
    };

    // Twee identieke beginmodellen.
    Sequential online_network = create_network();
    Sequential mini_batch_network = create_network();
    Sequential leaky_network = create_leaky_network();
    Sequential diagnostic_network = create_network();

    inspect_initial_gradients(
        diagnostic_network,
        inputs,
        targets
    );

    // Learning rate blijft 0.1.
    GradientDescent optimizer(0.1f);
    Trainer trainer;

    constexpr std::size_t epochs = 100;
    //25-11-26: tijdelijk wijzigen van: constexpr std::size_t batch_size = 2; naar
    constexpr std::size_t batch_size = 5;

    std::cout << std::fixed << std::setprecision(4);

    // Loss histories verzamelen over alle trainingsblokken.
std::vector<float> online_losses;
std::vector<float> mini_batch_losses;
std::vector<float> leaky_losses;

constexpr std::size_t checkpoint_interval = 10;

// Training opdelen in blokken van 10 epochs.
for (std::size_t checkpoint = 0;
     checkpoint < epochs;
     checkpoint += checkpoint_interval) {

    // Online gradient descent.
    std::vector<float> online_chunk =
        trainer.train(
            online_network,
            optimizer,
            inputs,
            targets,
            checkpoint_interval
        );

    for (float loss : online_chunk) {
        online_losses.push_back(loss);
    }

    // Mini-batch gradient descent.
    std::vector<float> mini_batch_chunk =
        trainer.train_mini_batch(
            mini_batch_network,
            optimizer,
            inputs,
            targets,
            checkpoint_interval,
            batch_size
        );

    for (float loss : mini_batch_chunk) {
        mini_batch_losses.push_back(loss);
    }
    // Leaky ReLU mini-batch gradient descent.
    std::vector<float> leaky_chunk =
        trainer.train_mini_batch(
            leaky_network,
            optimizer,
            inputs,
            targets,
            checkpoint_interval,
            batch_size
        );

    for (float loss : leaky_chunk) {
        leaky_losses.push_back(loss);
    }

    // Activaties bekijken na iedere 10 epochs.
    std::cout << "\n====================================\n";
    std::cout << "CHECKPOINT: " << checkpoint + checkpoint_interval
              << " EPOCHS\n";
    std::cout << "====================================\n";

    inspect_weights(mini_batch_network);

    std::cout << "\nMini-batch activations:\n";
    inspect_activations(mini_batch_network, inputs);

    inspect_weights(leaky_network);

    std::cout << "\nLeaky ReLU activations:\n";
    inspect_activations(leaky_network, inputs);

    std::cout << "\nOnline activations:\n";
    inspect_activations(online_network, inputs);
}
    
    inspect_gradients(mini_batch_network);

    std::cout << "\n=== Leaky ReLU gradients ===\n";
    inspect_gradients(leaky_network);

    std::cout << "\n=== Loss progression ===\n";

    std::cout << "\nOnline:\n";
    for (std::size_t i = 0; i < online_losses.size(); i += 10) {
        std::cout << "Epoch " << i + 1
                << ": " << online_losses[i] << '\n';
    }

    std::cout << "\nMini-batch:\n";
    for (std::size_t i = 0; i < mini_batch_losses.size(); i += 10) {
        std::cout << "Epoch " << i + 1
                << ": " << mini_batch_losses[i] << '\n';
    }

    std::cout << "\nMini-batch Leaky ReLU:\n";
    for (std::size_t i = 0; i < leaky_losses.size(); i += 10) {
        std::cout << "Epoch " << i + 1
              << ": " << leaky_losses[i] << '\n';
    }

    // Resultaten vergelijken.
    std::cout << "\n=== Training comparison ===\n";

    std::cout << "Learning rate: " << optimizer.learning_rate() << '\n';
    std::cout << "Epochs:        " << epochs << '\n';
    std::cout << "Batch size:    " << batch_size << '\n';

    std::cout << "\nOnline gradient descent:\n";
    std::cout << "Initial loss: "
              << online_losses.front() << '\n';
    std::cout << "Final loss:   "
              << online_losses.back() << '\n';

    std::cout << "\nMini-batch gradient descent:\n";
    std::cout << "Initial loss: "
              << mini_batch_losses.front() << '\n';
    std::cout << "Final loss:   "
              << mini_batch_losses.back() << '\n';

    std::cout << "\nMini-batch Leaky ReLU:\n";
    std::cout << "Initial loss: "
          << leaky_losses.front() << '\n';
    std::cout << "Final loss:   "
          << leaky_losses.back() << '\n';

    std::cout << "\n=== Online predictions ===\n";
    print_predictions(online_network, inputs, targets);

    std::cout << "\n=== Mini-batch predictions ===\n";
    print_predictions(mini_batch_network, inputs, targets);

    std::cout << "\n=== Leaky ReLU predictions ===\n";
    print_predictions(leaky_network, inputs, targets);

    std::cout << "\n=== Online network activations ===\n";
    inspect_activations(online_network, inputs);

    std::cout << "\n=== Mini-batch network activations ===\n";
    inspect_activations(mini_batch_network, inputs);

    std::cout << "\n=== Leaky ReLU network activations ===\n";
    inspect_activations(leaky_network, inputs);

    return 0;
}
