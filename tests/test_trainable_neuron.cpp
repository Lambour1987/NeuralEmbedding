//28-10-2026


#include "TrainableNeuron.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

void check(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main()
{
    TrainableNeuron neuron;

    check(neuron.weight() == 0.0f, "Initial weight should be zero");
    check(neuron.bias() == 0.0f, "Initial bias should be zero");
    check(neuron.forward(3.0f) == 0.0f,
          "Initial prediction should be zero");

    TrainableNeuron configured_neuron(2.0f, 1.0f);

    check(std::fabs(configured_neuron.forward(3.0f) - 7.0f) < 1e-6f,
          "Prediction should equal weight * input + bias");

    std::cout << "TrainableNeuron tests passed!\n";
    return 0;
}
