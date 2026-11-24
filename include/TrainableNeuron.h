//28-10-2026: Neuron training oefening


#pragma once

class TrainableNeuron
{
public:
    TrainableNeuron(float weight = 0.0f, float bias = 0.0f)
        : weight_(weight), bias_(bias)
    {
    }

    float forward(float input) const
    {
        return weight_ * input + bias_;
    }

    float weight() const
    {
        return weight_;
    }

    float bias() const
    {
        return bias_;
    }

private:
    float weight_;
    float bias_;
};
