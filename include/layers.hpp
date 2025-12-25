#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <vector>
#include <string>
#include <iostream>
#include "optimizer.hpp"

class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass
    virtual std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs) = 0;

    // Backward pass
    virtual std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradient) = 0;

    // Update weights
    virtual void update(Optimizer& optimizer) = 0;

    // Save and load layer parameters
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};

class DenseLayer : public Layer {
private:
    std::vector<std::vector<float>> weights;     // Weight matrix
    std::vector<float> biases;                   // Bias vector
    
    // Memory Cache (Allocated once, reused)
    std::vector<std::vector<float>> inputs;      // Cached inputs from forward pass
    std::vector<std::vector<float>> outputs;     // Cached outputs
    std::vector<std::vector<float>> input_gradients; // Cached gradients to pass back
    
    std::vector<std::vector<float>> weight_gradients;
    std::vector<float> bias_gradients;

public:
    DenseLayer(int input_size, int output_size);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs) override;
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradient) override;
    void update(Optimizer& optimizer) override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;
};

class ActivationLayer : public Layer {
private:
    std::string activation_type;                // Activation type (e.g., "relu", "softmax")
    std::vector<std::vector<float>> inputs;     // Cached inputs for backpropagation

public:
    explicit ActivationLayer(const std::string& type);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs) override;
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& gradient) override;
    void update(Optimizer& optimizer) override {}

    void save(std::ostream& os) const override {}
    void load(std::istream& is) override {}
};

#endif // LAYERS_HPP
