#include "../include/layers.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <omp.h>

// DenseLayer constructor
DenseLayer::DenseLayer(int input_size, int output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    weights.resize(input_size, std::vector<float>(output_size));
    biases.resize(output_size, 0.0f);

    for (auto& row : weights) {
        for (float& val : row) {
            val = dist(gen);
        }
    }
}

// DenseLayer forward pass
std::vector<std::vector<float>> DenseLayer::forward(const std::vector<std::vector<float>>& inputs) {
    this->inputs = inputs; // Cache inputs for backpropagation
    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(biases.size(), 0.0f));

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < biases.size(); ++j) {
            outputs[i][j] = biases[j];
            for (size_t k = 0; k < weights.size(); ++k) {
                outputs[i][j] += inputs[i][k] * weights[k][j];
            }
        }
    }

    return outputs;
}

// DenseLayer backward pass
std::vector<std::vector<float>> DenseLayer::backward(const std::vector<std::vector<float>>& gradient) {
    size_t batch_size = gradient.size();
    size_t input_size = weights.size();
    size_t output_size = weights[0].size();

    // Reset Gradients
    weight_gradients.assign(input_size, std::vector<float>(output_size, 0.0f));
    bias_gradients.assign(output_size, 0.0f);
    std::vector<std::vector<float>> input_gradients(batch_size, std::vector<float>(input_size, 0.0f));

    // FIX: Parallelize ONLY the batch loop (i). 
    // Each thread handles one distinct sample in the batch completely.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            float grad = gradient[i][j];

            // Atomic update for shared bias gradients (multiple threads write to bias_gradients[j])
            #pragma omp atomic
            bias_gradients[j] += grad;

            for (size_t k = 0; k < input_size; ++k) {
                float input_val = inputs[i][k];

                // Atomic update for shared weight gradients
                #pragma omp atomic
                weight_gradients[k][j] += input_val * grad;

                // No atomic needed here: This thread exclusively owns row 'i' of input_gradients
                input_gradients[i][k] += grad * weights[k][j];
            }
        }
    }

    // Average gradients over the batch (Parallel Safe - distinct rows)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            weight_gradients[i][j] /= static_cast<float>(batch_size);
        }
    }

    // Average biases (Parallel Safe - distinct indices)
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < output_size; ++j) {
        bias_gradients[j] /= static_cast<float>(batch_size);
    }

    return input_gradients; 
}

// DenseLayer update
void DenseLayer::update(Optimizer& optimizer) {
    optimizer.update(weights, weight_gradients);
    optimizer.update(biases, bias_gradients);
}

// ActivationLayer implementation
ActivationLayer::ActivationLayer(const std::string& type) : activation_type(type) {}

std::vector<std::vector<float>> ActivationLayer::forward(const std::vector<std::vector<float>>& inputs) {
    this->inputs = inputs; // Cache inputs for backpropagation
    std::vector<std::vector<float>> outputs = inputs;

    if (activation_type == "relu") {
        // Parallelize ReLU activation using traditional for loops
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < outputs.size(); ++i) {
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                outputs[i][j] = std::max(0.0f, outputs[i][j]); // ReLU: max(0, x)
            }
        }
    }
    else if (activation_type == "softmax") {
        // Parallelize Softmax activation per row
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < outputs.size(); ++i) {
            float max_val = *std::max_element(outputs[i].begin(), outputs[i].end());
            float sum_exp = 0.0f;

            // Compute sum of exponentials
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                outputs[i][j] = std::exp(outputs[i][j] - max_val); // For numerical stability
                sum_exp += outputs[i][j];
            }

            // Normalize to get probabilities
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                outputs[i][j] /= sum_exp; // Softmax formula
            }
        }
    }
    else {
        throw std::invalid_argument("Unsupported activation type: " + activation_type);
    }

    return outputs;
}

std::vector<std::vector<float>> ActivationLayer::backward(const std::vector<std::vector<float>>& gradient) {
    std::vector<std::vector<float>> input_gradients = gradient;

    if (activation_type == "relu") {
        // Parallelize the ReLU backward pass using traditional for loops
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                input_gradients[i][j] *= (inputs[i][j] > 0) ? 1.0f : 0.0f; // Gradient of ReLU
            }
        }
    }
    else if (activation_type == "softmax") {
        // Do not modify gradients; already handled by CrossEntropyLoss
    }
    else {
        throw std::invalid_argument("Unsupported activation type: " + activation_type);
    }

    return input_gradients;
}


// DenseLayer save implementation
void DenseLayer::save(std::ostream& os) const {
    // Save the size of the weights matrix
    size_t input_size = weights.size();
    size_t output_size = weights[0].size();
    os.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    os.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));

    // Save weights
    for (const auto& row : weights) {
        os.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    // Save biases
    os.write(reinterpret_cast<const char*>(biases.data()), biases.size() * sizeof(float));
}

// DenseLayer load implementation
void DenseLayer::load(std::istream& is) {
    // Load the size of the weights matrix
    size_t input_size = 0;
    size_t output_size = 0;
    is.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    is.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

    // Resize weights and biases accordingly
    weights.resize(input_size, std::vector<float>(output_size));
    biases.resize(output_size);

    // Load weights
    for (auto& row : weights) {
        is.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }

    // Load biases
    is.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
}
