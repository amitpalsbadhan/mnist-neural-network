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

    // ALLOCATE ONCE: Pre-allocate gradient vectors to avoid resizing during training
    weight_gradients.resize(input_size, std::vector<float>(output_size, 0.0f));
    bias_gradients.resize(output_size, 0.0f);

    for (auto& row : weights) {
        for (float& val : row) {
            val = dist(gen);
        }
    }
}

// DenseLayer forward pass
std::vector<std::vector<float>> DenseLayer::forward(const std::vector<std::vector<float>>& inputs) {
    this->inputs = inputs; // Cache inputs for backpropagation
    
    // MEMORY OPTIMIZATION: Check if we need to resize outputs (only happens on first run or batch size change)
    if (outputs.size() != inputs.size() || outputs[0].size() != biases.size()) {
        outputs.assign(inputs.size(), std::vector<float>(biases.size()));
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < biases.size(); ++j) {
            // Overwrite existing memory (no need to zero out first)
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

    // MEMORY OPTIMIZATION: Ensure input_gradients is sized correctly
    if (input_gradients.size() != batch_size || input_gradients[0].size() != input_size) {
        input_gradients.assign(batch_size, std::vector<float>(input_size, 0.0f));
    } else {
        // RESET MEMORY: Zero out cached gradients because we accumulate (+=) below
        #pragma omp parallel for
        for (size_t i = 0; i < batch_size; ++i) {
            std::fill(input_gradients[i].begin(), input_gradients[i].end(), 0.0f);
        }
    }

    // RESET WEIGHT/BIAS GRADIENTS
    // We cannot use .assign() here effectively if we want to save allocation costs, 
    // so we manually zero them out in parallel.
    #pragma omp parallel for
    for (size_t i = 0; i < input_size; ++i) {
        std::fill(weight_gradients[i].begin(), weight_gradients[i].end(), 0.0f);
    }
    std::fill(bias_gradients.begin(), bias_gradients.end(), 0.0f);

    // Parallelize ONLY the batch loop (i). 
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            float grad = gradient[i][j];

            // Atomic update for shared bias gradients
            #pragma omp atomic
            bias_gradients[j] += grad;

            for (size_t k = 0; k < input_size; ++k) {
                float input_val = inputs[i][k];

                // Atomic update for shared weight gradients
                #pragma omp atomic
                weight_gradients[k][j] += input_val * grad;

                // Thread-safe update (i is unique to thread)
                input_gradients[i][k] += grad * weights[k][j];
            }
        }
    }

    // Average gradients over the batch
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < input_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            weight_gradients[i][j] /= static_cast<float>(batch_size);
        }
    }

    // Average biases
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
    this->inputs = inputs; 
    std::vector<std::vector<float>> outputs = inputs;

    if (activation_type == "relu") {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < outputs.size(); ++i) {
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                outputs[i][j] = std::max(0.0f, outputs[i][j]); 
            }
        }
    }
    else if (activation_type == "softmax") {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < outputs.size(); ++i) {
            float max_val = *std::max_element(outputs[i].begin(), outputs[i].end());
            float sum_exp = 0.0f;
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                outputs[i][j] = std::exp(outputs[i][j] - max_val); 
                sum_exp += outputs[i][j];
            }
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                outputs[i][j] /= sum_exp; 
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
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                input_gradients[i][j] *= (inputs[i][j] > 0) ? 1.0f : 0.0f; 
            }
        }
    }
    else if (activation_type == "softmax") {
        // Handled by CrossEntropy
    }
    else {
        throw std::invalid_argument("Unsupported activation type: " + activation_type);
    }

    return input_gradients;
}


// DenseLayer save implementation
void DenseLayer::save(std::ostream& os) const {
    size_t input_size = weights.size();
    size_t output_size = weights[0].size();
    os.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
    os.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));
    for (const auto& row : weights) {
        os.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
    os.write(reinterpret_cast<const char*>(&biases.data()), biases.size() * sizeof(float));
}

// DenseLayer load implementation
void DenseLayer::load(std::istream& is) {
    size_t input_size = 0;
    size_t output_size = 0;
    is.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
    is.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));

    weights.resize(input_size, std::vector<float>(output_size));
    biases.resize(output_size);
    
    // MEMORY FIX: Resize gradients here too in case we load into a fresh object
    weight_gradients.resize(input_size, std::vector<float>(output_size, 0.0f));
    bias_gradients.resize(output_size, 0.0f);

    for (auto& row : weights) {
        is.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }
    is.read(reinterpret_cast<char*>(&biases.data()), biases.size() * sizeof(float));
}
