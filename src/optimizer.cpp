#include "../include/optimizer.hpp"
#include <cstddef>
#include <omp.h> // Include OpenMP

void SGDOptimizer::update(std::vector<std::vector<float>>& weights, const std::vector<std::vector<float>>& gradients) {
    // Parallelize the nested loop. Collapse(2) treats it as one giant loop for better load balancing.
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t i = 0; i < weights.size(); ++i) {
        for (std::size_t j = 0; j < weights[i].size(); ++j) {
            float grad = gradients[i][j];
            // Clip gradients to range [-1.0, 1.0]
            if (grad > 1.0f) grad = 1.0f;
            if (grad < -1.0f) grad = -1.0f;
            weights[i][j] -= learning_rate * grad;
        }
    }
}

void SGDOptimizer::update(std::vector<float>& biases, const std::vector<float>& gradients) {
    // Parallelize the single loop
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < biases.size(); ++i) {
        float grad = gradients[i];
        // Clip gradients to range [-1.0, 1.0]
        if (grad > 1.0f) grad = 1.0f;
        if (grad < -1.0f) grad = -1.0f;
        biases[i] -= learning_rate * grad;
    }
}
