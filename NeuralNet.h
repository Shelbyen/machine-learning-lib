#pragma once

#include "Layer.h"
#include "math_func.h"


class NeuralNet {
private:
    std::vector<Layer> layers;
public:
    Tensor forward(Tensor const &input) {
        Tensor result = input;
        for (auto layer : layers) {
            result = forward(result);
        }
        return result;
    }
    Tensor backward(double error_value) {
        for (auto i = layers.size() - 1; i >= 0; i--) {
            // layers[i].backward(error_value); TODO!!!
        }
    }
};
