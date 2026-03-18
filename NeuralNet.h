#pragma once

#include "Layer.h"
#include "math_func.h"


class NeuralNet {
private:
    std::vector<Layer> layers_;
    double learningRate_ = 0.01;
    double momentum_ = 0.9;

public:
    void setLearningRate(double lr) { learningRate_ = lr; }
    void setMomentum(double m) { momentum_ = m; }

    Tensor forward(Tensor const &input) {
        Tensor result = input;
        for (auto& layer : layers_) {
            result = layer.forward(result);
        }
        return result;
    }

    Tensor backward(const Tensor& outputGradient) {
        Tensor gradient = outputGradient;
        // Проходим слои в обратном порядке
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            gradient = it->backward(gradient, learningRate_, momentum_);
        }
        return gradient; // градиент по входу (может пригодиться)
    }

    void train(const Tensor& input, const Tensor& target) {
        Tensor output = forward(input);
        // Предполагаем, что функция потерь уже включена в градиент
        // Для примера возьмём MSE и вычислим градиент вручную (выход - цель)
        Tensor lossGradient = (output - target) * (2.0 / output.cols()); // производная MSE
        backward(lossGradient);
    }

    // Предсказание
    Tensor predict(const Tensor& input) {
        return forward(input);
    }
};
