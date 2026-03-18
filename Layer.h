#pragma once

#include <functional>
#include <stdexcept>
#include "Tensor.h"
#include "math_func.h"

class Layer
{
private:
    Tensor weights_;
    Tensor bias_;

    Tensor lastInput_;
    std::vector<double> lastValuesBeforAct; // before activation

    std::function<double(double)> activation;
    std::function<double(double)> derevativeActivation;

    Tensor linearOperations(const Tensor &input, bool memorize_values = true)
    {
        if (memorize_values)
        {
            lastInput_ = input;
        }

        Tensor final_values = input.multiplication(weights_) + bias_;

        if (memorize_values) {
            for (int i = 0; i < final_values.getRow(0).size(); i++)
            {
                lastValuesBeforAct.push_back(final_values(0, i));
            }
        }
        return final_values;
    }

    Tensor nonLinearOperations(const Tensor &input, bool memorize_values = true)
    {
        Tensor final_values = Tensor(1, input.getRow(0).size());

        for (int i = 0; i < input.getRow(0).size(); i++)
        {
            auto clear_value = activation(input(0, i));
            final_values(0, i) = clear_value;
        }
        return final_values;
    }

    Tensor deltaToLinear(Tensor const &deltaError)
    {
        std::vector<double> valueAfterDerevative; 
        for (double value : lastValuesBeforAct) {
            valueAfterDerevative.push_back(derevativeActivation(value));
        }
        return deltaError.adamarMultiplication(Tensor(valueAfterDerevative));
    }

    Tensor deltaWeight(Tensor const &deltaLinear)
    {
        return Tensor(lastInput_).transpose().multiplication(deltaLinear);
    }

    Tensor deltaToNext(Tensor const &deltaLinear)
    {
        return deltaLinear.multiplication(weights_.transpose());
    }

public:
    Layer(size_t inputSize, size_t outputSize)
        : weights_(inputSize, outputSize, 0.0), bias_(1, outputSize, 0.0) {
    }

    Layer(const Tensor& weights, const Tensor& bias)
        : weights_(weights), bias_(bias) {
    }

    Tensor& weights() {
        return weights_;
    }

    const Tensor& weights() const {
        return weights_;
    }

    Tensor& bias() {
        return bias_;
    }

    const Tensor& bias() const {
        return bias_;
    }

    void setActivation(std::function<double(double)> act, std::function<double(double)> deriv) {
        activation = act;
        derevativeActivation = deriv;
    }

    double get_weight(size_t inputIndex, size_t outputIndex) const {
        return weights_(inputIndex, outputIndex);
    }

    Tensor get_output_weights(size_t outputIndex) const {
        return weights_.getCol(outputIndex);
    }

    Tensor get_input_connections(size_t inputIndex) const {
        return weights_.getRow(inputIndex);
    }

    Tensor forward(const Tensor &input, bool memorizeValues = true)
    {
        if (input.rows() != 1 || input.cols() != weights_.rows()) {
            throw std::invalid_argument(
                "forward: input must be a row vector of size 1 x inputSize"
            );
        }

        if (memorizeValues)
        {
            lastValuesBeforAct.clear();
            lastInput_ = Tensor();    // TODO: clear func
        }

        return nonLinearOperations(linearOperations(input, memorizeValues), memorizeValues);
    }

    Tensor backward(Tensor deltaError, double speed, double moment)
    {
        Tensor deltaLinear = deltaToLinear(deltaError);
        Tensor dWeight = deltaWeight(deltaLinear);
        Tensor dBias = deltaLinear;

        // TODO: add prev delta
        for (auto i = 0; i < dWeight.getCol(0).size(); i++)
        {
            new_delta_weight(
                weights_, 
                lastInput_(0, i), 
                dWeight.getRow(i), 
                speed, moment, 
                Tensor(0, weights_.getRow(i).size()));
        }

        for (auto i = 0; i < dBias.getCol(0).size(); i++)
        {
            new_delta_weight(
                bias_, 
                1, 
                dBias.getRow(i), 
                speed, moment, 
                Tensor(0, bias_.getRow(i).size()));
        }
        return deltaToNext(deltaLinear);
    }
};
