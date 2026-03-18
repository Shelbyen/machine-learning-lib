#pragma once

#include <functional>
#include "Tensor.h"
#include "math_func.h"

class Layer
{
private:
    Tensor weight;
    Tensor bias;

    std::vector<double> lastValuesResult;   // after activation
    std::vector<double> lastValuesBeforAct; // before activation

    std::function<double(double)> activation;
    std::function<double(double)> derevativeActivation;

    Tensor linearOperations(const Tensor &input)
    {
        return input.multiplication(weight) + bias;
    }

    Tensor nonLinearOperations(const Tensor &input, bool memorize_values = true)
    {
        Tensor final_values = Tensor(1, input.getRow(0).size());

        for (int i = 0; i < input.getRow(0).size(); i++)
        {
            auto clear_value = activation(input(0, i));
            final_values(0, i) = clear_value;

            if (memorize_values)
            {
                lastValuesResult.push_back(clear_value);
                lastValuesBeforAct.push_back(final_values(0, i));
            }
        }
        return input;
    }

    Tensor deltaToLinear(Tensor const &deltaError)
    {
        // TODO: Vec derivative of activation func (Instead of Tensor())
        return deltaError.adamarMultiplication(Tensor(lastValuesBeforAct));
    }

    Tensor deltaWeight(Tensor const &deltaLinear)
    {
        return Tensor(lastValuesResult).transpose().multiplication(deltaLinear);
    }

    Tensor delteBias(Tensor const &deltaLinear)
    {
        return deltaLinear;
    }

    Tensor deltaToNext(Tensor const &deltaLinear)
    {
        return deltaLinear.multiplication(weight.transpose());
    }

public:
    Tensor forward(const Tensor &input, bool memorizeValues = true)
    {
        if (memorizeValues)
        {
            lastValuesBeforAct.clear();
            lastValuesResult.clear();
        }

        return nonLinearOperations(linearOperations(input), memorizeValues);
    }

    Tensor backward(Tensor deltaError, double speed, double moment)
    {
        Tensor deltaLinear = deltaToLinear(deltaError);
        Tensor dWeight = deltaWeight(deltaLinear);
        Tensor dBias = deltaLinear;
        for (auto i = 0; i < dWeight.getCol(0).size(); i++)
        {
            new_delta_weight(
                weight, 
                lastValuesResult[i], 
                dWeight.getRow(i), 
                speed, moment, 
                Tensor(0, weight.getRow(i).size()));
        }
        return deltaToNext(deltaLinear);
    }
};