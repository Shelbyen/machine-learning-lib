#pragma once

#include <functional>
#include <stdexcept>
#include "Tensor.h"
#include "math_func.h"

enum class ActivationType
{
	Identity,
	Sigmoid,
	ReLU
};


class Layer
{
private:
	Tensor weights_;
	Tensor bias_;

	Tensor lastInput_;
	std::vector<double> lastValuesBeforAct; // before activation

	std::function<double(double)> activation;
	std::function<double(double)> derevativeActivation;

	Tensor linearOperations(const Tensor& input, bool memorize_values = true)
	{
		if (memorize_values)
		{
			lastInput_ = input;
		}

		Tensor final_values = input.multiplication(weights_) + bias_;

		if (memorize_values)
		{
			for (int i = 0; i < final_values.getRow(0).size(); i++)
			{
				lastValuesBeforAct.push_back(final_values(0, i));
			}
		}
		return final_values;
	}

	Tensor nonLinearOperations(const Tensor& input, bool memorize_values = true)
	{
		Tensor final_values = Tensor(1, input.getRow(0).size());

		for (int i = 0; i < input.getRow(0).size(); i++)
		{
			auto clear_value = activation(input(0, i));
			final_values(0, i) = clear_value;
		}
		return final_values;
	}

	Tensor deltaToLinear(Tensor const& deltaError)
	{
		std::vector<double> valueAfterDerevative;
		for (double value : lastValuesBeforAct)
		{
			valueAfterDerevative.push_back(derevativeActivation(value));
		}
		return deltaError.adamarMultiplication(Tensor(valueAfterDerevative));
	}

	Tensor deltaWeight(Tensor const& deltaLinear)
	{
		return lastInput_.transpose().multiplication(deltaLinear);
	}

	Tensor deltaToNext(Tensor const& deltaLinear)
	{
		return deltaLinear.multiplication(weights_.transpose());
	}

public:
	Layer(size_t inputSize, size_t outputSize)
		: weights_(inputSize, outputSize), bias_(1, outputSize)
	{
		setActivation(ActivationType::Identity);
	}

	Layer(const Tensor& weights)
		: weights_(weights),
		bias_(1, weights.cols(), 0.0)
	{
		setActivation(ActivationType::Identity);
	}

	Tensor& weights()
	{
		return weights_;
	}

	const Tensor& weights() const
	{
		return weights_;
	}

	Tensor& bias()
	{
		return bias_;
	}

	const Tensor& bias() const
	{
		return bias_;
	}

	void setActivation(ActivationType type)
	{
		switch (type)
		{
		case ActivationType::Identity:
			activation = ::identity;
			derevativeActivation = dIdentity;
			break;

		case ActivationType::Sigmoid:
			activation = sigmoid;
			derevativeActivation = dSigmoidFromLinear;
			break;

		case ActivationType::ReLU:
			activation = relu;
			derevativeActivation = dReLU;
			break;

		default:
			throw std::invalid_argument("Unknown activation type");
		}
	}

	double get_weight(size_t inputIndex, size_t outputIndex) const
	{
		return weights_(inputIndex, outputIndex);
	}

	Tensor get_output_weights(size_t outputIndex) const
	{
		return weights_.getCol(outputIndex);
	}

	Tensor get_input_connections(size_t inputIndex) const
	{
		return weights_.getRow(inputIndex);
	}

	Tensor forward(const Tensor& input, bool memorizeValues = true)
	{
		if (input.rows() != 1 || input.cols() != weights_.rows())
		{
			throw std::invalid_argument(
				"forward: input must be a row vector of size 1 x inputSize");
		}

		if (memorizeValues)
		{
			lastValuesBeforAct.clear();
			lastInput_ = Tensor(); // TODO: clear func
		}

		return nonLinearOperations(linearOperations(input, memorizeValues), memorizeValues);
	}

	Tensor backward(Tensor deltaError, double speed, double moment)
	{
		Tensor deltaLinear = deltaToLinear(deltaError);
		Tensor dWeight = deltaWeight(deltaLinear);
		Tensor dBias = deltaLinear;

		for (size_t i = 0; i < weights_.rows(); ++i)
		{
			for (size_t j = 0; j < weights_.cols(); ++j)
			{
				weights_(i, j) -= speed * dWeight(i, j);
			}
		}

		for (size_t j = 0; j < bias_.cols(); ++j)
		{
			bias_(0, j) -= speed * dBias(0, j);
		}

		return deltaToNext(deltaLinear);
	}
};
