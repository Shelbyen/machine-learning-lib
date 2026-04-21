#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include "Tensor.h"

using namespace std;

inline bool zero_vec(const vector<double>& weights, double epsilon = 1e-10)
{
	for (double val : weights)
	{
		if (fabs(val) > epsilon)
		{
			return false;
		}
	}
	return true;
}

inline double summ(const vector<double>& weights, const vector<double>& deltas)
{
	if (weights.size() != deltas.size())
	{
		return 0.0;
	}

	double sum = 0.0;
	for (size_t i = 0; i < weights.size(); ++i)
	{
		sum += weights[i] * deltas[i];
	}
	return sum;
}

// double delta(double neuron_value, vector<double> &weights, vector<double> &deltas, double ideal_value) {
//     if (!zero_vec(weights, 1e-10)) {
//         return (1 - neuron_value) * neuron_value * summ(weights, deltas);
//     }
//     else {она 
//         return (ideal_value - neuron_value) * (1 - neuron_value) * neuron_value;
//     }
// }


inline void new_delta_weight(
	Tensor& weights,
	double neuron_value,
	Tensor const& deltas,
	double speed,
	double moment,
	const Tensor& previous_deltas)
{
	for (auto i = 0; i < deltas.getRow(0).size(); i++)
	{
		for (size_t j = 0; j < weights.getRow(0).size(); ++j)
		{
			double delta_weight = speed * (neuron_value * deltas(0, j)) + moment * previous_deltas(0, j);
			weights(i, j) += delta_weight;
		}
	}
}

inline double MSE(vector<double>& neuron_values, vector<double>& ideal_values)
{
	if (neuron_values.size() != ideal_values.size())
	{
		return 0.0;
	}
	double sum_mistake = 0.0;
	for (size_t i = 0; i < neuron_values.size(); ++i)
	{
		double diff = (neuron_values[i] - ideal_values[i]);
		sum_mistake += diff * diff;
	}
	return sum_mistake / neuron_values.size();
}

inline Tensor softmax(const Tensor& values)
{
	size_t n = values.cols();
	Tensor result(1, n);

	double maxVal = values(0, 0);
	for (size_t i = 1; i < n; ++i)
	{
		if (values(0, i) > maxVal)
		{
			maxVal = values(0, i);
		}
	}

	double exp_sum = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		result(0, i) = std::exp(values(0, i) - maxVal);
		exp_sum += result(0, i);
	}

	for (size_t i = 0; i < n; ++i)
	{
		result(0, i) /= exp_sum;
	}

	return result;
}

inline Tensor deltaSoftmaxCE(const Tensor& values, const Tensor& idealValues)
{
	return softmax(values) - idealValues;
}

inline double deltaSig(double neuron_value, vector<double>& weights, double ideal_value)
{
	return (ideal_value - neuron_value) * (1 - neuron_value) * neuron_value;
}

inline double deltaSig(double neuron_value, vector<double>& weights, vector<double>& deltas)
{
	return (1 - neuron_value) * neuron_value * summ(weights, deltas);
}


inline double identity(double x)
{
	return x;
}

inline double dIdentity(double)
{
	return 1.0;
}

inline double sigmoid(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

inline double dSigmoidFromLinear(double x)
{
	double s = sigmoid(x);
	return s * (1.0 - s);
}

inline double relu(double x)
{
	return x > 0.0 ? x : 0.0;
}

inline double dReLU(double x)
{
	return x > 0.0 ? 1.0 : 0.0;
}