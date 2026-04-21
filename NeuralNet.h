#pragma once

#include "Layer.h"
#include "math_func.h"

struct TrainingSample
{
	Tensor input;
	Tensor target;
};


class NeuralNet
{
private:
	std::vector<Layer> layers_;
	double learningRate_ = 0.01;
	double momentum_ = 0.9;

public:
	void setLearningRate(double lr) { learningRate_ = lr; }
	void setMomentum(double m) { momentum_ = m; }

	void addLayer(const Layer& layer)
	{
		layers_.push_back(layer);
	}

	Tensor forward(Tensor const& input)
	{
		Tensor result = input;
		for (auto& layer : layers_)
		{
			result = layer.forward(result);
		}
		return result;
	}

	Tensor backward(const Tensor& outputGradient)
	{
		Tensor gradient = outputGradient;
		// Проходим слои в обратном порядке
		for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
		{
			gradient = it->backward(gradient, learningRate_, momentum_);
		}
		return gradient; // градиент по входу (может пригодиться)
	}

	void train(const Tensor& input, const Tensor& target)
	{
		Tensor output = forward(input);
		// Предполагаем, что функция потерь уже включена в градиент
		// Для примера возьмём MSE и вычислим градиент вручную (выход - цель)
		Tensor lossGradient = (output - target) * (2.0 / output.cols()); // производная MSE
		backward(lossGradient);
	}

	void train_on_batch(const std::vector<TrainingSample>& samples,
		double learningRate,
		size_t epochs)
	{
		setLearningRate(learningRate);

		for (size_t epoch = 0; epoch < epochs; ++epoch)
		{
			for (const auto& sample : samples)
			{
				train(sample.input, sample.target);
			}
		}
	}

	size_t weightsCount() const
	{
		size_t total = 0;

		for (const auto& layer : layers_)
		{
			total += layer.weights().rows() * layer.weights().cols();
			total += layer.bias().rows() * layer.bias().cols();
		}

		return total;
	}

	void exportWeights(float *out) const
	{
    if (out == nullptr)
    {
        throw std::invalid_argument("exportWeights: output pointer is null");
    }

    size_t pos = 0;

    for (const auto &layer : layers_)
    {
        const Tensor &w = layer.weights();
        const Tensor &b = layer.bias();

        for (size_t i = 0; i < w.rows(); ++i)
        {
            for (size_t j = 0; j < w.cols(); ++j)
            {
                out[pos++] = static_cast<float>(w(i, j));
            }
        }

        for (size_t i = 0; i < b.rows(); ++i)
        {
            for (size_t j = 0; j < b.cols(); ++j)
            {
                out[pos++] = static_cast<float>(b(i, j));
            }
        }
    }
	}

	// Предсказание
	Tensor predict(const Tensor& input)
	{
		return forward(input);
	}

	Tensor predictProba(const Tensor& input)
	{
		return softmax(forward(input));
	}
};
