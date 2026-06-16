#pragma once

#include "Layer.h"
#include "math_func.h"

struct TrainingSample
{
	Tensor input;
	Tensor target;
};

struct Metrics
{
	float f1;
	float avgLoss;
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

	void addLayer(const Layer &layer)
	{
		layers_.push_back(layer);
	}

	Tensor forward(Tensor const &input)
	{
		Tensor result = input;
		for (auto &layer : layers_)
		{
			result = layer.forward(result);
		}
		return result;
	}

	Tensor backward(const Tensor &outputGradient)
	{
		Tensor gradient = outputGradient;
		// Проходим слои в обратном порядке
		for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
		{
			gradient = it->backward(gradient, learningRate_, momentum_);
		}
		return gradient; // градиент по входу (может пригодиться)
	}

	void train(const Tensor &input, const Tensor &target)
	{
		Tensor output = forward(input);
		// Предполагаем, что функция потерь уже включена в градиент
		// Для примера возьмём MSE и вычислим градиент вручную (выход - цель)
		Tensor lossGradient = (output - target) * (2.0 / output.cols()); // производная MSE
		backward(lossGradient);
	}

	void train_on_batch(const std::vector<TrainingSample> &samples,
						double learningRate,
						size_t epochs)
	{
		setLearningRate(learningRate);

		for (size_t epoch = 0; epoch < epochs; ++epoch)
		{
			for (const auto &sample : samples)
			{
				train(sample.input, sample.target);
			}
		}
	}

	size_t weightsCount() const
	{
		size_t total = 0;

		for (const auto &layer : layers_)
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
	Tensor predict(const Tensor &input)
	{
		return forward(input);
	}

	Tensor predictProba(const Tensor &input)
	{
		return softmax(forward(input));
	}

	Metrics evaluate(const std::vector<TrainingSample> &samples)
	{
		if (samples.empty())
			return {0.0f, 0.0f};

		int numClasses = (int)samples[0].target.cols();

		std::vector<int> tp(numClasses, 0);
		std::vector<int> fp(numClasses, 0);
		std::vector<int> fn(numClasses, 0);
		double totalLoss = 0.0;

		for (const auto &sample : samples)
		{
			Tensor output = forward(sample.input);

			Tensor diff = output - sample.target;
			for (size_t i = 0; i < diff.cols(); i++)
				totalLoss += diff(0, i) * diff(0, i);

			int pred = 0, actual = 0;
			double maxPred = output(0, 0), maxActual = sample.target(0, 0);
			for (int i = 1; i < numClasses; i++)
			{
				if (output(0, i) > maxPred)
				{
					maxPred = output(0, i);
					pred = i;
				}
				if (sample.target(0, i) > maxActual)
				{
					maxActual = sample.target(0, i);
					actual = i;
				}
			}

			if (pred == actual)
				tp[actual]++;
			else
			{
				fp[pred]++;
				fn[actual]++;
			}
		}

		float f1Sum = 0.0f;
		for (int c = 0; c < numClasses; c++)
		{
			float precision = (tp[c] + fp[c]) > 0 ? (float)tp[c] / (tp[c] + fp[c]) : 0.0f;
			float recall = (tp[c] + fn[c]) > 0 ? (float)tp[c] / (tp[c] + fn[c]) : 0.0f;
			float f1c = (precision + recall) > 0 ? 2.0f * precision * recall / (precision + recall) : 0.0f;
			f1Sum += f1c;
		}

		return {
			f1Sum / numClasses,
			(float)(totalLoss / samples.size())};
	}
};
