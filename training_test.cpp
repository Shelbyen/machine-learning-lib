#include <iostream>
#include <vector>
#include "NeuralNet.h"

int training_test()
{
    NeuralNet net;

    Layer l1(2, 3);
    Layer l2(3, 1);

    l1.setActivation(ActivationType::ReLU);
    l2.setActivation(ActivationType::Sigmoid);

    net.addLayer(l1);
    net.addLayer(l2);

    std::vector<TrainingSample> samples;

    TrainingSample s1;
    s1.input = Tensor(std::vector<double>{0.0, 0.0});
    s1.target = Tensor(std::vector<double>{0.0});
    samples.push_back(s1);

    TrainingSample s2;
    s2.input = Tensor(std::vector<double>{0.0, 1.0});
    s2.target = Tensor(std::vector<double>{1.0});
    samples.push_back(s2);

    TrainingSample s3;
    s3.input = Tensor(std::vector<double>{1.0, 0.0});
    s3.target = Tensor(std::vector<double>{1.0});
    samples.push_back(s3);

    TrainingSample s4;
    s4.input = Tensor(std::vector<double>{1.0, 1.0});
    s4.target = Tensor(std::vector<double>{0.0});
    samples.push_back(s4);

    std::cout << "=== BEFORE TRAINING ===\n";
    for (const auto& sample : samples)
    {
        Tensor out = net.predict(sample.input);
        out.print();
    }

    net.train_on_batch(samples, 0.1, 200); // можно менять lr и epochs

    std::cout << "\n=== AFTER TRAINING ===\n";
    for (const auto& sample : samples)
    {
        Tensor out = net.predict(sample.input);
        out.print();
    }

    size_t n = net.weightsCount();
    std::cout << "\nTotal parameters: " << n << "\n";

    float* buffer = new float[n];
    net.exportWeights(buffer);

    std::cout << "\nExported weights:\n";
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << buffer[i] << " ";
    }
    std::cout << "\n";

    delete[] buffer;

    std::cout << "\n=== SOFTMAX TEST ===\n";

    Tensor logits(1, 3);
    logits(0, 0) = 2.0;
    logits(0, 1) = 1.0;
    logits(0, 2) = 0.1;

    std::cout << "Manual logits:\n";
    logits.print();

    Tensor probs = softmax(logits);

    std::cout << "Softmax probabilities:\n";
    probs.print();

    double sum = 0.0;
    for (size_t i = 0; i < probs.cols(); ++i)
    {
        sum += probs(0, i);
    }

    std::cout << "Sum of probabilities: " << sum << "\n";
    return 0;
}