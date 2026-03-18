#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include "Tensor.h"

using namespace std;



bool zero_vec(const vector<double>& weights, double epsilon = 1e-10) {
    for(double val : weights) {
        if(fabs(val) > epsilon){
            return false;
        }
    }
    return true;
}

double summ(const vector<double>& weights, const vector<double>& deltas) {
    if (weights.size() != deltas.size()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for(size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * deltas[i];
    }
    return sum;
}

// double delta(double neuron_value, vector<double>& weights, vector<double>& deltas, double ideal_value) {
//     if (zero_vec(weights, 1e-10) == false) {
//         return (1 - neuron_value) * neuron_value * summ(weights, deltas);
//     }
//     else {
//         return (ideal_value - neuron_value) * (1 - neuron_value) * neuron_value;
//     }
// }


Tensor deltaSoftmaxCE(Tensor const &values, Tensor const &idealValues) {
    // CrossEntropy
    return softmax(values) - idealValues;
}


double deltaSig(double neuron_value, vector<double>& weights, double ideal_value) {
    return (ideal_value - neuron_value) * (1 - neuron_value) * neuron_value;
}

double deltaSig(double neuron_value, vector<double>& weights, vector<double>& deltas) {
    return (1 - neuron_value) * neuron_value * summ(weights, deltas);
}

void new_delta_weight(  // TODO: fix shit ()
        Tensor& weights,
        double neuron_value,
        Tensor const& deltas,
        double speed,
        double moment,
        const Tensor& previous_deltas
    ) {
    for (auto i = 0; i < deltas.getRow(0).size(); i++) {
        for(size_t j = 0; j < weights.getRow(0).size(); ++j) {
                double delta_weight = speed * (neuron_value * deltas(0, j)) + moment * previous_deltas(0, j);
                weights(i, j) += delta_weight;
        }
    }
    /*
    В классе Layer добавить:
    Tensor prevWeightDeltas_;
    Tensor prevBiasDeltas_;

    weights_ = weights_ - dWeight * speed + prevWeightDeltas_ * moment;
    prevWeightDeltas_ = dWeight * speed;

    bias_ = bias_ - dBias * speed + prevBiasDeltas_ * moment;
    prevBiasDeltas_ = dBias * speed;
    */
    
}

double MSE(vector<double>& neuron_values, vector<double>& ideal_values) {
    if (neuron_values.size() != ideal_values.size()) {
        return 0.0;
    }
    double sum_mistake = 0.0;
    for(size_t i = 0; i < neuron_values.size(); ++i) {
        double diff = (neuron_values[i] - ideal_values[i]);
        sum_mistake += diff * diff;
    }
    return sum_mistake / neuron_values.size();
}

Tensor softmax(Tensor const &values) {
    Tensor result = Tensor(1, values.getRow(0).size());
    double exp_sum = 0.0;

    for (auto i = 0; i < values.getRow(0).size(); i++) {
        exp_sum += exp(values(0, i));
    }
    for (auto i = 0; i < values.getRow(0).size(); i++) {
        result(0, i) = exp(values(0, i)) / exp_sum;
    }
    return result;
}
