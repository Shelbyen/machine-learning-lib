#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "NeuralNet.h"

namespace py = pybind11;

PYBIND11_MODULE(neural_net, m) {
    m.doc() = "Neural Network training library";


    py::enum_<ActivationType>(m, "ActivationType")
        .value("ReLU", ActivationType::ReLU)
        .value("Sigmoid", ActivationType::Sigmoid)
        .export_values();


    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def(py::init<const std::vector<double>&>())
        .def("__call__", [](Tensor& t, size_t i, size_t j) -> double& {
            return t(i, j);
        }, py::return_value_policy::reference)
        .def("rows", &Tensor::rows)
        .def("cols", &Tensor::cols)
        .def("print", &Tensor::print)
        .def("to_list", [](const Tensor& t) {
            std::vector<double> result;
            for (size_t i = 0; i < t.rows(); ++i) {
                for (size_t j = 0; j < t.cols(); ++j) {
                    result.push_back(t(i, j));
                }
            }
            return result;
        })
        .def("to_numpy", [](const Tensor& t) {
            size_t rows = t.rows();
            size_t cols = t.cols();
            py::array_t<double> result({rows, cols});
            auto buf = result.request();
            double* ptr = (double*)buf.ptr;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    ptr[i * cols + j] = t(i, j);
                }
            }
            return result;
        });


    py::class_<TrainingSample>(m, "TrainingSample")
        .def(py::init<>())
        .def_readwrite("input", &TrainingSample::input)
        .def_readwrite("target", &TrainingSample::target);


    py::class_<Layer>(m, "Layer")
        .def(py::init<size_t, size_t>())
        .def("setActivation", &Layer::setActivation);


    py::class_<NeuralNet>(m, "NeuralNet")
        .def(py::init<>())
        .def("addLayer", &NeuralNet::addLayer)
        .def("predict", &NeuralNet::predict)
        .def("predict_from_numpy", [](NeuralNet& net, py::array_t<double> input_array) {
            auto buf = input_array.request();
            if (buf.ndim == 1) {

                double* ptr = (double*)buf.ptr;
                std::vector<double> data(ptr, ptr + buf.shape[0]);
                Tensor input(data);
                return net.predict(input);
            } else if (buf.ndim == 2) {
                size_t cols = buf.shape[1];
                Tensor input(1, cols);
                double* ptr = (double*)buf.ptr;
                for (size_t j = 0; j < cols; ++j) {
                    input(0, j) = ptr[j];
                }
                return net.predict(input);
            }
            throw std::runtime_error("Input must be 1D or 2D array");
        })
        .def("predict_batch", [](NeuralNet& net, py::array_t<double> X) {
            auto buf = X.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Input must be 2D array");
            }
            
            size_t n_samples = buf.shape[0];
            size_t n_features = buf.shape[1];
            double* ptr = (double*)buf.ptr;
            
            py::list results;
            for (size_t i = 0; i < n_samples; ++i) {
                std::vector<double> input_data(n_features);
                for (size_t j = 0; j < n_features; ++j) {
                    input_data[j] = ptr[i * n_features + j];
                }
                Tensor input(input_data);
                Tensor output = net.predict(input);
                
                std::vector<double> output_data;
                for (size_t j = 0; j < output.cols(); ++j) {
                    output_data.push_back(output(0, j));
                }
                results.append(py::cast(output_data));
            }
            return results;
        })
        .def("train_on_batch", [](NeuralNet& net, 
                                  py::list samples_list,
                                  double learning_rate, 
                                  int epochs) {
            std::vector<TrainingSample> samples;
            for (auto item : samples_list) {
                TrainingSample sample;
                py::tuple t = item.cast<py::tuple>();
                sample.input = t[0].cast<Tensor>();
                sample.target = t[1].cast<Tensor>();
                samples.push_back(sample);
            }
            net.train_on_batch(samples, learning_rate, epochs);
        })
        .def("train_from_numpy", [](NeuralNet& net,
                                    py::array_t<double> X,
                                    py::array_t<double> y,
                                    double learning_rate,
                                    int epochs) {
            auto x_buf = X.request();
            auto y_buf = y.request();
            
            if (x_buf.ndim != 2) {
                throw std::runtime_error("X must be 2D array");
            }
            
            size_t n_samples = x_buf.shape[0];
            size_t n_features = x_buf.shape[1];
            size_t n_outputs = y_buf.ndim == 2 ? y_buf.shape[1] : 1;
            
            double* x_ptr = (double*)x_buf.ptr;
            double* y_ptr = (double*)y_buf.ptr;
            
            std::vector<TrainingSample> samples;
            for (size_t i = 0; i < n_samples; ++i) {
                TrainingSample sample;
                
                // Input
                std::vector<double> input_data(n_features);
                for (size_t j = 0; j < n_features; ++j) {
                    input_data[j] = x_ptr[i * n_features + j];
                }
                sample.input = Tensor(input_data);
                
                // Target
                std::vector<double> target_data(n_outputs);
                for (size_t j = 0; j < n_outputs; ++j) {
                    target_data[j] = y_buf.ndim == 2 ? y_ptr[i * n_outputs + j] : y_ptr[i];
                }
                sample.target = Tensor(target_data);
                
                samples.push_back(sample);
            }
            
            net.train_on_batch(samples, learning_rate, epochs);
        })
        .def("weightsCount", &NeuralNet::weightsCount)
        .def("exportWeights", [](NeuralNet& net) {
            size_t n = net.weightsCount();
            std::vector<float> weights(n);
            net.exportWeights(weights.data());
            return py::array_t<float>(n, weights.data());
        })
        .def("get_weights", [](NeuralNet& net) {
            size_t n = net.weightsCount();
            std::vector<float> weights(n);
            net.exportWeights(weights.data());
            py::list result;
            for (float w : weights) {
                result.append(w);
            }
            return result;
        });

    m.def("softmax", [](const Tensor& logits) {
        return softmax(logits);
    }, "Compute softmax of a tensor");
    
    m.def("tensor_from_list", [](py::list data) {
        std::vector<double> vec;
        for (auto item : data) {
            vec.push_back(item.cast<double>());
        }
        return Tensor(vec);
    }, "Create a 1D tensor from a list");
}
