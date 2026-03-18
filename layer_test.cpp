#include "Layer.h"
#include <iostream>


int layer_test() {
    Tensor w(2, 3);

    w(0, 0) = 1; w(0, 1) = 2; w(0, 2) = 3;
    w(1, 0) = 4; w(1, 1) = 5; w(1, 2) = 6;

    Layer layer(w);

    std::cout << "Weights:\n";
    layer.weights().print();

    std::cout << "\nWeights for output 0:\n";
    layer.get_output_weights(0).print();

    std::cout << "\nConnections of input 1:\n";
    layer.get_input_connections(1).print();

    Tensor input(1, 2);
    input(0, 0) = 2;
    input(0, 1) = 3;

    Tensor output = layer.forward(input);

    std::cout << "\nOutput:\n";
    output.print();

    return 0;
}