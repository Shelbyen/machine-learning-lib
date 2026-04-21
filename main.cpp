#include <iostream>

int tensor_test();
int layer_test();
int training_test();

int main()
{
    std::cout << "tensor_test()\n";
    tensor_test();
    std::cout << "\n";
    std::cout << "layer_test()\n";
    layer_test();
    std::cout << "\n";
    std::cout << "training_test()\n";
    training_test();
    return 0;
}