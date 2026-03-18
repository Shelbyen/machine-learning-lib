#include <iostream>
#include "Tensor.h"

int tensor_test() {
    {
        Tensor a(2, 3, 0.0);

        a(0, 0) = 1.0;
        a(0, 1) = 2.0;
        a(0, 2) = 3.0;
        a(1, 0) = 4.0;
        a(1, 1) = 5.0;
        a(1, 2) = 6.0;

        std::cout << "rows = " << a.rows() << "\n";
        std::cout << "cols = " << a.cols() << "\n";
        std::cout << "size = " << a.size() << "\n\n";

        for (size_t i = 0; i < a.rows(); ++i) {
            for (size_t j = 0; j < a.cols(); ++j) {
                std::cout << a(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

    {
        Tensor a(2, 2, 0.0);
        Tensor b(2, 2, 0.0);

        a(0, 0) = 1; a(0, 1) = 2;
        a(1, 0) = 3; a(1, 1) = 4;

        b(0, 0) = 5; b(0, 1) = 6;
        b(1, 0) = 7; b(1, 1) = 8;

        Tensor c = a + b;
        Tensor d = a - b;
        Tensor e = a * 2;

        for (size_t i = 0; i < c.rows(); ++i) {
            for (size_t j = 0; j < c.cols(); ++j)
                std::cout << c(i, j) << " ";
            std::cout << "\n";
        }

    }

    {
        Tensor a(2, 2);
        Tensor b(2, 2);

        a(0, 0) = 1; a(0, 1) = 2;
        a(1, 0) = 3; a(1, 1) = 4;

        b(0, 0) = 5; b(0, 1) = 6;
        b(1, 0) = 7; b(1, 1) = 8;

        Tensor c = a.multiplication(b);

        for (size_t i = 0; i < c.rows(); ++i) {
            for (size_t j = 0; j < c.cols(); ++j) {
                std::cout << c(i, j) << " ";
            }
            std::cout << "\n";

        }
    }

    {
        Tensor a(2, 3);

        a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
        a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

        Tensor t = a.transpose();

        for (size_t i = 0; i < t.rows(); ++i) {
            for (size_t j = 0; j < t.cols(); ++j)
                std::cout << t(i, j) << " ";
            std::cout << "\n";
        }
    }

    {
        Tensor t(2, 2);

        t(0, 0) = 1;
        t(0, 1) = 2;
        t(1, 0) = 3;
        t(1, 1) = 4;

        t.clear();

        t.print();
    }

    return 0;
}
