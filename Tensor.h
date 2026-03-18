#pragma once

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <iterator>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

class Tensor {
private:
    std::vector<double> data_;
    size_t rows_;
    size_t cols_;

    size_t index(size_t row, size_t col) const {
        return row * cols_ + col;
    }

public:
    Tensor() : rows_(0), cols_(0) {}

    Tensor(size_t rows, size_t cols, double value = NAN)
        : data_(rows* cols), rows_(rows), cols_(cols) {

        if (std::isnan(value)) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dist(-0.5, 0.5);

            for (size_t i = 0; i < data_.size(); ++i) {
                data_[i] = dist(gen);
            }
        }
        else {
            std::fill(data_.begin(), data_.end(), value);
        }
    }

    Tensor(std::vector<double> data)
        : data_(data), rows_(1), cols_(data.size()) {
    }

    void clear() {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = 0.0;
        }
    }

    size_t rows() const {
        return rows_;
    }

    size_t cols() const {
        return cols_;
    }

    size_t size() const {
        return data_.size();
    }

    double& operator()(size_t row, size_t col) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data_[index(row, col)];
    }

    double operator()(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data_[index(row, col)];
    }

    Tensor operator+(const Tensor& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Addition: shape mismatch");
        }

        Tensor result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }

        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Subtraction: shape mismatch");
        }

        Tensor result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }

        return result;
    }

    Tensor operator*(double scalar) const {
        Tensor result(rows_, cols_);

        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }

        return result;
    }

    friend Tensor operator*(double scalar, const Tensor& tensor) {
        return tensor * scalar;
    }

    Tensor adamarMultiplication(const Tensor& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("adamarMultiplication: shape mismatch");
        }

        Tensor result(rows_, cols_, 0.0);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) * other(i, j);
            }
        }

        return result;
    }

    Tensor multiplication(const Tensor& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("multiplication: shape mismatch");
        }

        Tensor result(rows_, other.cols_, 0.0);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                for (size_t k = 0; k < cols_; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }

        return result;
    }

    Tensor transpose() const {
        Tensor result(cols_, rows_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        return result;
    }

    Tensor getRow(size_t row) const {
        if (row >= rows_) {
            throw std::out_of_range("Row index out of range");
        }

        Tensor result(1, cols_);
        for (size_t j = 0; j < cols_; ++j) {
            result(0, j) = (*this)(row, j);
        }
        return result;
    }

    Tensor getCol(size_t col) const {
        if (col >= cols_) {
            throw std::out_of_range("Column index out of range");
        }

        Tensor result(rows_, 1);
        for (size_t i = 0; i < rows_; ++i) {
            result(i, 0) = (*this)(i, col);
        }
        return result;
    }

    void print() const {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

};
