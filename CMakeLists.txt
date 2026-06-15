cmake_minimum_required(VERSION 3.12)
project(pybind11_example)

# Find the installed pybind11 package on your system
find_package(pybind11 REQUIRED)

# Create the extension module (compiles example.cpp into my_module.so/.pyd)
pybind11_add_module(mchine to_python_lib.cpp)
