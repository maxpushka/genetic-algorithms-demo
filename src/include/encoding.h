#pragma once

#include <string>

// Binary encoding/decoding utilities
// Convert a real value to standard binary encoding
std::string encode_real_to_binary(double value, double min, double max,
                                  unsigned bits);

// Convert standard binary encoding to real value
double decode_binary_to_real(const std::string& binary, double min, double max);

// Binary to Gray code conversion
std::string binary_to_gray(const std::string& binary);

// Gray code to binary conversion
std::string gray_to_binary(const std::string& gray);

// Convert a real value to Gray code encoding
std::string encode_real_to_gray(double value, double min, double max,
                                unsigned bits);

// Convert Gray code encoding to real value
double decode_gray_to_real(const std::string& gray, double min, double max);

// Discretization nodes encoding/decoding
// Convert a real value to discretization node encoding (2 decimal places)
std::string encode_real_to_discretization(double value, double min, double max,
                                         unsigned dimension);

// Convert discretization node encoding to real value
double decode_discretization_to_real(const std::string& encoded, double min, double max,
                                    unsigned dimension);