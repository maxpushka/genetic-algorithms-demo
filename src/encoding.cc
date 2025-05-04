#include "include/encoding.h"
#include <cmath>
#include <iomanip>
#include <sstream>

// Binary encoding/decoding utilities
// Convert a real value to standard binary encoding
std::string encode_real_to_binary(double value, double min, double max,
                                  unsigned bits) {
  // Scale to 0...2^bits-1 range
  const double range = max - min;
  const double scaled = (value - min) / range * ((1 << bits) - 1);
  const unsigned long long intValue = static_cast<unsigned long long>(scaled);

  // Convert to binary string
  std::string binary;
  binary.reserve(bits);

  for (unsigned i = 0; i < bits; ++i) {
    binary.push_back(((intValue >> (bits - i - 1)) & 1) ? '1' : '0');
  }

  return binary;
}

// Convert standard binary encoding to real value
double decode_binary_to_real(const std::string& binary, double min,
                             double max) {
  const unsigned bits = binary.length();
  unsigned long long intValue = 0;

  // Convert from binary string to integer
  for (char bit : binary) {
    intValue = (intValue << 1) | (bit == '1' ? 1 : 0);
  }

  // Convert to real value in the original range
  const double range = max - min;
  return min + (intValue * range) / ((1ULL << bits) - 1);
}

// Binary to Gray code conversion
std::string binary_to_gray(const std::string& binary) {
  if (binary.empty()) return "";

  std::string gray;
  gray.reserve(binary.length());

  // First bit is the same in both binary and Gray code
  gray.push_back(binary[0]);

  // XOR operation for the rest of the bits
  for (size_t i = 1; i < binary.length(); ++i) {
    gray.push_back(binary[i - 1] == binary[i] ? '0' : '1');
  }

  return gray;
}

// Gray code to binary conversion
std::string gray_to_binary(const std::string& gray) {
  if (gray.empty()) return "";

  std::string binary;
  binary.reserve(gray.length());

  // First bit is the same in both binary and Gray code
  binary.push_back(gray[0]);

  // XOR operation for the rest of the bits
  for (size_t i = 1; i < gray.length(); ++i) {
    binary.push_back(binary[i - 1] == gray[i] ? '0' : '1');
  }

  return binary;
}

// Convert a real value to Gray code encoding
std::string encode_real_to_gray(double value, double min, double max,
                                unsigned bits) {
  // First convert to standard binary
  std::string binary = encode_real_to_binary(value, min, max, bits);

  // Then convert binary to Gray code
  return binary_to_gray(binary);
}

// Convert Gray code encoding to real value
double decode_gray_to_real(const std::string& gray, double min, double max) {
  // First convert Gray code to binary
  std::string binary = gray_to_binary(gray);

  // Then decode binary to real
  return decode_binary_to_real(binary, min, max);
}

// Discretization nodes encoding with 2 decimal places
// As specified in TASK.md: chain length l=10 for n=1 dimension
std::string encode_real_to_discretization(double value, double min, double max,
                                         unsigned dimension) {
  // Calculate the number of bits needed for each dimension
  // For n=1, we need 10 bits as specified in TASK.md
  constexpr unsigned bits_per_dimension = 10;
  const unsigned total_bits = bits_per_dimension * dimension;
  
  // Number of discrete points (for 2 decimal places precision)
  // For a range [min, max], we need (max-min)*100+1 points
  constexpr double precision = 0.01; // 2 decimal places
  const int num_points = static_cast<int>((max - min) / precision) + 1;
  
  // Scale value to discretized point index
  int point_index = static_cast<int>(std::round((value - min) / precision));
  
  // Clamp to valid range
  point_index = std::max(0, std::min(point_index, num_points - 1));
  
  // Convert to binary representation
  std::string binary;
  binary.reserve(bits_per_dimension);
  
  for (unsigned i = 0; i < bits_per_dimension; ++i) {
    binary.push_back(((point_index >> (bits_per_dimension - i - 1)) & 1) ? '1' : '0');
  }
  
  return binary;
}

// Convert discretization node encoding to real value
double decode_discretization_to_real(const std::string& encoded, double min, double max,
                                    unsigned dimension) {
  const unsigned bits_per_dimension = encoded.length() / dimension;
  
  // Parse binary to integer
  unsigned long long point_index = 0;
  for (unsigned i = 0; i < bits_per_dimension; ++i) {
    point_index = (point_index << 1) | (encoded[i] == '1' ? 1 : 0);
  }
  
  // Convert from discrete point index to real value
  const double precision = 0.01; // 2 decimal places
  
  // Calculate the value from the index
  double value = min + (point_index * precision);
  
  // Round to 2 decimal places to ensure precision
  value = std::round(value * 100.0) / 100.0;
  
  // Clamp to valid range
  return std::max(min, std::min(value, max));
}
