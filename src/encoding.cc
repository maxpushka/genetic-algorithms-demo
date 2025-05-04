#include "include/encoding.h"

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