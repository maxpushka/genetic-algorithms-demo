#include "include/encoding_operator.h"

#include <string>

#include "include/common.h"
#include "include/encoding.h"

// Encode a chromosome (vector of real values) to string representation
std::string EncodingOperator::encode(const pagmo::vector_double& x) const {
  // Get problem bounds
  const auto bounds = m_problem.get_bounds();
  std::string encoded;
  
  // Encode each dimension according to the selected method
  for (size_t i = 0; i < x.size(); ++i) {
    const double min = bounds.first[i];
    const double max = bounds.second[i];
    
    std::string dim_encoded;
    switch (m_method) {
      case EncodingMethod::StandardBinary:
        dim_encoded = encode_real_to_binary(x[i], min, max, 32);
        break;
      case EncodingMethod::GrayCode:
        dim_encoded = encode_real_to_gray(x[i], min, max, 32);
        break;
      case EncodingMethod::Discretization:
        dim_encoded = encode_real_to_discretization(x[i], min, max, 1);
        break;
      default:
        // Default to standard binary if unknown
        dim_encoded = encode_real_to_binary(x[i], min, max, 32);
    }
    
    encoded += dim_encoded;
  }
  
  return encoded;
}

// Decode a string representation to a chromosome (vector of real values)
pagmo::vector_double EncodingOperator::decode(const std::string& encoded) const {
  // Get problem bounds and dimension
  const auto bounds = m_problem.get_bounds();
  const auto dim = bounds.first.size();
  
  // Calculate bits per dimension based on encoding method
  const unsigned bits_per_dimension = calculate_bit_length(dim) / dim;
  
  // Decode the string to a vector of real values
  pagmo::vector_double x(dim);
  
  for (size_t i = 0; i < dim; ++i) {
    const double min = bounds.first[i];
    const double max = bounds.second[i];
    
    // Extract the encoded substring for this dimension
    const std::string dim_encoded = encoded.substr(i * bits_per_dimension, bits_per_dimension);
    
    // Decode according to encoding method
    switch (m_method) {
      case EncodingMethod::StandardBinary:
        x[i] = decode_binary_to_real(dim_encoded, min, max);
        break;
      case EncodingMethod::GrayCode:
        x[i] = decode_gray_to_real(dim_encoded, min, max);
        break;
      case EncodingMethod::Discretization:
        x[i] = decode_discretization_to_real(dim_encoded, min, max, 1);
        break;
      default:
        // Default to standard binary if unknown
        x[i] = decode_binary_to_real(dim_encoded, min, max);
    }
  }
  
  return x;
}

// Get the total bit length for the chromosome (depends on encoding method)
unsigned EncodingOperator::get_bit_length(unsigned dimension) const {
  return calculate_bit_length(dimension);
}

// Helper method to calculate bit length based on encoding method
unsigned EncodingOperator::calculate_bit_length(unsigned dimension) const {
  switch (m_method) {
    case EncodingMethod::StandardBinary:
    case EncodingMethod::GrayCode:
      // Standard encoding uses 32 bits per dimension
      return 32 * dimension;
    case EncodingMethod::Discretization:
      // Discretization uses 10 bits per dimension as specified in TASK.md
      return 10 * dimension;
    default:
      // Default to standard encoding
      return 32 * dimension;
  }
}