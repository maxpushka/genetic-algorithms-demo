#pragma once

#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <string>

#include "common.h"
#include "encoding.h"

// Custom encoding operator that wraps the genetic encoding/decoding process
class EncodingOperator {
 public:
  // Constructor takes the encoding method and problem
  explicit EncodingOperator(EncodingMethod method, const pagmo::problem& prob)
      : m_method(method), m_problem(prob) {}

  // Encode a chromosome (vector of real values) to string representation
  std::string encode(const pagmo::vector_double& x) const;

  // Decode a string representation to a chromosome (vector of real values)
  pagmo::vector_double decode(const std::string& encoded) const;

  // Get the total bit length for the chromosome (depends on encoding method)
  unsigned get_bit_length(unsigned dimension) const;

 private:
  EncodingMethod m_method;
  pagmo::problem m_problem;

  // Helper method to get the total bit length for a chromosome
  // For standard binary and gray code, we use 32 bits per dimension
  // For discretization, we use 10 bits per dimension (for 2 decimal places)
  unsigned calculate_bit_length(unsigned dimension) const;
};