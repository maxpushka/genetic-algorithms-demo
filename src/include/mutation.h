#pragma once

#include <pagmo/types.hpp>
#include <random>
#include <string>

#include "common.h"
#include "encoding_operator.h"

// Custom density-based mutation implementation as required by TASK.md
class DensityBasedMutation {
 public:
  // Constructor
  DensityBasedMutation(double mutation_prob, unsigned seed = 0);

  // Apply density-based mutation to a binary chromosome
  std::string mutate(const std::string& chromosome) const;

  // Mutate a real-valued chromosome with the corresponding encoding
  pagmo::vector_double mutate(const pagmo::vector_double& x,
                              const EncodingOperator& encoder) const;

 private:
  // Mutation probability (per bit)
  double m_mutation_prob;

  // Random number generator
  mutable std::mt19937 m_gen;

  // Uniform distribution for mutation decision
  mutable std::uniform_real_distribution<double> m_dist;
};