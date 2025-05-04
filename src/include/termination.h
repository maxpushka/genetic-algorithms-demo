#pragma once

#include <deque>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <vector>

#include "common.h"
#include "encoding_operator.h"

// Class for checking termination conditions
class TerminationChecker {
 public:
  // Constructor
  TerminationChecker(const ga_config& config);

  // Check if any termination condition is met
  bool check_termination(const pagmo::population& pop);

  // Get the reason for termination
  std::string get_termination_reason() const;

  // Get number of iterations performed
  unsigned get_iterations() const;

 private:
  // Configuration
  ga_config m_config;

  // Iterations counter
  unsigned m_iterations;

  // History of average fitness values for the last 10 generations
  std::deque<double> m_avg_fitness_history;

  // Termination reason
  std::string m_termination_reason;

  // Maximum number of iterations
  unsigned m_max_iterations;

  // Methods for checking specific termination conditions

  // Check if population has converged to homogeneity (99% of genes are the
  // same)
  bool check_homogeneity(const pagmo::population& pop,
                         const EncodingOperator& encoder);

  // Check if average fitness has not changed significantly for last 10
  // generations
  bool check_fitness_stability();

  // Check if maximum number of iterations has been reached
  bool check_max_iterations();
};