#pragma once

#include <cmath>
#include <string>
#include <utility>

#include "pagmo/problem.hpp"
#include "pagmo/types.hpp"

// Custom Deb function implementation
struct deb_func {
  explicit deb_func(const unsigned dim = 1);

  [[nodiscard]] pagmo::vector_double fitness(
      const pagmo::vector_double &x) const;
  [[nodiscard]] std::pair<pagmo::vector_double, pagmo::vector_double>
  get_bounds() const;

  // Additional methods to help with analysis
  [[nodiscard]] pagmo::vector_double get_optimal_point() const;
  [[nodiscard]] double get_optimal_fitness() const;
  [[nodiscard]] std::string get_name() const;

  unsigned m_dim;
};