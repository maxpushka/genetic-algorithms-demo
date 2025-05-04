#include "include/problems.h"

#include <cmath>

// Custom Deb function implementation
deb_func::deb_func(const unsigned dim) : m_dim(dim) {}

[[nodiscard]] pagmo::vector_double deb_func::fitness(
    const pagmo::vector_double &x) const {
  double result = 0.0;

  for (decltype(m_dim) i = 0u; i < m_dim; ++i) {
    const double xi = x[i];
    const double term1 =
        std::exp(-2.0 * std::log(2.0) * std::pow((xi - 0.1) / 0.8, 2.0));
    const double term2 = std::pow(std::sin(5.0 * M_PI * xi), 6.0);
    result += term1 * term2;
  }

  // We're maximizing
  return {-result};
}

[[nodiscard]] std::pair<pagmo::vector_double, pagmo::vector_double>
deb_func::get_bounds() const {
  pagmo::vector_double lb(m_dim, 0.0);
  pagmo::vector_double ub(m_dim, 1.023);
  return {lb, ub};
}

// Additional methods to help with analysis
[[nodiscard]] pagmo::vector_double deb_func::get_optimal_point() const {
  return pagmo::vector_double(m_dim, 0.1);
}

[[nodiscard]] double deb_func::get_optimal_fitness() const {
  return -m_dim;  // Each dimension contributes -1.0 at optimum
}

[[nodiscard]] std::string deb_func::get_name() const {
  return "Deb's function";
}