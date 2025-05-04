#pragma once

#include <memory>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>

#include "pagmo/problem.hpp"
#include "pagmo/types.hpp"
#include "pagmo/population.hpp"
#include "pagmo/detail/custom_comparisons.hpp"
#include "common.h"

// Base class for selection methods
class SelectionOperator {
public:
  virtual ~SelectionOperator() = default;
  virtual std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const = 0;
      
  // Helper to determine if we're minimizing or maximizing
  static bool is_minimization(const pagmo::problem& prob);
};

// Roulette Wheel Selection
class RouletteWheelSelection : public SelectionOperator {
public:
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Stochastic Universal Sampling
class StochasticUniversalSampling : public SelectionOperator {
public:
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Tournament selection with replacement
class TournamentWithReplacement : public SelectionOperator {
private:
  pagmo::population::size_type tournament_size;
  
public:
  explicit TournamentWithReplacement(pagmo::population::size_type t_size);
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Tournament selection without replacement
class TournamentWithoutReplacement : public SelectionOperator {
private:
  pagmo::population::size_type tournament_size;
  
public:
  explicit TournamentWithoutReplacement(pagmo::population::size_type t_size);
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Tournament with participation probability
class TournamentWithParticipation : public SelectionOperator {
private:
  pagmo::population::size_type tournament_size;
  
public:
  explicit TournamentWithParticipation(pagmo::population::size_type t_size);
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Exponential Ranking Selection (RWS-based)
class ExponentialRankRWS : public SelectionOperator {
private:
  double c;  // Base value
  
public:
  explicit ExponentialRankRWS(double c_val);
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Exponential Ranking Selection (SUS-based)
class ExponentialRankSUS : public SelectionOperator {
private:
  double c;  // Base value
  
public:
  explicit ExponentialRankSUS(double c_val);
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Linear Ranking Selection (RWS-based)
class LinearRankRWS : public SelectionOperator {
private:
  double beta;  // Selection pressure parameter (1.0 <= beta <= 2.0)
  
public:
  explicit LinearRankRWS(double b);
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Linear Ranking Selection (SUS-based)
class LinearRankSUS : public SelectionOperator {
private:
  double beta;  // Selection pressure parameter (1.0 <= beta <= 2.0)
  
public:
  explicit LinearRankSUS(double b);
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override;
};

// Factory function to create selection operator based on the selection method
std::unique_ptr<SelectionOperator> create_selection_operator(SelectionMethod method);