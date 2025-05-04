#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "pagmo/types.hpp"

// Utility for tracking statistics
struct run_stats {
  bool is_successful = false;
  unsigned iterations = 0;
  unsigned fitness_evals = 0;
  double f_max = 0.0;
  pagmo::vector_double x_max;
  double f_avg = 0.0;
  double convergence = 0.0;
  double peak_accuracy = 0.0;
  double distance_accuracy = 0.0;
  unsigned long long execution_time_ms = 0;

  // For CSV output
  [[nodiscard]] std::string to_csv() const;
};

// Aggregate statistics
struct aggregate_stats {
  double success_rate = 0.0;

  // For successful runs
  unsigned min_iterations = 0;
  unsigned max_iterations = 0;
  double avg_iterations = 0.0;
  double std_iterations = 0.0;

  unsigned min_evals = 0;
  unsigned max_evals = 0;
  double avg_evals = 0.0;
  double std_evals = 0.0;

  // Execution time statistics
  unsigned long long min_exec_time = 0;
  unsigned long long max_exec_time = 0;
  double avg_exec_time = 0.0;
  double std_exec_time = 0.0;

  double min_f_max = 0.0;
  double max_f_max = 0.0;
  double avg_f_max = 0.0;
  double std_f_max = 0.0;

  double min_f_avg = 0.0;
  double max_f_avg = 0.0;
  double avg_f_avg = 0.0;
  double std_f_avg = 0.0;

  double min_convergence = 0.0;
  double max_convergence = 0.0;
  double avg_convergence = 0.0;
  double std_convergence = 0.0;

  double min_peak_accuracy = 0.0;
  double max_peak_accuracy = 0.0;
  double avg_peak_accuracy = 0.0;
  double std_peak_accuracy = 0.0;

  double min_dist_accuracy = 0.0;
  double max_dist_accuracy = 0.0;
  double avg_dist_accuracy = 0.0;
  double std_dist_accuracy = 0.0;

  // For failed runs
  unsigned min_iterations_f = 0;
  unsigned max_iterations_f = 0;
  double avg_iterations_f = 0.0;
  double std_iterations_f = 0.0;

  // For CSV header
  static std::string csv_header();

  // For CSV output
  [[nodiscard]] std::string to_csv() const;
};

// Function to calculate aggregate statistics
aggregate_stats calculate_aggregate_stats(const std::vector<run_stats>& runs);

// Data structures for I/O operations
struct DetailedResultEntry {
  int config_id;
  unsigned run_id;
  run_stats stats;
};

struct SummaryResultEntry {
  int config_id;
  ga_config config;
  aggregate_stats stats;
};