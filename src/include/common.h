#pragma once

#include <array>
#include <string>
#include <stdexcept>
#include <vector>
#include <sstream>

#include "pagmo/types.hpp"

// Enums to reduce memory usage
enum class ProblemType {
  Ackley,
  Deb
};

enum class SelectionMethod {
  // Standard selection methods
  SUS,                   // Stochastic Universal Sampling
  RWS,                   // Roulette Wheel Selection
  Tournament,            // Tournament selection (default)
  
  // Tournament variations
  TournWITH_t2,          // Tournament with replacement, t=2
  TournWITH_t4,          // Tournament with replacement, t=4
  TournWITHOUT_t2,       // Tournament without replacement, t=2
  TournWITHOUT_t4,       // Tournament without replacement, t=4
  TournWITHPART_t2,      // Tournament with participation, t=2
  
  // Ranking selection methods
  ExpRankRWS_c0_9801,    // Exponential Ranking RWS, c=0.9801
  ExpRankRWS_c0_9606,    // Exponential Ranking RWS, c=0.9606
  ExpRankSUS_c0_9801,    // Exponential Ranking SUS, c=0.9801
  ExpRankSUS_c0_9606,    // Exponential Ranking SUS, c=0.9606
  LinRankRWS_b2,         // Linear Ranking RWS, beta=2
  LinRankRWS_b1_6,       // Linear Ranking RWS, beta=1.6
  LinRankSUS_b2,         // Linear Ranking SUS, beta=2
  LinRankSUS_b1_6        // Linear Ranking SUS, beta=1.6
};

enum class CrossoverType {
  Single,       // single-point crossover
  SBX,          // simulated binary crossover
  Uniform       // standard uniform crossover
};

enum class MutationType {
  Polynomial,   // Polynomial mutation
  Density       // Density-based mutation (as required in TASK.md)
};

// Encoding methods
enum class EncodingMethod {
  StandardBinary,  // Standard binary encoding
  GrayCode         // Gray code encoding
};

// Reproduction type
enum class ReproductionType {
  Generational,    // Generational replacement
  SteadyState      // Steady-state replacement
};

// For steady-state reproduction: selection for parent pool
enum class ParentSelectionMethod {
  Elite,           // Select the best individuals
  RWS              // Roulette Wheel Selection
};

// For steady-state reproduction: selection for next generation (or for deletion)
enum class ReplacementMethod {
  WorstComma,      // Replace worst individuals from parent population
  RandComma,       // Replace random individuals from parent population
  WorstPlus,       // Replace worst individuals from combined parent+child population
  RandPlus         // Replace random individuals from combined parent+child population
};

// Helper functions to convert enums to strings
std::string to_string(const ProblemType type);
std::string to_string(const SelectionMethod method);
std::string to_string(const CrossoverType type);
std::string to_string(const MutationType type);
std::string to_string(const EncodingMethod method);
std::string to_string(const ReproductionType type);
std::string to_string(const ParentSelectionMethod method);
std::string to_string(const ReplacementMethod method);

// Helper functions to convert strings to enums
CrossoverType crossover_from_string(const std::string& str);
MutationType mutation_from_string(const std::string& str);
SelectionMethod selection_from_string(const std::string& str);
EncodingMethod encoding_from_string(const std::string& str);
ReproductionType reproduction_from_string(const std::string& str);
ParentSelectionMethod parent_selection_from_string(const std::string& str);
ReplacementMethod replacement_from_string(const std::string& str);

// Configuration parameters
struct ga_config {
  // Problem parameters
  ProblemType problem_type;
  unsigned dimension;

  // GA parameters
  unsigned population_size;
  unsigned island_count;
  unsigned generations_per_evolution;
  unsigned total_evolutions;

  double crossover_prob;
  double mutation_prob;
  SelectionMethod selection_method;
  CrossoverType crossover_type;
  MutationType mutation_type;
  EncodingMethod encoding_method;
  
  // Reproduction parameters
  ReproductionType reproduction_type = ReproductionType::Generational;
  double generation_gap = 0.0;  // For steady-state (GG parameter)
  ParentSelectionMethod parent_selection_method = ParentSelectionMethod::Elite;
  ReplacementMethod replacement_method = ReplacementMethod::WorstComma;

  // For CSV header
  static std::string csv_header();

  // For CSV output
  std::string to_csv(int config_id) const;
};

// Helper function to calculate Euclidean distance
double euclidean_distance(const pagmo::vector_double &a, const pagmo::vector_double &b);

// Function to calculate standard deviation
double calculate_std_dev(const std::vector<double> &values, const double avg);

// Constants for determining successful runs
constexpr double DELTA = 0.01;  // Fitness threshold for success
constexpr double SIGMA = 0.01;  // Distance threshold for success