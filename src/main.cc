#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "pagmo/algorithm.hpp"
#include "pagmo/algorithms/sga.hpp"
#include "pagmo/archipelago.hpp"
#include "pagmo/io.hpp"
#include "pagmo/problem.hpp"
#include "pagmo/problems/ackley.hpp"
#include "pagmo/types.hpp"

// Custom Deb function implementation
struct deb_func {
  explicit deb_func(const unsigned dim = 1) : m_dim(dim) {}

  [[nodiscard]] pagmo::vector_double fitness(
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
  get_bounds() const {
    pagmo::vector_double lb(m_dim, 0.0);
    pagmo::vector_double ub(m_dim, 1.023);
    return {lb, ub};
  }

  // Additional methods to help with analysis
  [[nodiscard]] pagmo::vector_double get_optimal_point() const {
    return pagmo::vector_double(m_dim, 0.1);
  }

  [[nodiscard]] double get_optimal_fitness() const {
    return -m_dim;  // Each dimension contributes -1.0 at optimum
  }

  [[nodiscard]] std::string get_name() const { return "Deb's function"; }

  unsigned m_dim;
};

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
  [[nodiscard]] std::string to_csv() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << (is_successful ? 1 : 0) << ",";
    ss << iterations << ",";
    ss << fitness_evals << ",";
    ss << execution_time_ms << ",";
    ss << f_max << ",";

    // x_max as a string
    for (const auto x : x_max) {
      ss << x << ";";
    }
    ss << ",";

    ss << f_avg << ",";
    ss << convergence << ",";
    ss << peak_accuracy << ",";
    ss << distance_accuracy;

    return ss.str();
  }
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
  static std::string csv_header() {
    return "Success_Rate,"
           "Min_NI,Max_NI,Avg_NI,Sigma_NI,"
           "Min_NFE,Max_NFE,Avg_NFE,Sigma_NFE,"
           "Min_Time,Max_Time,Avg_Time,Sigma_Time,"
           "Min_Fmax,Max_Fmax,Avg_Fmax,Sigma_Fmax,"
           "Min_Favg,Max_Favg,Avg_Favg,Sigma_Favg,"
           "Min_FC,Max_FC,Avg_FC,Sigma_FC,"
           "Min_PA,Max_PA,Avg_PA,Sigma_PA,"
           "Min_DA,Max_DA,Avg_DA,Sigma_DA,"
           "Min_NI_f,Max_NI_f,Avg_NI_f,Sigma_NI_f";
  }

  // For CSV output
  [[nodiscard]] std::string to_csv() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6);

    ss << success_rate << ",";

    ss << min_iterations << "," << max_iterations << "," << avg_iterations
       << "," << std_iterations << ",";
    ss << min_evals << "," << max_evals << "," << avg_evals << "," << std_evals
       << ",";
    ss << min_exec_time << "," << max_exec_time << "," << avg_exec_time << "," << std_exec_time
       << ",";
    ss << min_f_max << "," << max_f_max << "," << avg_f_max << "," << std_f_max
       << ",";
    ss << min_f_avg << "," << max_f_avg << "," << avg_f_avg << "," << std_f_avg
       << ",";
    ss << min_convergence << "," << max_convergence << "," << avg_convergence
       << "," << std_convergence << ",";
    ss << min_peak_accuracy << "," << max_peak_accuracy << ","
       << avg_peak_accuracy << "," << std_peak_accuracy << ",";
    ss << min_dist_accuracy << "," << max_dist_accuracy << ","
       << avg_dist_accuracy << "," << std_dist_accuracy << ",";

    ss << min_iterations_f << "," << max_iterations_f << "," << avg_iterations_f
       << "," << std_iterations_f;

    return ss.str();
  }
};

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

// Helper functions to convert enums to strings
inline std::string to_string(const ProblemType type) {
  switch (type) {
    case ProblemType::Ackley: return "Ackley";
    case ProblemType::Deb: return "Deb";
    default: return "Unknown";
  }
}

inline std::string to_string(const SelectionMethod method) {
  switch (method) {
    // Standard selection methods
    case SelectionMethod::SUS: return "sus";
    case SelectionMethod::RWS: return "rws";
    case SelectionMethod::Tournament: return "tournament";
    
    // Tournament variations
    case SelectionMethod::TournWITH_t2: return "tournament_with_t2";
    case SelectionMethod::TournWITH_t4: return "tournament_with_t4";
    case SelectionMethod::TournWITHOUT_t2: return "tournament_without_t2";
    case SelectionMethod::TournWITHOUT_t4: return "tournament_without_t4";
    case SelectionMethod::TournWITHPART_t2: return "tournament_part_t2";
    
    // Ranking selection methods
    case SelectionMethod::ExpRankRWS_c0_9801: return "exp_rank_rws_c0_9801";
    case SelectionMethod::ExpRankRWS_c0_9606: return "exp_rank_rws_c0_9606";
    case SelectionMethod::ExpRankSUS_c0_9801: return "exp_rank_sus_c0_9801";
    case SelectionMethod::ExpRankSUS_c0_9606: return "exp_rank_sus_c0_9606";
    case SelectionMethod::LinRankRWS_b2: return "lin_rank_rws_b2";
    case SelectionMethod::LinRankRWS_b1_6: return "lin_rank_rws_b1_6";
    case SelectionMethod::LinRankSUS_b2: return "lin_rank_sus_b2";
    case SelectionMethod::LinRankSUS_b1_6: return "lin_rank_sus_b1_6";
    
    default: return "unknown";
  }
}

inline std::string to_string(const CrossoverType type) {
  switch (type) {
    case CrossoverType::Single: return "single";
    case CrossoverType::SBX: return "sbx";
    case CrossoverType::Uniform: return "uniform";
    default: return "unknown";
  }
}

inline std::string to_string(const MutationType type) {
  switch (type) {
    case MutationType::Polynomial: return "polynomial";
    case MutationType::Density: return "density";
    default: return "unknown";
  }
}

// Helper functions to convert strings to enums
inline CrossoverType crossover_from_string(const std::string& str) {
  if (str == "single") return CrossoverType::Single;
  if (str == "sbx") return CrossoverType::SBX;
  if (str == "uniform") return CrossoverType::Uniform;
  throw std::runtime_error("Unknown crossover type: " + str);
}

inline MutationType mutation_from_string(const std::string& str) {
  if (str == "polynomial") return MutationType::Polynomial;
  if (str == "density") return MutationType::Density;
  throw std::runtime_error("Unknown mutation type: " + str);
}

inline SelectionMethod selection_from_string(const std::string& str) {
  // Standard selection methods
  if (str == "sus") return SelectionMethod::SUS;
  if (str == "rws") return SelectionMethod::RWS;
  if (str == "tournament") return SelectionMethod::Tournament;
  
  // Tournament variations
  if (str == "tournament_with_t2") return SelectionMethod::TournWITH_t2;
  if (str == "tournament_with_t4") return SelectionMethod::TournWITH_t4;
  if (str == "tournament_without_t2") return SelectionMethod::TournWITHOUT_t2;
  if (str == "tournament_without_t4") return SelectionMethod::TournWITHOUT_t4;
  if (str == "tournament_part_t2") return SelectionMethod::TournWITHPART_t2;
  
  // Ranking selection methods
  if (str == "exp_rank_rws_c0_9801") return SelectionMethod::ExpRankRWS_c0_9801;
  if (str == "exp_rank_rws_c0_9606") return SelectionMethod::ExpRankRWS_c0_9606;
  if (str == "exp_rank_sus_c0_9801") return SelectionMethod::ExpRankSUS_c0_9801;
  if (str == "exp_rank_sus_c0_9606") return SelectionMethod::ExpRankSUS_c0_9606;
  if (str == "lin_rank_rws_b2") return SelectionMethod::LinRankRWS_b2;
  if (str == "lin_rank_rws_b1_6") return SelectionMethod::LinRankRWS_b1_6;
  if (str == "lin_rank_sus_b2") return SelectionMethod::LinRankSUS_b2;
  if (str == "lin_rank_sus_b1_6") return SelectionMethod::LinRankSUS_b1_6;
  
  throw std::runtime_error("Unknown selection method: " + str);
}

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
inline std::string to_string(const EncodingMethod method) {
  switch (method) {
    case EncodingMethod::StandardBinary: return "standard_binary";
    case EncodingMethod::GrayCode: return "gray_code";
    default: return "unknown";
  }
}

inline std::string to_string(const ReproductionType type) {
  switch (type) {
    case ReproductionType::Generational: return "generational";
    case ReproductionType::SteadyState: return "steady_state";
    default: return "unknown";
  }
}

inline std::string to_string(const ParentSelectionMethod method) {
  switch (method) {
    case ParentSelectionMethod::Elite: return "elite";
    case ParentSelectionMethod::RWS: return "rws";
    default: return "unknown";
  }
}

inline std::string to_string(const ReplacementMethod method) {
  switch (method) {
    case ReplacementMethod::WorstComma: return "worst_comma";
    case ReplacementMethod::RandComma: return "rand_comma";
    case ReplacementMethod::WorstPlus: return "worst_plus";
    case ReplacementMethod::RandPlus: return "rand_plus";
    default: return "unknown";
  }
}

inline EncodingMethod encoding_from_string(const std::string& str) {
  if (str == "standard_binary") return EncodingMethod::StandardBinary;
  if (str == "gray_code") return EncodingMethod::GrayCode;
  throw std::runtime_error("Unknown encoding method: " + str);
}

inline ReproductionType reproduction_from_string(const std::string& str) {
  if (str == "generational") return ReproductionType::Generational;
  if (str == "steady_state") return ReproductionType::SteadyState;
  throw std::runtime_error("Unknown reproduction type: " + str);
}

inline ParentSelectionMethod parent_selection_from_string(const std::string& str) {
  if (str == "elite") return ParentSelectionMethod::Elite;
  if (str == "rws") return ParentSelectionMethod::RWS;
  throw std::runtime_error("Unknown parent selection method: " + str);
}

inline ReplacementMethod replacement_from_string(const std::string& str) {
  if (str == "worst_comma") return ReplacementMethod::WorstComma;
  if (str == "rand_comma") return ReplacementMethod::RandComma;
  if (str == "worst_plus") return ReplacementMethod::WorstPlus;
  if (str == "rand_plus") return ReplacementMethod::RandPlus;
  throw std::runtime_error("Unknown replacement method: " + str);
}

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
  static std::string csv_header() {
    return "Config_ID,Problem,Dimension,Population_Size,Islands,"
           "Generations_Per_Evolution,Total_Evolutions,"
           "Encoding_Method,Crossover_Type,Crossover_Prob,Mutation_Type,Mutation_Prob,"
           "Selection_Method,Reproduction_Type,Generation_Gap,Parent_Selection,Replacement";
  }

  // For CSV output
  std::string to_csv(int config_id) const {
    std::stringstream ss;
    ss << config_id << ",";
    ss << to_string(problem_type) << ",";
    ss << dimension << ",";
    ss << population_size << ",";
    ss << island_count << ",";
    ss << generations_per_evolution << ",";
    ss << total_evolutions << ",";
    ss << to_string(encoding_method) << ",";
    ss << to_string(crossover_type) << ",";
    ss << crossover_prob << ",";
    ss << to_string(mutation_type) << ",";
    ss << mutation_prob << ",";
    ss << to_string(selection_method) << ",";
    ss << to_string(reproduction_type) << ",";
    ss << generation_gap << ",";
    ss << to_string(parent_selection_method) << ",";
    ss << to_string(replacement_method);

    return ss.str();
  }
};

// Binary encoding/decoding utilities
// Convert a real value to standard binary encoding
std::string encode_real_to_binary(double value, double min, double max, unsigned bits) {
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
double decode_binary_to_real(const std::string& binary, double min, double max) {
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
std::string encode_real_to_gray(double value, double min, double max, unsigned bits) {
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

// Function to calculate Euclidean distance
double euclidean_distance(const pagmo::vector_double &a,
                          const pagmo::vector_double &b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

// Function to calculate standard deviation
double calculate_std_dev(const std::vector<double> &values, const double avg) {
  if (values.empty() || values.size() == 1) return 0.0;

  double sum_squared_diff = 0.0;
  for (const double value : values) {
    const double diff = value - avg;
    sum_squared_diff += diff * diff;
  }

  return std::sqrt(sum_squared_diff / (values.size() - 1));
}

// Base class for selection methods
class SelectionOperator {
public:
  virtual ~SelectionOperator() = default;
  virtual std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const = 0;
      
  // Helper to determine if we're minimizing or maximizing
  static bool is_minimization(const pagmo::problem& prob) {
    // In PaGMO, all problems are formulated as minimization problems
    // However, we negate the objective function for problems we want to maximize
    // So for our implementation, we need to check the problem type:
    const std::string name = prob.get_name();
    
    // Ackley is a minimization problem in PaGMO
    if (name == "Ackley Function") {
      return true;
    }
    
    // Our custom Deb function is set up to return negated values
    // because we want to maximize it
    if (name == "Deb's function") {
      return false;
    }
    
    // Default: PaGMO uses minimization
    return true;
  }
};

// Roulette Wheel Selection
class RouletteWheelSelection : public SelectionOperator {
public:
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // For minimization problems, invert fitness values
    std::vector<double> adjusted_fitness = fitness;
    const double min_fitness = *std::min_element(fitness.begin(), fitness.end());
    
    // Convert to maximization problem if needed and ensure positive values
    if (min_fitness < 0) {
      const double offset = std::abs(min_fitness) + 1.0;
      for (auto& f : adjusted_fitness) {
        f += offset;
      }
    }
    
    // Calculate total fitness
    const double total_fitness = std::accumulate(adjusted_fitness.begin(), adjusted_fitness.end(), 0.0);
    
    if (total_fitness <= 0.0) {
      // If total fitness is zero or negative, use uniform selection
      std::uniform_int_distribution<pagmo::population::size_type> dist(0, pop_size - 1);
      for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
        selected.push_back(dist(rng));
      }
      return selected;
    }
    
    // Create distribution based on fitness
    std::uniform_real_distribution<double> dist(0.0, total_fitness);
    
    // Select individuals
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      const double r = dist(rng);
      double running_sum = 0.0;
      
      for (pagmo::population::size_type j = 0; j < pop_size; ++j) {
        running_sum += adjusted_fitness[j];
        if (running_sum >= r) {
          selected.push_back(j);
          break;
        }
      }
    }
    
    return selected;
  }
};

// Stochastic Universal Sampling
class StochasticUniversalSampling : public SelectionOperator {
public:
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // For minimization problems, invert fitness values
    std::vector<double> adjusted_fitness = fitness;
    const double min_fitness = *std::min_element(fitness.begin(), fitness.end());
    
    // Convert to maximization problem if needed
    if (min_fitness < 0) {
      const double offset = std::abs(min_fitness) + 1.0;
      for (auto& f : adjusted_fitness) {
        f += offset;
      }
    }
    
    // Calculate total fitness
    const double total_fitness = std::accumulate(adjusted_fitness.begin(), adjusted_fitness.end(), 0.0);
    
    if (total_fitness <= 0.0) {
      // If total fitness is zero or negative, use uniform selection
      std::uniform_int_distribution<pagmo::population::size_type> dist(0, pop_size - 1);
      for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
        selected.push_back(dist(rng));
      }
      return selected;
    }
    
    // Distance between pointers
    const double distance = total_fitness / selection_size;
    
    // Generate random starting point
    std::uniform_real_distribution<double> dist(0.0, distance);
    const double start = dist(rng);
    
    // Place pointers
    std::vector<double> pointers(selection_size);
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      pointers[i] = start + i * distance;
    }
    
    // Select individuals
    for (const double pointer : pointers) {
      double running_sum = 0.0;
      
      for (pagmo::population::size_type j = 0; j < pop_size; ++j) {
        running_sum += adjusted_fitness[j];
        if (running_sum >= pointer) {
          selected.push_back(j);
          break;
        }
      }
    }
    
    return selected;
  }
};

// Tournament selection with replacement
class TournamentWithReplacement : public SelectionOperator {
private:
  pagmo::population::size_type tournament_size;
  
public:
  explicit TournamentWithReplacement(pagmo::population::size_type t_size) : tournament_size(t_size) {}
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // Distribution for selecting individuals for tournament
    std::uniform_int_distribution<pagmo::population::size_type> dist(0, pop_size - 1);
    
    // Run tournaments
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      pagmo::population::size_type best_idx = dist(rng);
      double best_fitness = fitness[best_idx];
      
      // Conduct tournament
      for (pagmo::population::size_type j = 1; j < tournament_size; ++j) {
        const auto idx = dist(rng);
        if (fitness[idx] < best_fitness) {  // Assuming minimization, adjust if maximizing
          best_idx = idx;
          best_fitness = fitness[idx];
        }
      }
      
      selected.push_back(best_idx);
    }
    
    return selected;
  }
};

// Tournament selection without replacement
class TournamentWithoutReplacement : public SelectionOperator {
private:
  pagmo::population::size_type tournament_size;
  
public:
  explicit TournamentWithoutReplacement(pagmo::population::size_type t_size) : tournament_size(t_size) {}
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // Run tournaments
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      // Create tournament participants without replacement
      std::vector<pagmo::population::size_type> tournament;
      tournament.reserve(tournament_size);
      
      // Sample without replacement
      std::vector<pagmo::population::size_type> candidates(pop_size);
      std::iota(candidates.begin(), candidates.end(), 0);  // Fill with 0, 1, 2, ...
      std::shuffle(candidates.begin(), candidates.end(), rng);
      
      // Take first 'tournament_size' elements
      const auto actual_size = std::min(tournament_size, pop_size);
      tournament.assign(candidates.begin(), candidates.begin() + actual_size);
      
      // Find the best from the tournament
      pagmo::population::size_type best_idx = tournament[0];
      double best_fitness = fitness[best_idx];
      
      for (size_t j = 1; j < tournament.size(); ++j) {
        const auto idx = tournament[j];
        if (fitness[idx] < best_fitness) {  // Assuming minimization
          best_idx = idx;
          best_fitness = fitness[idx];
        }
      }
      
      selected.push_back(best_idx);
    }
    
    return selected;
  }
};

// Tournament with participation probability
class TournamentWithParticipation : public SelectionOperator {
private:
  pagmo::population::size_type tournament_size;
  
public:
  explicit TournamentWithParticipation(pagmo::population::size_type t_size) : tournament_size(t_size) {}
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // Distribution for selecting individuals for tournament
    std::uniform_int_distribution<pagmo::population::size_type> dist(0, pop_size - 1);
    
    // Participation probability distribution
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    constexpr double participation_prob = 0.75;  // 75% chance to participate
    
    // Run tournaments
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      std::vector<pagmo::population::size_type> tournament;
      
      // Fill tournament with participants based on participation probability
      size_t attempts = 0;
      const size_t max_attempts = pop_size * 2;  // Avoid infinite loop
      
      while (tournament.size() < tournament_size && attempts < max_attempts) {
        const auto idx = dist(rng);
        if (prob_dist(rng) <= participation_prob) {
          tournament.push_back(idx);
        }
        ++attempts;
      }
      
      if (tournament.empty()) {
        // If no participants, select randomly
        selected.push_back(dist(rng));
      } else {
        // Find the best from the tournament
        pagmo::population::size_type best_idx = tournament[0];
        double best_fitness = fitness[best_idx];
        
        for (size_t j = 1; j < tournament.size(); ++j) {
          const auto idx = tournament[j];
          if (fitness[idx] < best_fitness) {  // Assuming minimization
            best_idx = idx;
            best_fitness = fitness[idx];
          }
        }
        
        selected.push_back(best_idx);
      }
    }
    
    return selected;
  }
};

// Exponential Ranking Selection (RWS-based)
class ExponentialRankRWS : public SelectionOperator {
private:
  double c;  // Base value
  
public:
  explicit ExponentialRankRWS(double c_val) : c(c_val) {}
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // Create indexes and sort by fitness (assuming minimization)
    std::vector<pagmo::population::size_type> indices(pop_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&fitness](size_t a, size_t b) {
      return fitness[a] < fitness[b];  // Ascending order for minimization
    });
    
    // Calculate exponential ranks
    std::vector<double> ranks(pop_size);
    for (pagmo::population::size_type i = 0; i < pop_size; ++i) {
      ranks[indices[i]] = c * std::pow(1.0 - c, i);
    }
    
    // Calculate total rank sum
    const double total_rank = std::accumulate(ranks.begin(), ranks.end(), 0.0);
    
    // Perform selection using RWS on ranks
    std::uniform_real_distribution<double> dist(0.0, total_rank);
    
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      const double r = dist(rng);
      double running_sum = 0.0;
      
      for (pagmo::population::size_type j = 0; j < pop_size; ++j) {
        running_sum += ranks[j];
        if (running_sum >= r) {
          selected.push_back(j);
          break;
        }
      }
    }
    
    return selected;
  }
};

// Exponential Ranking Selection (SUS-based)
class ExponentialRankSUS : public SelectionOperator {
private:
  double c;  // Base value
  
public:
  explicit ExponentialRankSUS(double c_val) : c(c_val) {}
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // Create indexes and sort by fitness (assuming minimization)
    std::vector<pagmo::population::size_type> indices(pop_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&fitness](size_t a, size_t b) {
      return fitness[a] < fitness[b];  // Ascending order for minimization
    });
    
    // Calculate exponential ranks
    std::vector<double> ranks(pop_size);
    for (pagmo::population::size_type i = 0; i < pop_size; ++i) {
      ranks[indices[i]] = c * std::pow(1.0 - c, i);
    }
    
    // Calculate total rank sum
    const double total_rank = std::accumulate(ranks.begin(), ranks.end(), 0.0);
    
    // Distance between pointers for SUS
    const double distance = total_rank / selection_size;
    
    // Generate random starting point
    std::uniform_real_distribution<double> dist(0.0, distance);
    const double start = dist(rng);
    
    // Place pointers
    std::vector<double> pointers(selection_size);
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      pointers[i] = start + i * distance;
    }
    
    // Select individuals
    for (const double pointer : pointers) {
      double running_sum = 0.0;
      
      for (pagmo::population::size_type j = 0; j < pop_size; ++j) {
        running_sum += ranks[j];
        if (running_sum >= pointer) {
          selected.push_back(j);
          break;
        }
      }
    }
    
    return selected;
  }
};

// Linear Ranking Selection (RWS-based)
class LinearRankRWS : public SelectionOperator {
private:
  double beta;  // Selection pressure parameter (1.0 <= beta <= 2.0)
  
public:
  explicit LinearRankRWS(double b) : beta(b) {}
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // Create indexes and sort by fitness (assuming minimization)
    std::vector<pagmo::population::size_type> indices(pop_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&fitness](size_t a, size_t b) {
      return fitness[a] < fitness[b];  // Ascending order for minimization
    });
    
    // Calculate linear ranks using beta
    std::vector<double> ranks(pop_size);
    for (pagmo::population::size_type i = 0; i < pop_size; ++i) {
      // Formula: 2-beta + 2*(beta-1)*(i/(N-1))
      // For i=0 (best): rank = 2-beta
      // For i=N-1 (worst): rank = beta
      const double rank_value = 2.0 - beta + 2.0 * (beta - 1.0) * (static_cast<double>(i) / (pop_size - 1));
      ranks[indices[i]] = rank_value;
    }
    
    // Calculate total rank sum
    const double total_rank = std::accumulate(ranks.begin(), ranks.end(), 0.0);
    
    // Perform selection using RWS on ranks
    std::uniform_real_distribution<double> dist(0.0, total_rank);
    
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      const double r = dist(rng);
      double running_sum = 0.0;
      
      for (pagmo::population::size_type j = 0; j < pop_size; ++j) {
        running_sum += ranks[j];
        if (running_sum >= r) {
          selected.push_back(j);
          break;
        }
      }
    }
    
    return selected;
  }
};

// Linear Ranking Selection (SUS-based)
class LinearRankSUS : public SelectionOperator {
private:
  double beta;  // Selection pressure parameter (1.0 <= beta <= 2.0)
  
public:
  explicit LinearRankSUS(double b) : beta(b) {}
  
  std::vector<pagmo::population::size_type> select(
      const std::vector<double>& fitness, 
      const pagmo::population::size_type selection_size,
      pagmo::detail::random_engine_type& rng) const override {
    
    const auto pop_size = fitness.size();
    std::vector<pagmo::population::size_type> selected;
    selected.reserve(selection_size);
    
    // Create indexes and sort by fitness (assuming minimization)
    std::vector<pagmo::population::size_type> indices(pop_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&fitness](size_t a, size_t b) {
      return fitness[a] < fitness[b];  // Ascending order for minimization
    });
    
    // Calculate linear ranks using beta
    std::vector<double> ranks(pop_size);
    for (pagmo::population::size_type i = 0; i < pop_size; ++i) {
      // Formula: 2-beta + 2*(beta-1)*(i/(N-1))
      // For i=0 (best): rank = 2-beta
      // For i=N-1 (worst): rank = beta
      const double rank_value = 2.0 - beta + 2.0 * (beta - 1.0) * (static_cast<double>(i) / (pop_size - 1));
      ranks[indices[i]] = rank_value;
    }
    
    // Calculate total rank sum
    const double total_rank = std::accumulate(ranks.begin(), ranks.end(), 0.0);
    
    // Distance between pointers for SUS
    const double distance = total_rank / selection_size;
    
    // Generate random starting point
    std::uniform_real_distribution<double> dist(0.0, distance);
    const double start = dist(rng);
    
    // Place pointers
    std::vector<double> pointers(selection_size);
    for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
      pointers[i] = start + i * distance;
    }
    
    // Select individuals
    for (const double pointer : pointers) {
      double running_sum = 0.0;
      
      for (pagmo::population::size_type j = 0; j < pop_size; ++j) {
        running_sum += ranks[j];
        if (running_sum >= pointer) {
          selected.push_back(j);
          break;
        }
      }
    }
    
    return selected;
  }
};

// Factory function to create selection operator based on the selection method
std::unique_ptr<SelectionOperator> create_selection_operator(SelectionMethod method) {
  switch (method) {
    case SelectionMethod::SUS:
      return std::make_unique<StochasticUniversalSampling>();
      
    case SelectionMethod::RWS:
      return std::make_unique<RouletteWheelSelection>();
      
    case SelectionMethod::Tournament:
      return std::make_unique<TournamentWithReplacement>(2);  // Default tournament size 2
      
    case SelectionMethod::TournWITH_t2:
      return std::make_unique<TournamentWithReplacement>(2);
      
    case SelectionMethod::TournWITH_t4:
      return std::make_unique<TournamentWithReplacement>(4);
      
    case SelectionMethod::TournWITHOUT_t2:
      return std::make_unique<TournamentWithoutReplacement>(2);
      
    case SelectionMethod::TournWITHOUT_t4:
      return std::make_unique<TournamentWithoutReplacement>(4);
      
    case SelectionMethod::TournWITHPART_t2:
      return std::make_unique<TournamentWithParticipation>(2);
      
    case SelectionMethod::ExpRankRWS_c0_9801:
      return std::make_unique<ExponentialRankRWS>(0.9801);
      
    case SelectionMethod::ExpRankRWS_c0_9606:
      return std::make_unique<ExponentialRankRWS>(0.9606);
      
    case SelectionMethod::ExpRankSUS_c0_9801:
      return std::make_unique<ExponentialRankSUS>(0.9801);
      
    case SelectionMethod::ExpRankSUS_c0_9606:
      return std::make_unique<ExponentialRankSUS>(0.9606);
      
    case SelectionMethod::LinRankRWS_b2:
      return std::make_unique<LinearRankRWS>(2.0);
      
    case SelectionMethod::LinRankRWS_b1_6:
      return std::make_unique<LinearRankRWS>(1.6);
      
    case SelectionMethod::LinRankSUS_b2:
      return std::make_unique<LinearRankSUS>(2.0);
      
    case SelectionMethod::LinRankSUS_b1_6:
      return std::make_unique<LinearRankSUS>(1.6);
      
    default:
      throw std::runtime_error("Unknown selection method");
  }
}

// Constants for determining successful runs
constexpr double DELTA = 0.01;  // Fitness threshold for success
constexpr double SIGMA = 0.01;  // Distance threshold for success

// Function to run a single GA experiment with given parameters
run_stats run_experiment(const ga_config &config, unsigned seed) {
  run_stats stats;

  // Create the problem
  pagmo::problem prob;
  switch (config.problem_type) {
    case ProblemType::Ackley:
      // We negate since PaGMO minimizes by default and we want to maximize
      prob = pagmo::problem{pagmo::ackley{config.dimension}};
      break;
    case ProblemType::Deb:
      prob = pagmo::problem{deb_func{config.dimension}};
      break;
    default:
      throw std::runtime_error("Unknown problem type");
  }

  // Set up algorithm
  pagmo::algorithm algo{pagmo::sga(
      config.generations_per_evolution,  // Generations per evolution
      config.crossover_prob,             // Crossover probability
      1.0,                               // eta_c (distribution index for SBX)
      config.mutation_prob,              // Mutation probability
      1.0,                               // param_m (mutation parameter)
      2,                               // param_s (selection parameter - tournament size)
      to_string(config.crossover_type),  // Convert enum to string for PaGMO
      to_string(config.mutation_type),   // Convert enum to string for PaGMO
      to_string(config.selection_method),// Convert enum to string for PaGMO
      seed                               // Random seed
      )};

  // Set up archipelago
  pagmo::archipelago archi{config.island_count, algo, prob,
                           config.population_size, seed};

  // Run the evolutions
  auto start_time = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < config.total_evolutions; ++i) {
    archi.evolve();
    archi.wait_check();
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  stats.execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time).count();

  // Collect statistics
  stats.iterations = config.generations_per_evolution * config.total_evolutions;

  // Find the best solution across all islands
  pagmo::vector_double best_x;
  double best_f = std::numeric_limits<double>::lowest();
  if (config.problem_type == ProblemType::Ackley) {
    best_f = std::numeric_limits<double>::max();
  }

  double total_fitness = 0.0;
  int total_individuals = 0;

  // Get results from all islands and find the best
  for (const auto &isl : archi) {
    const auto &pop = isl.get_population();

    // Calculate average fitness for this island
    for (pagmo::population::size_type i = 0; i < pop.size(); ++i) {
      double fitness;
      if (config.problem_type == ProblemType::Ackley) {
        fitness = pop.get_f()[i][0];  // Minimizing
      } else {
        fitness =
            -pop.get_f()[i]
                        [0];  // Negating since Deb is set up for minimization
      }
      total_fitness += fitness;
      total_individuals++;
    }

    // Check if this island has the best solution
    if (config.problem_type == ProblemType::Ackley) {
      // For Ackley, lower is better (minimization)
      if (pop.champion_f()[0] < best_f) {
        best_f = pop.champion_f()[0];
        best_x = pop.champion_x();
      }
    } else {
      // For Deb, higher is better (maximization despite problem setup)
      if (-pop.champion_f()[0] > best_f) {
        best_f = -pop.champion_f()[0];
        best_x = pop.champion_x();
      }
    }
  }

  // Calculate average fitness across all individuals
  stats.f_avg = total_fitness / total_individuals;

  // Store the best solution
  stats.f_max = best_f;
  stats.x_max = best_x;

  // Calculate convergence (population homogeneity)
  // We measure the average distance from individuals to the best solution
  double total_distance = 0.0;
  for (const auto &isl : archi) {
    const auto &pop = isl.get_population();
    for (pagmo::population::size_type i = 0; i < pop.size(); ++i) {
      total_distance += euclidean_distance(pop.get_x()[i], best_x);
    }
  }
  stats.convergence =
      1.0 - (total_distance / (total_individuals * config.dimension));

  // Get optimal solution for comparison
  pagmo::vector_double optimal_x;
  double optimal_f;

  if (config.problem_type == ProblemType::Ackley) {
    // Ackley optimum is at the origin (0,0,...,0) with f=0
    optimal_x = pagmo::vector_double(config.dimension, 0.0);
    optimal_f = 0.0;
  } else {
    // Deb optimum is at (0.1, 0.1, ..., 0.1) with f=dimension
    optimal_x = pagmo::vector_double(config.dimension, 0.1);
    optimal_f = config.dimension;
  }

  // Calculate peak accuracy and distance accuracy
  if (config.problem_type == ProblemType::Ackley) {
    // For Ackley, f_optimal = 0 (minimum), so we calculate differently
    stats.peak_accuracy = 1.0 / (1.0 + std::abs(best_f - optimal_f));
  } else {
    stats.peak_accuracy = best_f / optimal_f;
  }
  stats.distance_accuracy = 1.0 / (1.0 + euclidean_distance(best_x, optimal_x));

  // Determine if run was successful
  if (config.problem_type == ProblemType::Ackley) {
    stats.is_successful = (std::abs(best_f - optimal_f) <= DELTA) &&
                          (euclidean_distance(best_x, optimal_x) <= SIGMA);
  } else {
    stats.is_successful = (std::abs(best_f - optimal_f) <= DELTA * optimal_f) &&
                          (euclidean_distance(best_x, optimal_x) <= SIGMA);
  }

  // Approximate fitness evaluations (population_size * islands * generations)
  stats.fitness_evals = config.population_size * config.island_count *
                        config.generations_per_evolution *
                        config.total_evolutions;

  return stats;
}

// Function to calculate aggregate statistics
aggregate_stats calculate_aggregate_stats(const std::vector<run_stats> &runs) {
  aggregate_stats agg;

  // Count successful runs
  std::vector<run_stats> successful_runs;
  std::vector<run_stats> failed_runs;
  successful_runs.reserve(runs.size());
  failed_runs.reserve(runs.size());

  for (const auto &run : runs) {
    if (run.is_successful) {
      successful_runs.push_back(run);
    } else {
      failed_runs.push_back(run);
    }
  }

  agg.success_rate = 100.0 * successful_runs.size() / runs.size();

  // Process successful runs
  if (!successful_runs.empty()) {
    // Calculate min, max, avg for iterations
    std::vector<double> iterations_vec;
    iterations_vec.reserve(successful_runs.size());
    agg.min_iterations = successful_runs[0].iterations;
    agg.max_iterations = successful_runs[0].iterations;

    for (const auto &run : successful_runs) {
      agg.min_iterations = std::min(agg.min_iterations, run.iterations);
      agg.max_iterations = std::max(agg.max_iterations, run.iterations);
      iterations_vec.push_back(run.iterations);
    }
    agg.avg_iterations =
        std::accumulate(iterations_vec.begin(), iterations_vec.end(), 0.0) /
        iterations_vec.size();
    agg.std_iterations = calculate_std_dev(iterations_vec, agg.avg_iterations);

    // Calculate min, max, avg for fitness evaluations
    std::vector<double> evals_vec;
    evals_vec.reserve(successful_runs.size());
    agg.min_evals = successful_runs[0].fitness_evals;
    agg.max_evals = successful_runs[0].fitness_evals;

    for (const auto &run : successful_runs) {
      agg.min_evals = std::min(agg.min_evals, run.fitness_evals);
      agg.max_evals = std::max(agg.max_evals, run.fitness_evals);
      evals_vec.emplace_back(run.fitness_evals);
    }
    agg.avg_evals = std::accumulate(evals_vec.begin(), evals_vec.end(), 0.0) /
                    evals_vec.size();
    agg.std_evals = calculate_std_dev(evals_vec, agg.avg_evals);
    
    // Calculate min, max, avg for execution time
    std::vector<double> exec_time_vec;
    exec_time_vec.reserve(successful_runs.size());
    agg.min_exec_time = successful_runs[0].execution_time_ms;
    agg.max_exec_time = successful_runs[0].execution_time_ms;
    
    for (const auto &run : successful_runs) {
      agg.min_exec_time = std::min(agg.min_exec_time, run.execution_time_ms);
      agg.max_exec_time = std::max(agg.max_exec_time, run.execution_time_ms);
      exec_time_vec.emplace_back(static_cast<double>(run.execution_time_ms));
    }
    agg.avg_exec_time = std::accumulate(exec_time_vec.begin(), exec_time_vec.end(), 0.0) /
                       exec_time_vec.size();
    agg.std_exec_time = calculate_std_dev(exec_time_vec, agg.avg_exec_time);

    // Calculate min, max, avg for f_max
    std::vector<double> f_max_vec;
    f_max_vec.reserve(successful_runs.size());
    agg.min_f_max = successful_runs[0].f_max;
    agg.max_f_max = successful_runs[0].f_max;

    for (const auto &run : successful_runs) {
      agg.min_f_max = std::min(agg.min_f_max, run.f_max);
      agg.max_f_max = std::max(agg.max_f_max, run.f_max);
      f_max_vec.emplace_back(run.f_max);
    }
    agg.avg_f_max = std::accumulate(f_max_vec.begin(), f_max_vec.end(), 0.0) /
                    f_max_vec.size();
    agg.std_f_max = calculate_std_dev(f_max_vec, agg.avg_f_max);

    // Calculate min, max, avg for f_avg
    std::vector<double> f_avg_vec;
    f_avg_vec.reserve(successful_runs.size());
    agg.min_f_avg = successful_runs[0].f_avg;
    agg.max_f_avg = successful_runs[0].f_avg;

    for (const auto &run : successful_runs) {
      agg.min_f_avg = std::min(agg.min_f_avg, run.f_avg);
      agg.max_f_avg = std::max(agg.max_f_avg, run.f_avg);
      f_avg_vec.emplace_back(run.f_avg);
    }
    agg.avg_f_avg = std::accumulate(f_avg_vec.begin(), f_avg_vec.end(), 0.0) /
                    f_avg_vec.size();
    agg.std_f_avg = calculate_std_dev(f_avg_vec, agg.avg_f_avg);

    // Calculate min, max, avg for convergence
    std::vector<double> conv_vec;
    conv_vec.reserve(successful_runs.size());
    agg.min_convergence = successful_runs[0].convergence;
    agg.max_convergence = successful_runs[0].convergence;

    for (const auto &run : successful_runs) {
      agg.min_convergence = std::min(agg.min_convergence, run.convergence);
      agg.max_convergence = std::max(agg.max_convergence, run.convergence);
      conv_vec.emplace_back(run.convergence);
    }
    agg.avg_convergence =
        std::accumulate(conv_vec.begin(), conv_vec.end(), 0.0) /
        conv_vec.size();
    agg.std_convergence = calculate_std_dev(conv_vec, agg.avg_convergence);

    // Calculate min, max, avg for peak accuracy
    std::vector<double> pa_vec;
    pa_vec.reserve(successful_runs.size());
    agg.min_peak_accuracy = successful_runs[0].peak_accuracy;
    agg.max_peak_accuracy = successful_runs[0].peak_accuracy;

    for (const auto &run : successful_runs) {
      agg.min_peak_accuracy =
          std::min(agg.min_peak_accuracy, run.peak_accuracy);
      agg.max_peak_accuracy =
          std::max(agg.max_peak_accuracy, run.peak_accuracy);
      pa_vec.emplace_back(run.peak_accuracy);
    }
    agg.avg_peak_accuracy =
        std::accumulate(pa_vec.begin(), pa_vec.end(), 0.0) / pa_vec.size();
    agg.std_peak_accuracy = calculate_std_dev(pa_vec, agg.avg_peak_accuracy);

    // Calculate min, max, avg for distance accuracy
    std::vector<double> da_vec;
    da_vec.reserve(successful_runs.size());
    agg.min_dist_accuracy = successful_runs[0].distance_accuracy;
    agg.max_dist_accuracy = successful_runs[0].distance_accuracy;

    for (const auto &run : successful_runs) {
      agg.min_dist_accuracy =
          std::min(agg.min_dist_accuracy, run.distance_accuracy);
      agg.max_dist_accuracy =
          std::max(agg.max_dist_accuracy, run.distance_accuracy);
      da_vec.emplace_back(run.distance_accuracy);
    }
    agg.avg_dist_accuracy =
        std::accumulate(da_vec.begin(), da_vec.end(), 0.0) / da_vec.size();
    agg.std_dist_accuracy = calculate_std_dev(da_vec, agg.avg_dist_accuracy);
  }

  // Process failed runs
  if (!failed_runs.empty()) {
    // Calculate min, max, avg for iterations (failed runs)
    std::vector<double> iterations_f_vec;
    iterations_f_vec.reserve(failed_runs.size());
    agg.min_iterations_f = failed_runs[0].iterations;
    agg.max_iterations_f = failed_runs[0].iterations;

    for (const auto &run : failed_runs) {
      agg.min_iterations_f = std::min(agg.min_iterations_f, run.iterations);
      agg.max_iterations_f = std::max(agg.max_iterations_f, run.iterations);
      iterations_f_vec.emplace_back(run.iterations);
    }
    agg.avg_iterations_f =
        std::accumulate(iterations_f_vec.begin(), iterations_f_vec.end(), 0.0) /
        iterations_f_vec.size();
    agg.std_iterations_f =
        calculate_std_dev(iterations_f_vec, agg.avg_iterations_f);
  }

  return agg;
}

// Thread-safe queue for asynchronous I/O operations
template <typename T>
class ThreadSafeQueue {
 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  bool done_ = false;

 public:
  ThreadSafeQueue() = default;

  void push(T item) {
    std::lock_guard lock(mutex_);
    queue_.push(std::move(item)); // Using push with move since we have a complete item
    cond_.notify_one();
  }
  
  // Variadic template for emplacing items directly
  template<typename... Args>
  void emplace(Args&&... args) {
    std::lock_guard lock(mutex_);
    queue_.emplace(std::forward<Args>(args)...); // Construct in-place
    cond_.notify_one();
  }

  bool try_pop(T &item) {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) {
      return nullptr;
    }
    std::shared_ptr<T> res(std::make_shared<T>(std::move(queue_.front())));
    queue_.pop();
    return res;
  }

  void wait_and_pop(T &item) {
    std::unique_lock lock(mutex_);
    cond_.wait(lock, [this]() { return !queue_.empty() || done_; });
    if (done_ && queue_.empty()) {
      return;
    }
    item = std::move(queue_.front());
    queue_.pop();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock lock(mutex_);
    cond_.wait(lock, [this]() { return !queue_.empty() || done_; });
    if (done_ && queue_.empty()) {
      return nullptr;
    }
    std::shared_ptr<T> res(std::make_shared<T>(std::move(queue_.front())));
    queue_.pop();
    return res;
  }

  bool empty() const {
    std::lock_guard lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
  }

  void done() {
    std::lock_guard lock(mutex_);
    done_ = true;
    cond_.notify_all();
  }

  bool is_done() const {
    std::lock_guard lock(mutex_);
    return done_;
  }
};

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

// Function to run experiments with thread-local storage
std::vector<run_stats> run_experiment_batch(
    const int config_id, const ga_config &config,
    ThreadSafeQueue<DetailedResultEntry> &detailed_queue,
    std::mutex &cout_mutex, const unsigned num_runs) {
  {
    std::lock_guard lock(cout_mutex);
    std::cout << "Starting config " << config_id << ": " << to_string(config.problem_type)
              << ", dim=" << config.dimension
              << ", pop=" << config.population_size
              << ", islands=" << config.island_count << std::endl;
  }

  // Thread-local storage for results
  std::vector<run_stats> local_runs;
  local_runs.reserve(num_runs);

  for (unsigned run = 0; run < num_runs; ++run) {
    {
      std::lock_guard lock(cout_mutex);
      std::cout << "  Config " << config_id << " - Run " << run + 1 << "/"
                << num_runs << "..." << std::flush;
    }

    // Use a different seed for each run
    const unsigned seed = config_id * 1000 + run;
    auto stats = run_experiment(config, seed);

    // Queue detailed results for asynchronous writing
    detailed_queue.emplace(config_id, run + 1, stats); // Using emplace for in-place construction

    // Store locally
    local_runs.push_back(stats);

    {
      std::lock_guard lock(cout_mutex);
      std::cout << (stats.is_successful ? " Success" : " Failure") << std::endl;
    }
  }

  {
    std::lock_guard lock(cout_mutex);
    std::cout << "Finished processing config " << config_id << std::endl;
  }

  return local_runs;
}

// Legacy function maintained for compatibility
void run_config(const int config_id, const ga_config &config,
                std::ofstream &detailed_csv, std::ofstream &summary_csv,
                const unsigned num_runs) {
  std::cout << "Running config " << config_id << ": " << to_string(config.problem_type)
            << ", dim=" << config.dimension
            << ", pop=" << config.population_size
            << ", islands=" << config.island_count << std::endl;

  std::vector<run_stats> runs;

  for (unsigned run = 0; run < num_runs; ++run) {
    std::cout << "  Run " << run + 1 << "/" << num_runs << "..." << std::flush;

    // Use a different seed for each run
    const unsigned seed = config_id * 1000 + run;
    auto stats = run_experiment(config, seed);

    // Write detailed results for this run
    detailed_csv << config_id << "," << run + 1 << "," << stats.to_csv()
                 << std::endl;

    runs.push_back(stats);

    std::cout << (stats.is_successful ? " Success" : " Failure") << std::endl;
  }

  // Calculate aggregate statistics
  const auto agg_stats = calculate_aggregate_stats(runs);

  // Write summary for this config
  summary_csv << config_id << "," << config.to_csv(config_id) << ","
              << agg_stats.to_csv() << std::endl;
}

int main() {
  std::cout << "Genetic Algorithm Optimization - Ackley and Deb Functions"
            << std::endl;
  std::cout << "========================================================="
            << std::endl;

  // Create output directory
  std::system("mkdir -p results");

  // Setup CSV files for output
  std::ofstream detailed_csv("results/detailed_results.csv");
  std::ofstream summary_csv("results/summary_results.csv");

  // Write headers
  detailed_csv << "Config_ID,Run_ID,Is_Successful,Iterations,Fitness_Evals,Execution_Time_ms,F_"
                  "max,X_max,F_avg,Convergence,Peak_Accuracy,Distance_Accuracy"
               << std::endl;
  summary_csv << ga_config::csv_header() << "," << aggregate_stats::csv_header()
              << std::endl;

  // Configuration options to test
  std::vector<ga_config> configs;
  int config_id = 0;

  // Number of runs as specified in the task
  constexpr unsigned NUM_RUNS = 100;  // As required in TASK.md, line 158

  // ==============================
  // ACKLEY FUNCTION CONFIGURATIONS
  // ==============================

  // Following the task requirements to analyze in order:
  // 1. N=100, n=1, 2, 3, 5
  // 2. Then repeat for N=200, 300, 400

  // Define population sizes to test (as per TASK.md)
  constexpr std::array population_sizes = {100, 200, 300, 400};

  // Define dimensions to test (as per TASK.md)
  constexpr std::array dimensions = {1, 2, 3, 5};

  // Encoding methods
  constexpr std::array encoding_methods = {
      EncodingMethod::StandardBinary,
      EncodingMethod::GrayCode
  };

  // Crossover types (as per TASK.md)
  constexpr std::array crossover_types = {
      CrossoverType::Single,   // Single-point crossover
      CrossoverType::Uniform   // Standard uniform crossover
  };

  // Crossover probabilities (as per TASK.md)
  constexpr std::array crossover_probs = {0.0, 0.6, 0.8, 1.0};

  // Mutation probabilities (as per TASK.md, for different dimensions)
  std::map<unsigned, std::array<double, 3>> mutation_probs_by_dim = {
      {1, {0.0, 0.001, 0.01}},    // n=1
      {2, {0.0, 0.0005, 0.005}},  // n=2
      {3, {0.0, 0.0003, 0.003}},  // n=3
      {5, {0.0, 0.0002, 0.0005}}  // n=5
  };

  // Mutation types
  constexpr std::array mutation_types = {
      MutationType::Polynomial,  // Using polynomial mutation for density-based mutation
      MutationType::Density      // Explicit density-based mutation
  };

  // Reproduction types
  constexpr std::array reproduction_types = {
      ReproductionType::Generational,  // Generational replacement
      ReproductionType::SteadyState    // Steady-state replacement
  };

  // Generation gap values for steady-state (as per TASK.md)
  constexpr std::array generation_gaps = {0.05, 0.1, 0.2, 0.5};

  // Parent selection methods for steady-state
  constexpr std::array parent_selection_methods = {
      ParentSelectionMethod::Elite,  // Select best individuals
      ParentSelectionMethod::RWS     // Roulette wheel selection
  };

  // Replacement methods for steady-state
  constexpr std::array replacement_methods = {
      ReplacementMethod::WorstComma,  // Replace worst from parent population
      ReplacementMethod::RandComma,   // Replace random from parent population
      ReplacementMethod::WorstPlus,   // Replace worst from combined population
      ReplacementMethod::RandPlus     // Replace random from combined population
  };

  // All selection methods (same for each population size as per TASK.md)
  constexpr std::array selection_methods = {
      SelectionMethod::SUS,
      SelectionMethod::RWS,
      SelectionMethod::TournWITH_t2,
      SelectionMethod::TournWITHOUT_t2,
      SelectionMethod::TournWITHPART_t2,
      SelectionMethod::ExpRankRWS_c0_9801,
      SelectionMethod::ExpRankSUS_c0_9801,
      SelectionMethod::LinRankRWS_b2,
      SelectionMethod::LinRankSUS_b2,
      SelectionMethod::TournWITH_t4,
      SelectionMethod::TournWITHOUT_t4,
      SelectionMethod::ExpRankRWS_c0_9606,
      SelectionMethod::ExpRankSUS_c0_9606,
      SelectionMethod::LinRankRWS_b1_6,
      SelectionMethod::LinRankSUS_b1_6
  };

  // Define problem types to test
  constexpr std::array problem_types = {
      ProblemType::Ackley,
      ProblemType::Deb
  };

  // Loop through all combinations of parameters for all problems
  for (const auto problem_type : problem_types) {
    for (auto pop_size : population_sizes) {
      for (auto dim : dimensions) {
        // Get the appropriate mutation probabilities for this dimension
        const auto &mutation_probs = mutation_probs_by_dim[dim];
        
        for (const auto encoding_method : encoding_methods) {
          for (const auto &crossover_type : crossover_types) {
            for (double crossover_prob : crossover_probs) {
              for (const auto mutation_type : mutation_types) {
                for (double mutation_prob : mutation_probs) {
                  for (const auto reproduction_type : reproduction_types) {
                    if (reproduction_type == ReproductionType::Generational) {
                      // For generational reproduction type
                      for (const auto &selection_method : selection_methods) {
                        // Create config for generational reproduction type
                        ga_config config;
                        config.problem_type = problem_type;
                        config.dimension = dim;
                        config.population_size = pop_size;
                        config.island_count = 16;  // Using 16 islands for parallelization
                        config.generations_per_evolution = 50;
                        config.total_evolutions = 10;
                        
                        config.encoding_method = encoding_method;
                        config.crossover_type = crossover_type;
                        config.crossover_prob = crossover_prob;
                        config.mutation_type = mutation_type;
                        config.mutation_prob = mutation_prob;
                        config.selection_method = selection_method;
                        
                        // Set to generational reproduction type
                        config.reproduction_type = reproduction_type;
                        config.generation_gap = 0.0;  // Not used for generational
                        
                        configs.push_back(config);
                        config_id++;
                      }
                    } else if (reproduction_type == ReproductionType::SteadyState) {
                      // For steady-state reproduction type
                      for (double gap : generation_gaps) {
                        for (const auto parent_selection : parent_selection_methods) {
                          for (const auto replacement_method : replacement_methods) {
                            // Create config for steady-state reproduction type
                            ga_config config;
                            config.problem_type = problem_type;
                            config.dimension = dim;
                            config.population_size = pop_size;
                            config.island_count = 16;  // Using 16 islands for parallelization
                            config.generations_per_evolution = 50;
                            config.total_evolutions = 10;
                            
                            // Default selection method for steady-state
                            config.selection_method = SelectionMethod::Tournament;
                            
                            config.encoding_method = encoding_method;
                            config.crossover_type = crossover_type;
                            config.crossover_prob = crossover_prob;
                            config.mutation_type = mutation_type;
                            config.mutation_prob = mutation_prob;
                            
                            // Set steady-state specific parameters
                            config.reproduction_type = reproduction_type;
                            config.generation_gap = gap;
                            config.parent_selection_method = parent_selection;
                            config.replacement_method = replacement_method;
                            
                            configs.push_back(config);
                            config_id++;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // For demonstration purposes, only run a small subset of configurations
  // to ensure the program works correctly
  // Store the original number of configurations before potentially reducing for
  // demo mode
  const size_t total_configs = configs.size();

  if constexpr (NUM_RUNS < 100) {
    // When running in demo mode
    std::vector<ga_config> demo_configs;

    // Select a small representative sample of configurations
    for (unsigned i = 0; i < std::min(static_cast<size_t>(8), configs.size());
         i += configs.size() / 8) {
      demo_configs.push_back(configs[i]);
    }

    std::cout << "Demo mode: Running " << demo_configs.size()
              << " configurations instead of the full set of " << total_configs
              << " configurations." << std::endl;
    configs = demo_configs;
  } else {
    std::cout << "Running all " << total_configs << " configurations with "
              << NUM_RUNS << " runs each." << std::endl;
  }

  // Print configuration breakdown
  size_t ackley_configs = 0;
  size_t deb_configs = 0;
  for (const auto &config : configs) {
    if (config.problem_type == ProblemType::Ackley) {
      ackley_configs++;
    } else if (config.problem_type == ProblemType::Deb) {
      deb_configs++;
    }
  }
  std::cout << "Configuration breakdown: " << ackley_configs
            << " Ackley configurations, " << deb_configs
            << " Deb configurations" << std::endl;
  std::cout << "Total expected runs: " << configs.size() * NUM_RUNS
            << std::endl;

  // Maximum number of concurrent configurations to run
  // Adjust this based on your hardware capabilities
  constexpr unsigned MAX_CONCURRENT_CONFIGS = 4;

  // Thread-safe queues for asynchronous I/O
  ThreadSafeQueue<DetailedResultEntry> detailed_queue;
  ThreadSafeQueue<SummaryResultEntry> summary_queue;
  std::mutex cout_mutex;

  // Start background I/O threads
  std::cout << "Starting I/O worker threads..." << std::endl;

  // Thread for writing detailed results
  std::thread detailed_writer([&]() {
    DetailedResultEntry entry;
    while (!detailed_queue.is_done() || !detailed_queue.empty()) {
      if (detailed_queue.try_pop(entry)) {
        // Write to CSV without contention
        detailed_csv << entry.config_id << "," << entry.run_id << ","
                     << entry.stats.to_csv() << std::endl;
      } else {
        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  });

  // Thread for writing summary results
  std::thread summary_writer([&]() {
    SummaryResultEntry entry;
    while (!summary_queue.is_done() || !summary_queue.empty()) {
      if (summary_queue.try_pop(entry)) {
        // Write to CSV without contention
        summary_csv << entry.config_id << ","
                    << entry.config.to_csv(entry.config_id) << ","
                    << entry.stats.to_csv() << std::endl;
      } else {
        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  });

  // Process configurations in batches
  for (int start = 0; start < configs.size(); start += MAX_CONCURRENT_CONFIGS) {
    const int end = std::min(static_cast<int>(configs.size()),
                             static_cast<int>(start + MAX_CONCURRENT_CONFIGS));

    {
      std::lock_guard lock(cout_mutex);
      std::cout << "Starting batch " << (start / MAX_CONCURRENT_CONFIGS) + 1
                << " of "
                << (configs.size() + MAX_CONCURRENT_CONFIGS - 1) /
                       MAX_CONCURRENT_CONFIGS
                << " (" << end - start << " configs)" << std::endl;
    }

    // Create a vector of threads and futures for this batch
    std::vector<std::thread> threads;
    std::vector<std::future<std::vector<run_stats>>> futures(end - start);

    // Launch a thread for each configuration in the batch
    for (int i = start; i < end; ++i) {
      const int config_idx = i - start;
      const int config_id = i + 1;

      // Create promise/future pair to get results
      std::promise<std::vector<run_stats>> promise;
      futures[config_idx] = promise.get_future();

      threads.emplace_back(
          [&, i, config_id, config_idx, p = std::move(promise)]() mutable {
            try {
              // Run experiments with thread-local storage
              std::vector<run_stats> runs = run_experiment_batch(
                  config_id, configs[i], detailed_queue, cout_mutex, NUM_RUNS);

              // Set promise with results
              p.set_value(std::move(runs));
            } catch (...) {
              // Handle any exceptions
              try {
                p.set_exception(std::current_exception());
              } catch (...) {
                // Promise already satisfied
              }
            }
          });
    }

    // Wait for all threads in this batch to complete
    for (auto &thread : threads) {
      thread.join();
    }

    // Process results from futures
    for (int i = 0; i < futures.size(); ++i) {
      try {
        // Get the results from the future
        std::vector<run_stats> runs = futures[i].get();

        // Calculate aggregate statistics
        const int config_id = start + i + 1;
        auto agg_stats = calculate_aggregate_stats(runs);

        // Queue summary results for asynchronous writing
        summary_queue.emplace(config_id, configs[start + i], agg_stats); // Using emplace for in-place construction

        {
          std::lock_guard lock(cout_mutex);
          std::cout << "Processed results for config " << config_id
                    << std::endl;
        }
      } catch (const std::exception &e) {
        std::lock_guard lock(cout_mutex);
        std::cerr << "Error processing config " << (start + i + 1) << ": "
                  << e.what() << std::endl;
      }
    }
  }

  // Signal I/O threads to finish
  {
    std::lock_guard lock(cout_mutex);
    std::cout << "All configurations processed. Waiting for I/O to complete..."
              << std::endl;
  }

  detailed_queue.done();
  summary_queue.done();

  // Wait for I/O threads to complete
  detailed_writer.join();
  summary_writer.join();

  std::cout
      << "All configurations completed. Results saved to 'results' directory."
      << std::endl;

  return 0;
}
