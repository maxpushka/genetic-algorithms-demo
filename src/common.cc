#include "include/common.h"

#include <cmath>
#include <numeric>

// Helper functions to convert enums to strings
std::string to_string(const ProblemType type) {
  switch (type) {
    case ProblemType::Ackley:
      return "Ackley";
    case ProblemType::Deb:
      return "Deb";
    default:
      return "Unknown";
  }
}

std::string to_string(const SelectionMethod method) {
  switch (method) {
    // Standard selection methods
    case SelectionMethod::SUS:
      return "sus";
    case SelectionMethod::RWS:
      return "rws";
    case SelectionMethod::Tournament:
      return "tournament";

    // Tournament variations
    case SelectionMethod::TournWITH_t2:
      return "tournament_with_t2";
    case SelectionMethod::TournWITH_t4:
      return "tournament_with_t4";
    case SelectionMethod::TournWITHOUT_t2:
      return "tournament_without_t2";
    case SelectionMethod::TournWITHOUT_t4:
      return "tournament_without_t4";
    case SelectionMethod::TournWITHPART_t2:
      return "tournament_part_t2";

    // Ranking selection methods
    case SelectionMethod::ExpRankRWS_c0_9801:
      return "exp_rank_rws_c0_9801";
    case SelectionMethod::ExpRankRWS_c0_9606:
      return "exp_rank_rws_c0_9606";
    case SelectionMethod::ExpRankSUS_c0_9801:
      return "exp_rank_sus_c0_9801";
    case SelectionMethod::ExpRankSUS_c0_9606:
      return "exp_rank_sus_c0_9606";
    case SelectionMethod::LinRankRWS_b2:
      return "lin_rank_rws_b2";
    case SelectionMethod::LinRankRWS_b1_6:
      return "lin_rank_rws_b1_6";
    case SelectionMethod::LinRankSUS_b2:
      return "lin_rank_sus_b2";
    case SelectionMethod::LinRankSUS_b1_6:
      return "lin_rank_sus_b1_6";

    default:
      return "unknown";
  }
}

std::string to_string(const CrossoverType type) {
  switch (type) {
    case CrossoverType::Single:
      return "single";
    case CrossoverType::SBX:
      return "sbx";
    case CrossoverType::Uniform:
      return "uniform";
    default:
      return "unknown";
  }
}

std::string to_string(const MutationType type) {
  switch (type) {
    case MutationType::Polynomial:
      return "polynomial";
    case MutationType::Density:
      return "density";
    default:
      return "unknown";
  }
}

std::string to_string(const EncodingMethod method) {
  switch (method) {
    case EncodingMethod::StandardBinary:
      return "standard_binary";
    case EncodingMethod::GrayCode:
      return "gray_code";
    case EncodingMethod::Discretization:
      return "discretization";
    default:
      return "unknown";
  }
}

std::string to_string(const ReproductionType type) {
  switch (type) {
    case ReproductionType::Generational:
      return "generational";
    case ReproductionType::SteadyState:
      return "steady_state";
    default:
      return "unknown";
  }
}

std::string to_string(const ParentSelectionMethod method) {
  switch (method) {
    case ParentSelectionMethod::Elite:
      return "elite";
    case ParentSelectionMethod::RWS:
      return "rws";
    default:
      return "unknown";
  }
}

std::string to_string(const ReplacementMethod method) {
  switch (method) {
    case ReplacementMethod::WorstComma:
      return "worst_comma";
    case ReplacementMethod::RandComma:
      return "rand_comma";
    case ReplacementMethod::WorstPlus:
      return "worst_plus";
    case ReplacementMethod::RandPlus:
      return "rand_plus";
    default:
      return "unknown";
  }
}

// Helper functions to convert strings to enums
CrossoverType crossover_from_string(const std::string& str) {
  if (str == "single") return CrossoverType::Single;
  if (str == "sbx") return CrossoverType::SBX;
  if (str == "uniform") return CrossoverType::Uniform;
  throw std::runtime_error("Unknown crossover type: " + str);
}

MutationType mutation_from_string(const std::string& str) {
  if (str == "polynomial") return MutationType::Polynomial;
  if (str == "density") return MutationType::Density;
  throw std::runtime_error("Unknown mutation type: " + str);
}

SelectionMethod selection_from_string(const std::string& str) {
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

EncodingMethod encoding_from_string(const std::string& str) {
  if (str == "standard_binary") return EncodingMethod::StandardBinary;
  if (str == "gray_code") return EncodingMethod::GrayCode;
  if (str == "discretization") return EncodingMethod::Discretization;
  throw std::runtime_error("Unknown encoding method: " + str);
}

ReproductionType reproduction_from_string(const std::string& str) {
  if (str == "generational") return ReproductionType::Generational;
  if (str == "steady_state") return ReproductionType::SteadyState;
  throw std::runtime_error("Unknown reproduction type: " + str);
}

ParentSelectionMethod parent_selection_from_string(const std::string& str) {
  if (str == "elite") return ParentSelectionMethod::Elite;
  if (str == "rws") return ParentSelectionMethod::RWS;
  throw std::runtime_error("Unknown parent selection method: " + str);
}

ReplacementMethod replacement_from_string(const std::string& str) {
  if (str == "worst_comma") return ReplacementMethod::WorstComma;
  if (str == "rand_comma") return ReplacementMethod::RandComma;
  if (str == "worst_plus") return ReplacementMethod::WorstPlus;
  if (str == "rand_plus") return ReplacementMethod::RandPlus;
  throw std::runtime_error("Unknown replacement method: " + str);
}

// For CSV header
std::string ga_config::csv_header() {
  return "Config_ID,Problem,Dimension,Population_Size,Islands,"
         "Generations_Per_Evolution,Total_Evolutions,"
         "Encoding_Method,Crossover_Type,Crossover_Prob,Mutation_Type,Mutation_Prob,"
         "Selection_Method,Reproduction_Type,Generation_Gap,Parent_Selection,"
         "Replacement";
}

// For CSV output
std::string ga_config::to_csv(int config_id) const {
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

// Helper function to calculate Euclidean distance
double euclidean_distance(const pagmo::vector_double& a,
                          const pagmo::vector_double& b) {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

// Function to calculate standard deviation
double calculate_std_dev(const std::vector<double>& values, const double avg) {
  if (values.empty() || values.size() == 1) return 0.0;

  double sum_squared_diff = 0.0;
  for (const double value : values) {
    const double diff = value - avg;
    sum_squared_diff += diff * diff;
  }

  return std::sqrt(sum_squared_diff / (values.size() - 1));
}
