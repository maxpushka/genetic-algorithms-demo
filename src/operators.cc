#include "include/operators.h"

#include <algorithm>
#include <numeric>
#include <random>

// Helper to determine if we're minimizing or maximizing
bool SelectionOperator::is_minimization(const pagmo::problem& prob) {
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

// Roulette Wheel Selection
std::vector<pagmo::population::size_type> RouletteWheelSelection::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
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
  const double total_fitness =
      std::accumulate(adjusted_fitness.begin(), adjusted_fitness.end(), 0.0);

  if (total_fitness <= 0.0) {
    // If total fitness is zero or negative, use uniform selection
    std::uniform_int_distribution<pagmo::population::size_type> dist(
        0, pop_size - 1);
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

// Stochastic Universal Sampling
std::vector<pagmo::population::size_type> StochasticUniversalSampling::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
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
  const double total_fitness =
      std::accumulate(adjusted_fitness.begin(), adjusted_fitness.end(), 0.0);

  if (total_fitness <= 0.0) {
    // If total fitness is zero or negative, use uniform selection
    std::uniform_int_distribution<pagmo::population::size_type> dist(
        0, pop_size - 1);
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

// Tournament selection with replacement
TournamentWithReplacement::TournamentWithReplacement(
    pagmo::population::size_type t_size)
    : tournament_size(t_size) {}

std::vector<pagmo::population::size_type> TournamentWithReplacement::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
  const auto pop_size = fitness.size();
  std::vector<pagmo::population::size_type> selected;
  selected.reserve(selection_size);

  // Distribution for selecting individuals for tournament
  std::uniform_int_distribution<pagmo::population::size_type> dist(
      0, pop_size - 1);

  // Run tournaments
  for (pagmo::population::size_type i = 0; i < selection_size; ++i) {
    pagmo::population::size_type best_idx = dist(rng);
    double best_fitness = fitness[best_idx];

    // Conduct tournament
    for (pagmo::population::size_type j = 1; j < tournament_size; ++j) {
      const auto idx = dist(rng);
      if (fitness[idx] <
          best_fitness) {  // Assuming minimization, adjust if maximizing
        best_idx = idx;
        best_fitness = fitness[idx];
      }
    }

    selected.push_back(best_idx);
  }

  return selected;
}

// Tournament selection without replacement
TournamentWithoutReplacement::TournamentWithoutReplacement(
    pagmo::population::size_type t_size)
    : tournament_size(t_size) {}

std::vector<pagmo::population::size_type> TournamentWithoutReplacement::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
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
    std::iota(candidates.begin(), candidates.end(),
              0);  // Fill with 0, 1, 2, ...
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

// Tournament with participation probability
TournamentWithParticipation::TournamentWithParticipation(
    pagmo::population::size_type t_size)
    : tournament_size(t_size) {}

std::vector<pagmo::population::size_type> TournamentWithParticipation::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
  const auto pop_size = fitness.size();
  std::vector<pagmo::population::size_type> selected;
  selected.reserve(selection_size);

  // Distribution for selecting individuals for tournament
  std::uniform_int_distribution<pagmo::population::size_type> dist(
      0, pop_size - 1);

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

// Exponential Ranking Selection (RWS-based)
ExponentialRankRWS::ExponentialRankRWS(double c_val) : c(c_val) {}

std::vector<pagmo::population::size_type> ExponentialRankRWS::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
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

// Exponential Ranking Selection (SUS-based)
ExponentialRankSUS::ExponentialRankSUS(double c_val) : c(c_val) {}

std::vector<pagmo::population::size_type> ExponentialRankSUS::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
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

// Linear Ranking Selection (RWS-based)
LinearRankRWS::LinearRankRWS(double b) : beta(b) {}

std::vector<pagmo::population::size_type> LinearRankRWS::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
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
    const double rank_value =
        2.0 - beta +
        2.0 * (beta - 1.0) * (static_cast<double>(i) / (pop_size - 1));
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

// Linear Ranking Selection (SUS-based)
LinearRankSUS::LinearRankSUS(double b) : beta(b) {}

std::vector<pagmo::population::size_type> LinearRankSUS::select(
    const std::vector<double>& fitness,
    const pagmo::population::size_type selection_size,
    pagmo::detail::random_engine_type& rng) const {
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
    const double rank_value =
        2.0 - beta +
        2.0 * (beta - 1.0) * (static_cast<double>(i) / (pop_size - 1));
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

// Factory function to create selection operator based on the selection method
std::unique_ptr<SelectionOperator> create_selection_operator(
    SelectionMethod method) {
  switch (method) {
    case SelectionMethod::SUS:
      return std::make_unique<StochasticUniversalSampling>();

    case SelectionMethod::RWS:
      return std::make_unique<RouletteWheelSelection>();

    case SelectionMethod::Tournament:
      return std::make_unique<TournamentWithReplacement>(
          2);  // Default tournament size 2

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