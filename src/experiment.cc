#include "include/experiment.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>

#include "include/common.h"
#include "include/encoding_operator.h"
#include "include/initialization.h"
#include "include/mutation.h"
#include "include/operators.h"
#include "include/problems.h"
#include "include/queue.h"
#include "include/statistics.h"
#include "include/termination.h"
#include "pagmo/algorithm.hpp"
#include "pagmo/algorithms/sga.hpp"
#include "pagmo/archipelago.hpp"
#include "pagmo/io.hpp"
#include "pagmo/problem.hpp"
#include "pagmo/problems/ackley.hpp"
#include "pagmo/types.hpp"

// Function to run a single GA experiment with given parameters
run_stats run_experiment(const ga_config& config, unsigned seed) {
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

  // Map our selection methods to PaGMO's supported types (roulette, tournament,
  // truncated)
  std::string pagmo_selection;
  int selection_param = 2;  // Default tournament size

  // PaGMO only supports "roulette", "tournament", and "truncated" selection
  // types
  switch (config.selection_method) {
    case SelectionMethod::RWS:
    case SelectionMethod::ExpRankRWS_c0_9801:
    case SelectionMethod::ExpRankRWS_c0_9606:
    case SelectionMethod::LinRankRWS_b2:
    case SelectionMethod::LinRankRWS_b1_6:
      pagmo_selection = "roulette";
      break;

    case SelectionMethod::Tournament:
    case SelectionMethod::TournWITH_t2:
    case SelectionMethod::TournWITHOUT_t2:
    case SelectionMethod::TournWITHPART_t2:
      pagmo_selection = "tournament";
      selection_param = 2;
      break;

    case SelectionMethod::TournWITH_t4:
    case SelectionMethod::TournWITHOUT_t4:
      pagmo_selection = "tournament";
      selection_param = 4;
      break;

    // All other selection methods, including SUS-based ones
    default:
      pagmo_selection = "tournament";  // Default to tournament
      selection_param = 2;
      break;
  }

  // Map our crossover and mutation types to PaGMO's supported types
  std::string pagmo_crossover;
  std::string pagmo_mutation;

  // PaGMO supports "single", "sbx", "uniform" for crossover
  switch (config.crossover_type) {
    case CrossoverType::Single:
      pagmo_crossover = "single";
      break;
    case CrossoverType::SBX:
      pagmo_crossover = "sbx";
      break;
    case CrossoverType::Uniform:
      pagmo_crossover = "uniform";
      break;
    default:
      pagmo_crossover = "single";  // Default
      break;
  }

  // PaGMO supports "polynomial", "gaussian" for mutation
  // Map our "Density" to "polynomial" for compatibility
  switch (config.mutation_type) {
    case MutationType::Polynomial:
      pagmo_mutation = "polynomial";
      break;
    case MutationType::Density:
      pagmo_mutation =
          "polynomial";  // Use polynomial as a substitute for density-based
      break;
    default:
      pagmo_mutation = "polynomial";  // Default
      break;
  }

  // Create our encoding operator based on configuration
  EncodingOperator encoding_op(config.encoding_method, prob);
  
  // Adjust mutation probability based on encoding method if necessary
  double adjusted_mutation_prob = config.mutation_prob;
  if (config.encoding_method == EncodingMethod::Discretization) {
    // For discretization encoding, we need a different mutation probability scaling
    // because the chromosome length is different (10 bits per dimension vs 32)
    adjusted_mutation_prob = config.mutation_prob * 3.2; // 32/10 scaling factor
  }
  
  // Create custom mutations for density-based mutation
  DensityBasedMutation density_mutation(adjusted_mutation_prob, seed);
  
  // Set up PaGMO's algorithm but we'll use our own mutation if needed
  pagmo::algorithm algo{pagmo::sga(
      config.generations_per_evolution,  // Generations per evolution
      config.crossover_prob,             // Crossover probability
      1.0,                               // eta_c (distribution index for SBX)
      adjusted_mutation_prob,            // Mutation probability (adjusted for encoding)
      1.0,                               // param_m (mutation parameter)
      selection_param,  // param_s (selection parameter - tournament size)
      pagmo_crossover,  // Use PaGMO-compatible crossover type
      pagmo_mutation,   // Use PaGMO-compatible mutation type
      pagmo_selection,  // Use PaGMO-compatible selection type
      seed              // Random seed
      )};

  // Initialize the population using binomial distribution
  pagmo::population pop = initialize_population_binomial(
      prob, config.population_size, config.encoding_method, seed);
      
  // Create archipelago with our custom initialized population
  pagmo::archipelago archi{config.island_count, algo, pop};
  
  // Create termination checker
  TerminationChecker termination(config);
  
  // Run the evolutions with proper termination conditions
  auto start_time = std::chrono::high_resolution_clock::now();
  
  bool terminated = false;
  for (unsigned i = 0; i < config.total_evolutions && !terminated; ++i) {
    // Evolve one step
    archi.evolve();
    archi.wait_check();
    
    // Check termination conditions for each island
    for (const auto& isl : archi) {
      if (termination.check_termination(isl.get_population())) {
        terminated = true;
        stats.termination_reason = termination.get_termination_reason();
        break;
      }
    }
  }
  
  // If using density-based mutation, we need to apply it manually
  if (config.mutation_type == MutationType::Density) {
    // For each island in the archipelago
    for (auto& isl : archi) {
      // We need to use a temporary population that we can modify
      pagmo::population temp_pop = isl.get_population();
      
      // Apply our custom density-based mutation to each individual
      for (pagmo::population::size_type i = 0; i < temp_pop.size(); ++i) {
        auto x = temp_pop.get_x()[i];
        auto mutated_x = density_mutation.mutate(x, encoding_op);
        
        // Replace with mutated individual if fitness improves
        auto f = temp_pop.get_f()[i][0];
        auto mutated_f = prob.fitness(mutated_x)[0];
        
        if ((config.problem_type == ProblemType::Ackley && mutated_f < f) || 
            (config.problem_type != ProblemType::Ackley && mutated_f > f)) {
          temp_pop.set_xf(i, mutated_x, prob.fitness(mutated_x));
        }
      }
      
      // Set the updated population back to the island
      isl.set_population(temp_pop);
    }
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  stats.execution_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time)
          .count();
          
  // Store actual iterations performed
  stats.iterations = termination.get_iterations();

  // Find the best solution across all islands
  pagmo::vector_double best_x;
  double best_f = std::numeric_limits<double>::lowest();
  if (config.problem_type == ProblemType::Ackley) {
    best_f = std::numeric_limits<double>::max();
  }

  double total_fitness = 0.0;
  int total_individuals = 0;

  // Get results from all islands and find the best
  for (const auto& isl : archi) {
    const auto& pop = isl.get_population();

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
  for (const auto& isl : archi) {
    const auto& pop = isl.get_population();
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

// Function to run experiments with thread-local storage
std::vector<run_stats> run_experiment_batch(
    const int config_id, const ga_config& config,
    ThreadSafeQueue<DetailedResultEntry>& detailed_queue,
    std::mutex& cout_mutex, const unsigned num_runs) {
  {
    std::lock_guard lock(cout_mutex);
    std::cout << "Starting config " << config_id << ": "
              << to_string(config.problem_type) << ", dim=" << config.dimension
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
    detailed_queue.emplace(config_id, run + 1,
                           stats);  // Using emplace for in-place construction

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
void run_config(const int config_id, const ga_config& config,
                std::ofstream& detailed_csv, std::ofstream& summary_csv,
                const unsigned num_runs) {
  std::cout << "Running config " << config_id << ": "
            << to_string(config.problem_type) << ", dim=" << config.dimension
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