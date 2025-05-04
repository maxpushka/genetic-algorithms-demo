#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "include/common.h"
#include "include/encoding.h"
#include "include/experiment.h"
#include "include/operators.h"
#include "include/problems.h"
#include "include/queue.h"
#include "include/statistics.h"
#include "pagmo/algorithm.hpp"
#include "pagmo/algorithms/sga.hpp"
#include "pagmo/archipelago.hpp"
#include "pagmo/io.hpp"
#include "pagmo/problem.hpp"
#include "pagmo/problems/ackley.hpp"
#include "pagmo/types.hpp"

// Function to generate all configurations for testing
std::vector<ga_config> create_configurations() {
  std::vector<ga_config> configs;
  int config_id = 0;
  
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
      EncodingMethod::GrayCode,
      EncodingMethod::Discretization
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
        const auto& mutation_probs = mutation_probs_by_dim[dim];
        
        for (const auto encoding_method : encoding_methods) {
          for (const auto& crossover_type : crossover_types) {
            for (double crossover_prob : crossover_probs) {
              for (const auto mutation_type : mutation_types) {
                for (double mutation_prob : mutation_probs) {
                  for (const auto reproduction_type : reproduction_types) {
                    if (reproduction_type == ReproductionType::Generational) {
                      // For generational reproduction type
                      for (const auto& selection_method : selection_methods) {
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
  
  return configs;
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
  detailed_csv << "Config_ID,Run_ID,Is_Successful,Iterations,Fitness_Evals,"
                  "Execution_Time_ms,F_"
                  "max,X_max,F_avg,Convergence,Peak_Accuracy,Distance_Accuracy"
               << std::endl;
  summary_csv << ga_config::csv_header() << "," << aggregate_stats::csv_header()
              << std::endl;

  // Number of runs as specified in the task
  constexpr unsigned NUM_RUNS = 100;  // as per TASK.md, line 158

  // Generate all configurations
  std::vector<ga_config> configs = create_configurations();

  // Store the original number of configurations before potentially reducing for
  // demo mode
  const size_t total_configs = configs.size();
  std::cout << "Total configurations generated: " << total_configs << std::endl;

  if constexpr (NUM_RUNS < 100) {
    // When running in demo mode
    std::vector<ga_config> demo_configs;

    std::cout << "Demo mode: Running a subset of configurations for testing."
              << std::endl;

    // Select a small representative sample of configurations
    if (total_configs <= 8) {
      // If we have 8 or fewer configs, use all of them
      demo_configs = configs;
    } else {
      // For larger config sets, select a representative sample
      size_t step = total_configs / 8;
      for (unsigned i = 0; i < std::min(static_cast<size_t>(8), total_configs);
           i++) {
        size_t idx = std::min(i * step, total_configs - 1);
        demo_configs.push_back(configs[idx]);
      }
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
  for (const auto& config : configs) {
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
    for (auto& thread : threads) {
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
        summary_queue.emplace(
            config_id, configs[start + i],
            agg_stats);  // Using emplace for in-place construction

        {
          std::lock_guard lock(cout_mutex);
          std::cout << "Processed results for config " << config_id
                    << std::endl;
        }
      } catch (const std::exception& e) {
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
