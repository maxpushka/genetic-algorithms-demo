#pragma once

#include <vector>
#include <mutex>
#include <future>
#include "common.h"
#include "statistics.h"
#include "queue.h"

// Function to run a single GA experiment with given parameters
run_stats run_experiment(const ga_config& config, unsigned seed);

// Function to run experiments with thread-local storage
std::vector<run_stats> run_experiment_batch(
    const int config_id, const ga_config& config,
    ThreadSafeQueue<DetailedResultEntry>& detailed_queue,
    std::mutex& cout_mutex, const unsigned num_runs);

// Legacy function maintained for compatibility
void run_config(const int config_id, const ga_config& config,
                std::ofstream& detailed_csv, std::ofstream& summary_csv,
                const unsigned num_runs);