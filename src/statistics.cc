#include "include/statistics.h"

#include <algorithm>
#include <numeric>

// For CSV output
[[nodiscard]] std::string run_stats::to_csv() const {
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
  ss << distance_accuracy << ",";
  ss << "\"" << termination_reason << "\"";  // Quote termination reason to handle commas

  return ss.str();
}

// For CSV header
std::string aggregate_stats::csv_header() {
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
[[nodiscard]] std::string aggregate_stats::to_csv() const {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(6);

  ss << success_rate << ",";

  ss << min_iterations << "," << max_iterations << "," << avg_iterations << ","
     << std_iterations << ",";
  ss << min_evals << "," << max_evals << "," << avg_evals << "," << std_evals
     << ",";
  ss << min_exec_time << "," << max_exec_time << "," << avg_exec_time << ","
     << std_exec_time << ",";
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

// Function to calculate aggregate statistics
aggregate_stats calculate_aggregate_stats(const std::vector<run_stats>& runs) {
  aggregate_stats agg;

  // Count successful runs
  std::vector<run_stats> successful_runs;
  std::vector<run_stats> failed_runs;
  successful_runs.reserve(runs.size());
  failed_runs.reserve(runs.size());

  for (const auto& run : runs) {
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

    for (const auto& run : successful_runs) {
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

    for (const auto& run : successful_runs) {
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

    for (const auto& run : successful_runs) {
      agg.min_exec_time = std::min(agg.min_exec_time, run.execution_time_ms);
      agg.max_exec_time = std::max(agg.max_exec_time, run.execution_time_ms);
      exec_time_vec.emplace_back(static_cast<double>(run.execution_time_ms));
    }
    agg.avg_exec_time =
        std::accumulate(exec_time_vec.begin(), exec_time_vec.end(), 0.0) /
        exec_time_vec.size();
    agg.std_exec_time = calculate_std_dev(exec_time_vec, agg.avg_exec_time);

    // Calculate min, max, avg for f_max
    std::vector<double> f_max_vec;
    f_max_vec.reserve(successful_runs.size());
    agg.min_f_max = successful_runs[0].f_max;
    agg.max_f_max = successful_runs[0].f_max;

    for (const auto& run : successful_runs) {
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

    for (const auto& run : successful_runs) {
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

    for (const auto& run : successful_runs) {
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

    for (const auto& run : successful_runs) {
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

    for (const auto& run : successful_runs) {
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

    for (const auto& run : failed_runs) {
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