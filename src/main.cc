#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <bitset>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/io.hpp>

// Constants
constexpr int NUM_RUNS = 10; // Reduced for demonstration (should be 100 for full tests)
constexpr double DELTA = 0.01; // Fitness threshold for success
constexpr double SIGMA = 0.01; // Distance threshold for success

using namespace pagmo;

// Binary encoding utilities
std::vector<bool> to_binary(double value, double min_val, double max_val, int num_bits) {
    // Scale to [0, 2^num_bits - 1]
    double scaled = (value - min_val) / (max_val - min_val) * ((1 << num_bits) - 1);
    int int_val = static_cast<int>(scaled);
    
    std::vector<bool> result(num_bits);
    for (int i = 0; i < num_bits; ++i) {
        result[num_bits - 1 - i] = (int_val & (1 << i)) != 0;
    }
    return result;
}

double from_binary(const std::vector<bool>& binary, double min_val, double max_val) {
    int int_val = 0;
    for (size_t i = 0; i < binary.size(); ++i) {
        if (binary[i]) {
            int_val |= (1 << (binary.size() - 1 - i));
        }
    }
    
    // Scale back to [min_val, max_val]
    double scaled = min_val + (int_val * (max_val - min_val)) / ((1 << binary.size()) - 1);
    return scaled;
}

// Gray code utilities
std::vector<bool> to_gray(const std::vector<bool>& binary) {
    std::vector<bool> gray(binary.size());
    gray[0] = binary[0];
    for (size_t i = 1; i < binary.size(); ++i) {
        gray[i] = binary[i-1] ^ binary[i];
    }
    return gray;
}

std::vector<bool> from_gray(const std::vector<bool>& gray) {
    std::vector<bool> binary(gray.size());
    binary[0] = gray[0];
    for (size_t i = 1; i < gray.size(); ++i) {
        binary[i] = binary[i-1] ^ gray[i];
    }
    return binary;
}

// Ackley function (to be maximized, so we negate the standard minimization form)
struct ackley_func {
    ackley_func(unsigned dim = 1) : m_dim(dim) {}
    
    vector_double fitness(const vector_double &x) const {
        double sum1 = 0.0;
        double sum2 = 0.0;
        
        for (decltype(m_dim) i = 0u; i < m_dim; ++i) {
            sum1 += x[i] * x[i];
            sum2 += std::cos(2.0 * M_PI * x[i]);
        }
        
        sum1 = -0.2 * std::sqrt(sum1 / m_dim);
        sum2 = sum2 / m_dim;
        
        // Original Ackley is to be minimized, we maximize (21 - ackley)
        double result = 20.0 * std::exp(sum1) + std::exp(sum2);
        return {result};
    }
    
    std::pair<vector_double, vector_double> get_bounds() const {
        vector_double lb(m_dim, -5.12);
        vector_double ub(m_dim, 5.12);
        return {lb, ub};
    }
    
    vector_double get_optimal_params() const {
        return vector_double(m_dim, 0.0);
    }
    
    double get_optimal_fitness() const {
        return 21.0;
    }
    
    unsigned m_dim;
};

// Deb function (to be maximized)
struct deb_func {
    deb_func(unsigned dim = 1) : m_dim(dim) {}
    
    vector_double fitness(const vector_double &x) const {
        double result = 0.0;
        
        for (decltype(m_dim) i = 0u; i < m_dim; ++i) {
            double xi = x[i];
            double term1 = std::exp(-2.0 * std::log(2.0) * std::pow((xi - 0.1) / 0.8, 2.0));
            double term2 = std::pow(std::sin(5.0 * M_PI * xi), 6.0);
            result += term1 * term2;
        }
        
        return {result};
    }
    
    std::pair<vector_double, vector_double> get_bounds() const {
        vector_double lb(m_dim, 0.0);
        vector_double ub(m_dim, 1.023);
        return {lb, ub};
    }
    
    vector_double get_optimal_params() const {
        return vector_double(m_dim, 0.1);
    }
    
    double get_optimal_fitness() const {
        return m_dim;  // Each dimension contributes 1.0 at its optimum
    }
    
    unsigned m_dim;
};

// Binary-encoded Ackley function
struct binary_ackley_func {
    binary_ackley_func(unsigned dim = 1, unsigned bits_per_dim = 10, bool use_gray = false)
        : m_dim(dim), m_bits_per_dim(bits_per_dim), m_use_gray(use_gray), m_fitness_count(0) {}
    
    vector_double fitness(const vector_double &x) const {
        // Track number of fitness evaluations
        ++m_fitness_count;
        
        // Decode binary representation
        vector_double decoded = decode(x);
        
        // Use the base Ackley function for evaluation
        ackley_func base_func(m_dim);
        return base_func.fitness(decoded);
    }
    
    vector_double decode(const vector_double &x) const {
        vector_double result(m_dim);
        auto bounds = get_base_bounds();
        
        for (unsigned i = 0; i < m_dim; ++i) {
            // Extract the bits for this dimension
            std::vector<bool> bits(m_bits_per_dim);
            for (unsigned j = 0; j < m_bits_per_dim; ++j) {
                bits[j] = x[i * m_bits_per_dim + j] > 0.5;
            }
            
            // Convert from Gray code if needed
            if (m_use_gray) {
                bits = from_gray(bits);
            }
            
            // Convert to real value
            result[i] = from_binary(bits, bounds.first[i], bounds.second[i]);
        }
        
        return result;
    }
    
    std::pair<vector_double, vector_double> get_bounds() const {
        // Binary encoding: 0 or 1 for each bit
        vector_double lb(m_dim * m_bits_per_dim, 0.0);
        vector_double ub(m_dim * m_bits_per_dim, 1.0);
        return {lb, ub};
    }
    
    std::pair<vector_double, vector_double> get_base_bounds() const {
        ackley_func base_func(m_dim);
        return base_func.get_bounds();
    }
    
    vector_double get_optimal_params() const {
        // Get optimal params in real space
        ackley_func base_func(m_dim);
        auto opt_real = base_func.get_optimal_params();
        auto bounds = base_func.get_bounds();
        
        // Encode to binary space
        vector_double result(m_dim * m_bits_per_dim);
        for (unsigned i = 0; i < m_dim; ++i) {
            std::vector<bool> bits = to_binary(opt_real[i], bounds.first[i], bounds.second[i], m_bits_per_dim);
            
            // Convert to Gray code if needed
            if (m_use_gray) {
                bits = to_gray(bits);
            }
            
            // Store in result vector (as real values 0.0 or 1.0)
            for (unsigned j = 0; j < m_bits_per_dim; ++j) {
                result[i * m_bits_per_dim + j] = bits[j] ? 1.0 : 0.0;
            }
        }
        
        return result;
    }
    
    double get_optimal_fitness() const {
        ackley_func base_func(m_dim);
        return base_func.get_optimal_fitness();
    }
    
    // Getters for stats
    unsigned get_fitness_count() const {
        return m_fitness_count;
    }
    
    void reset_fitness_count() const {
        m_fitness_count = 0;
    }
    
    unsigned m_dim;
    unsigned m_bits_per_dim;
    bool m_use_gray;
    mutable unsigned m_fitness_count;
};

// Binary-encoded Deb function
struct binary_deb_func {
    binary_deb_func(unsigned dim = 1, unsigned bits_per_dim = 10, bool use_gray = false)
        : m_dim(dim), m_bits_per_dim(bits_per_dim), m_use_gray(use_gray), m_fitness_count(0) {}
    
    vector_double fitness(const vector_double &x) const {
        // Track number of fitness evaluations
        ++m_fitness_count;
        
        // Decode binary representation
        vector_double decoded = decode(x);
        
        // Use the base Deb function for evaluation
        deb_func base_func(m_dim);
        return base_func.fitness(decoded);
    }
    
    vector_double decode(const vector_double &x) const {
        vector_double result(m_dim);
        auto bounds = get_base_bounds();
        
        for (unsigned i = 0; i < m_dim; ++i) {
            // Extract the bits for this dimension
            std::vector<bool> bits(m_bits_per_dim);
            for (unsigned j = 0; j < m_bits_per_dim; ++j) {
                bits[j] = x[i * m_bits_per_dim + j] > 0.5;
            }
            
            // Convert from Gray code if needed
            if (m_use_gray) {
                bits = from_gray(bits);
            }
            
            // Convert to real value
            result[i] = from_binary(bits, bounds.first[i], bounds.second[i]);
        }
        
        return result;
    }
    
    std::pair<vector_double, vector_double> get_bounds() const {
        // Binary encoding: 0 or 1 for each bit
        vector_double lb(m_dim * m_bits_per_dim, 0.0);
        vector_double ub(m_dim * m_bits_per_dim, 1.0);
        return {lb, ub};
    }
    
    std::pair<vector_double, vector_double> get_base_bounds() const {
        deb_func base_func(m_dim);
        return base_func.get_bounds();
    }
    
    vector_double get_optimal_params() const {
        // Get optimal params in real space
        deb_func base_func(m_dim);
        auto opt_real = base_func.get_optimal_params();
        auto bounds = base_func.get_bounds();
        
        // Encode to binary space
        vector_double result(m_dim * m_bits_per_dim);
        for (unsigned i = 0; i < m_dim; ++i) {
            std::vector<bool> bits = to_binary(opt_real[i], bounds.first[i], bounds.second[i], m_bits_per_dim);
            
            // Convert to Gray code if needed
            if (m_use_gray) {
                bits = to_gray(bits);
            }
            
            // Store in result vector (as real values 0.0 or 1.0)
            for (unsigned j = 0; j < m_bits_per_dim; ++j) {
                result[i * m_bits_per_dim + j] = bits[j] ? 1.0 : 0.0;
            }
        }
        
        return result;
    }
    
    double get_optimal_fitness() const {
        deb_func base_func(m_dim);
        return base_func.get_optimal_fitness();
    }
    
    // Getters for stats
    unsigned get_fitness_count() const {
        return m_fitness_count;
    }
    
    void reset_fitness_count() const {
        m_fitness_count = 0;
    }
    
    unsigned m_dim;
    unsigned m_bits_per_dim;
    bool m_use_gray;
    mutable unsigned m_fitness_count;
};

// Statistics structure
struct run_stats {
    bool is_successful = false;
    unsigned iterations = 0;
    unsigned fitness_evals = 0;
    double f_max = 0.0;
    vector_double x_max;
    double f_avg = 0.0;
    double convergence = 0.0;
    double peak_accuracy = 0.0;
    double distance_accuracy = 0.0;
    
    // For CSV output
    std::string to_csv() const {
        std::stringstream ss;
        ss << (is_successful ? 1 : 0) << ",";
        ss << iterations << ",";
        ss << fitness_evals << ",";
        ss << f_max << ",";
        
        // x_max as a string
        for (auto x : x_max) {
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
    
    // Additional failed stats would go here...
    
    // For CSV header
    static std::string csv_header() {
        return "Success_Rate,"
               "Min_NI,Max_NI,Avg_NI,Sigma_NI,"
               "Min_NFE,Max_NFE,Avg_NFE,Sigma_NFE,"
               "Min_Fmax,Max_Fmax,Avg_Fmax,Sigma_Fmax,"
               "Min_Favg,Max_Favg,Avg_Favg,Sigma_Favg,"
               "Min_FC,Max_FC,Avg_FC,Sigma_FC,"
               "Min_PA,Max_PA,Avg_PA,Sigma_PA,"
               "Min_DA,Max_DA,Avg_DA,Sigma_DA,"
               "Min_NI_f,Max_NI_f,Avg_NI_f,Sigma_NI_f";
    }
    
    // For CSV output
    std::string to_csv() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(6);
        
        ss << success_rate << ",";
        
        ss << min_iterations << "," << max_iterations << "," << avg_iterations << "," << std_iterations << ",";
        ss << min_evals << "," << max_evals << "," << avg_evals << "," << std_evals << ",";
        ss << min_f_max << "," << max_f_max << "," << avg_f_max << "," << std_f_max << ",";
        ss << min_f_avg << "," << max_f_avg << "," << avg_f_avg << "," << std_f_avg << ",";
        ss << min_convergence << "," << max_convergence << "," << avg_convergence << "," << std_convergence << ",";
        ss << min_peak_accuracy << "," << max_peak_accuracy << "," << avg_peak_accuracy << "," << std_peak_accuracy << ",";
        ss << min_dist_accuracy << "," << max_dist_accuracy << "," << avg_dist_accuracy << "," << std_dist_accuracy << ",";
        
        ss << min_iterations_f << "," << max_iterations_f << "," << avg_iterations_f << "," << std_iterations_f;
        
        return ss.str();
    }
};

// Configuration parameters
struct ga_config {
    // Problem parameters
    std::string problem_name;
    unsigned dimension;
    bool use_gray_code;
    unsigned bits_per_dim;
    
    // GA parameters
    unsigned population_size;
    double crossover_prob;
    double mutation_prob;
    std::string selection_method;
    std::string crossover_type;
    
    // For generational GA
    bool generational = true;
    
    // For steady-state GA
    double generation_gap = 0.1;
    std::string parent_selection = "Elite";
    std::string survivor_selection = "WorstComma";
    
    // For CSV header
    static std::string csv_header() {
        return "Config_ID,Problem,Dimension,Encoding,Population_Size,"
               "Crossover_Type,Crossover_Prob,Mutation_Prob,Selection_Method,"
               "Generational,Generation_Gap,Parent_Selection,Survivor_Selection";
    }
    
    // For CSV output
    std::string to_csv(int config_id) const {
        std::stringstream ss;
        ss << config_id << ",";
        ss << problem_name << ",";
        ss << dimension << ",";
        ss << (use_gray_code ? "Gray" : "Binary") << ",";
        ss << population_size << ",";
        ss << crossover_type << ",";
        ss << crossover_prob << ",";
        ss << mutation_prob << ",";
        ss << selection_method << ",";
        ss << (generational ? "Generational" : "Steady-state") << ",";
        ss << generation_gap << ",";
        ss << parent_selection << ",";
        ss << survivor_selection;
        
        return ss.str();
    }
};

// Function to calculate Euclidean distance
double euclidean_distance(const vector_double &a, const vector_double &b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Function to calculate standard deviation
double calculate_std_dev(const std::vector<double> &values, double avg) {
    if (values.empty()) return 0.0;
    
    double sum_squared_diff = 0.0;
    for (double value : values) {
        double diff = value - avg;
        sum_squared_diff += diff * diff;
    }
    
    return std::sqrt(sum_squared_diff / values.size());
}

// Function to run a single GA experiment with given parameters
run_stats run_experiment(const ga_config &config, unsigned seed) {
    run_stats stats;
    
    // Set up problem
    std::unique_ptr<problem> prob;
    if (config.problem_name == "Ackley") {
        prob = std::make_unique<problem>(binary_ackley_func(config.dimension, config.bits_per_dim, config.use_gray_code));
    } else if (config.problem_name == "Deb") {
        prob = std::make_unique<problem>(binary_deb_func(config.dimension, config.bits_per_dim, config.use_gray_code));
    } else {
        throw std::runtime_error("Unknown problem: " + config.problem_name);
    }
    
    // Track fitness evaluations - store a pointer to the UDP
    binary_ackley_func *ackley_udp = nullptr;
    binary_deb_func *deb_udp = nullptr;
    
    if (config.problem_name == "Ackley") {
        ackley_udp = prob->extract<binary_ackley_func>();
        ackley_udp->reset_fitness_count();
    } else {
        deb_udp = prob->extract<binary_deb_func>();
        deb_udp->reset_fitness_count();
    }
    
    // Set up algorithm
    std::unique_ptr<algorithm> algo;
    
    // Set up selection method
    unsigned tournament_size = 2;  // Default to tournament size 2
    if (config.selection_method == "TournWITH_t=2") {
        tournament_size = 2;
    } else if (config.selection_method == "TournWITH_t=4") {
        tournament_size = 4;
    }
    
    // Generational GA
    algo = std::make_unique<algorithm>(sga(
        1000000, // Max generations
        config.crossover_prob, 
        1.0, // eta_c - distribution index for SBX crossover
        config.mutation_prob,
        1.0, // eta_m - distribution index for polynomial mutation
        tournament_size,
        config.crossover_type, // SBX, exponential, single point
        "polynomial", // Mutation type
        config.selection_method, // Selection type
        seed // Random seed
    ));
    
    // Create population with the specified size
    population pop(*prob, config.population_size, seed);
    
    // Run the algorithm
    pop = algo->evolve(pop);
    
    // Extract results and compute statistics
    stats.iterations = algo->extract<sga>()->get_log().size();
    
    // Get fitness evaluation count
    if (config.problem_name == "Ackley") {
        stats.fitness_evals = ackley_udp->get_fitness_count();
    } else {
        stats.fitness_evals = deb_udp->get_fitness_count();
    }
    
    // Get the best solution
    auto champion_x_bin = pop.champion_x();
    auto champion_f = pop.champion_f()[0];
    
    // Decode binary solution to real space
    vector_double champion_x_real;
    if (config.problem_name == "Ackley") {
        champion_x_real = ackley_udp->decode(champion_x_bin);
    } else {
        champion_x_real = deb_udp->decode(champion_x_bin);
    }
    
    stats.f_max = champion_f;
    stats.x_max = champion_x_real;
    
    // Calculate population average fitness
    double total_fitness = 0.0;
    for (unsigned i = 0; i < pop.size(); ++i) {
        total_fitness += pop.get_f()[i][0];
    }
    stats.f_avg = total_fitness / pop.size();
    
    // Calculate convergence (population homogeneity)
    // Simplified: just measure average distance from champion
    double total_distance = 0.0;
    for (unsigned i = 0; i < pop.size(); ++i) {
        auto individual = pop.get_x()[i];
        vector_double individual_real;
        
        if (config.problem_name == "Ackley") {
            individual_real = ackley_udp->decode(individual);
        } else {
            individual_real = deb_udp->decode(individual);
        }
        
        total_distance += euclidean_distance(champion_x_real, individual_real);
    }
    stats.convergence = 1.0 - (total_distance / (pop.size() * config.dimension));
    
    // Get optimal solution for comparison
    vector_double optimal_x;
    double optimal_f;
    
    if (config.problem_name == "Ackley") {
        optimal_x = ackley_udp->get_base_bounds().first; // Initialize with correct size
        for (size_t i = 0; i < optimal_x.size(); ++i) {
            optimal_x[i] = 0.0; // Ackley optimum is at origin
        }
        optimal_f = ackley_udp->get_optimal_fitness();
    } else {
        optimal_x = deb_udp->get_base_bounds().first; // Initialize with correct size
        for (size_t i = 0; i < optimal_x.size(); ++i) {
            optimal_x[i] = 0.1; // Deb optimum is at 0.1
        }
        optimal_f = deb_udp->get_optimal_fitness();
    }
    
    // Calculate peak accuracy and distance accuracy
    stats.peak_accuracy = champion_f / optimal_f;
    stats.distance_accuracy = 1.0 / (1.0 + euclidean_distance(champion_x_real, optimal_x));
    
    // Determine if run was successful
    stats.is_successful = (std::abs(champion_f - optimal_f) <= DELTA * optimal_f) && 
                         (euclidean_distance(champion_x_real, optimal_x) <= SIGMA);
    
    return stats;
}

// Function to calculate aggregate statistics
aggregate_stats calculate_aggregate_stats(const std::vector<run_stats> &runs) {
    aggregate_stats agg;
    
    // Count successful runs
    std::vector<run_stats> successful_runs;
    std::vector<run_stats> failed_runs;
    
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
        agg.min_iterations = successful_runs[0].iterations;
        agg.max_iterations = successful_runs[0].iterations;
        
        for (const auto &run : successful_runs) {
            agg.min_iterations = std::min(agg.min_iterations, run.iterations);
            agg.max_iterations = std::max(agg.max_iterations, run.iterations);
            iterations_vec.push_back(run.iterations);
        }
        agg.avg_iterations = std::accumulate(iterations_vec.begin(), iterations_vec.end(), 0.0) / iterations_vec.size();
        agg.std_iterations = calculate_std_dev(iterations_vec, agg.avg_iterations);
        
        // Calculate min, max, avg for fitness evaluations
        std::vector<double> evals_vec;
        agg.min_evals = successful_runs[0].fitness_evals;
        agg.max_evals = successful_runs[0].fitness_evals;
        
        for (const auto &run : successful_runs) {
            agg.min_evals = std::min(agg.min_evals, run.fitness_evals);
            agg.max_evals = std::max(agg.max_evals, run.fitness_evals);
            evals_vec.push_back(run.fitness_evals);
        }
        agg.avg_evals = std::accumulate(evals_vec.begin(), evals_vec.end(), 0.0) / evals_vec.size();
        agg.std_evals = calculate_std_dev(evals_vec, agg.avg_evals);
        
        // Calculate min, max, avg for f_max
        std::vector<double> f_max_vec;
        agg.min_f_max = successful_runs[0].f_max;
        agg.max_f_max = successful_runs[0].f_max;
        
        for (const auto &run : successful_runs) {
            agg.min_f_max = std::min(agg.min_f_max, run.f_max);
            agg.max_f_max = std::max(agg.max_f_max, run.f_max);
            f_max_vec.push_back(run.f_max);
        }
        agg.avg_f_max = std::accumulate(f_max_vec.begin(), f_max_vec.end(), 0.0) / f_max_vec.size();
        agg.std_f_max = calculate_std_dev(f_max_vec, agg.avg_f_max);
        
        // Calculate min, max, avg for f_avg
        std::vector<double> f_avg_vec;
        agg.min_f_avg = successful_runs[0].f_avg;
        agg.max_f_avg = successful_runs[0].f_avg;
        
        for (const auto &run : successful_runs) {
            agg.min_f_avg = std::min(agg.min_f_avg, run.f_avg);
            agg.max_f_avg = std::max(agg.max_f_avg, run.f_avg);
            f_avg_vec.push_back(run.f_avg);
        }
        agg.avg_f_avg = std::accumulate(f_avg_vec.begin(), f_avg_vec.end(), 0.0) / f_avg_vec.size();
        agg.std_f_avg = calculate_std_dev(f_avg_vec, agg.avg_f_avg);
        
        // Calculate min, max, avg for convergence
        std::vector<double> conv_vec;
        agg.min_convergence = successful_runs[0].convergence;
        agg.max_convergence = successful_runs[0].convergence;
        
        for (const auto &run : successful_runs) {
            agg.min_convergence = std::min(agg.min_convergence, run.convergence);
            agg.max_convergence = std::max(agg.max_convergence, run.convergence);
            conv_vec.push_back(run.convergence);
        }
        agg.avg_convergence = std::accumulate(conv_vec.begin(), conv_vec.end(), 0.0) / conv_vec.size();
        agg.std_convergence = calculate_std_dev(conv_vec, agg.avg_convergence);
        
        // Calculate min, max, avg for peak accuracy
        std::vector<double> pa_vec;
        agg.min_peak_accuracy = successful_runs[0].peak_accuracy;
        agg.max_peak_accuracy = successful_runs[0].peak_accuracy;
        
        for (const auto &run : successful_runs) {
            agg.min_peak_accuracy = std::min(agg.min_peak_accuracy, run.peak_accuracy);
            agg.max_peak_accuracy = std::max(agg.max_peak_accuracy, run.peak_accuracy);
            pa_vec.push_back(run.peak_accuracy);
        }
        agg.avg_peak_accuracy = std::accumulate(pa_vec.begin(), pa_vec.end(), 0.0) / pa_vec.size();
        agg.std_peak_accuracy = calculate_std_dev(pa_vec, agg.avg_peak_accuracy);
        
        // Calculate min, max, avg for distance accuracy
        std::vector<double> da_vec;
        agg.min_dist_accuracy = successful_runs[0].distance_accuracy;
        agg.max_dist_accuracy = successful_runs[0].distance_accuracy;
        
        for (const auto &run : successful_runs) {
            agg.min_dist_accuracy = std::min(agg.min_dist_accuracy, run.distance_accuracy);
            agg.max_dist_accuracy = std::max(agg.max_dist_accuracy, run.distance_accuracy);
            da_vec.push_back(run.distance_accuracy);
        }
        agg.avg_dist_accuracy = std::accumulate(da_vec.begin(), da_vec.end(), 0.0) / da_vec.size();
        agg.std_dist_accuracy = calculate_std_dev(da_vec, agg.avg_dist_accuracy);
    }
    
    // Process failed runs
    if (!failed_runs.empty()) {
        // Calculate min, max, avg for iterations (failed runs)
        std::vector<double> iterations_f_vec;
        agg.min_iterations_f = failed_runs[0].iterations;
        agg.max_iterations_f = failed_runs[0].iterations;
        
        for (const auto &run : failed_runs) {
            agg.min_iterations_f = std::min(agg.min_iterations_f, run.iterations);
            agg.max_iterations_f = std::max(agg.max_iterations_f, run.iterations);
            iterations_f_vec.push_back(run.iterations);
        }
        agg.avg_iterations_f = std::accumulate(iterations_f_vec.begin(), iterations_f_vec.end(), 0.0) / iterations_f_vec.size();
        agg.std_iterations_f = calculate_std_dev(iterations_f_vec, agg.avg_iterations_f);
        
        // Additional failed run stats would be calculated here
    }
    
    return agg;
}

// Function to run a set of experiments with a given configuration
void run_config(int config_id, const ga_config &config, std::ofstream &detailed_csv, std::ofstream &summary_csv) {
    std::cout << "Running config " << config_id << ": " 
              << config.problem_name << ", dim=" << config.dimension 
              << ", pop=" << config.population_size << std::endl;
    
    std::vector<run_stats> runs;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        std::cout << "  Run " << run + 1 << "/" << NUM_RUNS << "..." << std::flush;
        
        // Use a different seed for each run
        unsigned seed = config_id * 1000 + run;
        auto stats = run_experiment(config, seed);
        
        // Write detailed results for this run
        detailed_csv << config_id << ","
                    << run + 1 << ","
                    << stats.to_csv() << std::endl;
        
        runs.push_back(stats);
        
        std::cout << (stats.is_successful ? " Success" : " Failure") << std::endl;
    }
    
    // Calculate aggregate statistics
    auto agg_stats = calculate_aggregate_stats(runs);
    
    // Write summary for this config
    summary_csv << config_id << ","
               << config.to_csv(config_id) << ","
               << agg_stats.to_csv() << std::endl;
}

int main() {
    std::cout << "Genetic Algorithm Optimization - Ackley and Deb Functions" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    // Create output directory
    std::system("mkdir -p results");
    
    // Setup CSV files for output
    std::ofstream detailed_csv("results/detailed_results.csv");
    std::ofstream summary_csv("results/summary_results.csv");
    
    // Write headers
    detailed_csv << "Config_ID,Run_ID,Is_Successful,Iterations,Fitness_Evals,F_max,X_max,F_avg,Convergence,Peak_Accuracy,Distance_Accuracy" << std::endl;
    summary_csv << ga_config::csv_header() << "," << aggregate_stats::csv_header() << std::endl;
    
    // Configuration options to test
    std::vector<ga_config> configs;
    int config_id = 0;
    
    // For demonstration, we're using a limited set of configurations
    // Reduced test configuration for the Ackley function
    {
        ga_config base_config;
        base_config.problem_name = "Ackley";
        base_config.dimension = 1;
        base_config.bits_per_dim = 10;
        base_config.population_size = 100;
        
        // Only two configurations for demonstration
        // 1. Binary encoding, single-point crossover
        base_config.use_gray_code = false;
        base_config.crossover_type = "single"; // "single" is the correct name in pagmo/sga
        base_config.crossover_prob = 0.8;
        base_config.mutation_prob = 0.01;
        base_config.selection_method = "tournament";
        base_config.generational = true;
        
        configs.push_back(base_config);
        config_id++;
        
        // 2. Gray code encoding, SBX crossover
        base_config.use_gray_code = true;
        base_config.crossover_type = "sbx";
        base_config.crossover_prob = 0.8;
        base_config.mutation_prob = 0.01;
        base_config.selection_method = "tournament";
        base_config.generational = true;
        
        configs.push_back(base_config);
        config_id++;
    }
    
    // Also test the Deb function
    {
        ga_config base_config;
        base_config.problem_name = "Deb";
        base_config.dimension = 1;
        base_config.bits_per_dim = 10;
        base_config.population_size = 100;
        
        // Binary encoding, single-point crossover
        base_config.use_gray_code = false;
        base_config.crossover_type = "single"; // "single" is the correct name in pagmo/sga
        base_config.crossover_prob = 0.8;
        base_config.mutation_prob = 0.01;
        base_config.selection_method = "tournament";
        base_config.generational = true;
        
        configs.push_back(base_config);
        config_id++;
    }
    
    // Run all configurations
    for (int i = 0; i < configs.size(); ++i) {
        run_config(i + 1, configs[i], detailed_csv, summary_csv);
    }
    
    std::cout << "All configurations completed. Results saved to 'results' directory." << std::endl;
    
    return 0;
}