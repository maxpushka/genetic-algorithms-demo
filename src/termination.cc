#include "include/termination.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "include/logging.h"

// Constructor
TerminationChecker::TerminationChecker(const ga_config& config)
    : m_config(config), m_iterations(0), m_termination_reason("") {
    
    // Set maximum iterations based on reproduction type
    if (config.reproduction_type == ReproductionType::Generational) {
        m_max_iterations = 1000000; // 1 million as per TASK.md
    } else {
        // For steady state: GG * 1000000
        m_max_iterations = static_cast<unsigned>(config.generation_gap * 1000000);
    }
}

// Check if any termination condition is met
bool TerminationChecker::check_termination(const pagmo::population& pop) {
    // Increment iteration counter
    m_iterations++;
    
    // Create encoder for this problem
    EncodingOperator encoder(m_config.encoding_method, pop.get_problem());
    
    // Calculate average fitness of the population
    double avg_fitness = 0.0;
    for (pagmo::population::size_type i = 0; i < pop.size(); ++i) {
        double fitness;
        if (m_config.problem_type == ProblemType::Ackley) {
            fitness = pop.get_f()[i][0]; // Minimizing
        } else {
            fitness = -pop.get_f()[i][0]; // Negating for maximization
        }
        avg_fitness += fitness;
    }
    avg_fitness /= pop.size();
    
    // Add to fitness history
    m_avg_fitness_history.push_back(avg_fitness);
    
    // Keep only the last 10 (or adjusted for steady-state) values
    unsigned history_size = 10;
    if (m_config.reproduction_type == ReproductionType::SteadyState) {
        // For steady-state: 10 * N / GG as per TASK.md
        history_size = static_cast<unsigned>(10 * m_config.population_size / m_config.generation_gap);
    }
    
    while (m_avg_fitness_history.size() > history_size) {
        m_avg_fitness_history.pop_front();
    }
    
    // Check termination conditions
    
    // 1. Check population homogeneity
    if (check_homogeneity(pop, encoder)) {
        m_termination_reason = "Population homogeneity reached 99%";
        return true;
    }
    
    // 2. Check fitness stability
    if (m_avg_fitness_history.size() >= 10 && check_fitness_stability()) {
        m_termination_reason = "Average fitness stabilized";
        return true;
    }
    
    // 3. Check max iterations
    if (check_max_iterations()) {
        m_termination_reason = "Maximum iterations reached";
        return true;
    }
    
    // No termination condition met
    return false;
}

// Check if population has converged to homogeneity (99% of genes are the same)
bool TerminationChecker::check_homogeneity(const pagmo::population& pop, 
                                          const EncodingOperator& encoder) {
    if (pop.size() <= 1) {
        LOG_DEBUG("Population size <= 1, skipping homogeneity check");
        return false;
    }
    
    // Get binary representation of all individuals
    std::vector<std::string> binary_chromosomes;
    for (pagmo::population::size_type i = 0; i < pop.size(); ++i) {
        binary_chromosomes.push_back(encoder.encode(pop.get_x()[i]));
    }
    
    // All chromosomes should have the same length
    const auto chromosome_length = binary_chromosomes[0].size();
    LOG_DEBUG("Checking homogeneity for {} chromosomes of length {}", 
              binary_chromosomes.size(), chromosome_length);
    
    // Count how many chromosomes have the same value at each position
    unsigned non_homogeneous_genes = 0;
    
    for (size_t gene_pos = 0; gene_pos < chromosome_length; ++gene_pos) {
        // Count '0's and '1's at this position
        size_t zeros = 0;
        size_t ones = 0;
        
        for (const auto& chromosome : binary_chromosomes) {
            if (chromosome[gene_pos] == '0') {
                zeros++;
            } else {
                ones++;
            }
        }
        
        // Calculate the percentage of majority value
        const auto total = zeros + ones;
        const auto max_count = std::max(zeros, ones);
        const double homogeneity = static_cast<double>(max_count) / total;
        
        // If any gene position has less than 99% homogeneity, population is not homogeneous
        if (homogeneity < 0.99) {
            non_homogeneous_genes++;
            // Only log a sample of non-homogeneous genes to avoid excessive output
            if (non_homogeneous_genes <= 5) {
                LOG_DEBUG("Gene position {} not homogeneous: {:.2f}% ({}:{}) homogeneity", 
                          gene_pos, homogeneity * 100.0, zeros, ones);
            }
        }
    }
    
    if (non_homogeneous_genes > 0) {
        LOG_DEBUG("Population not homogeneous: {} of {} genes below 99% threshold", 
                 non_homogeneous_genes, chromosome_length);
        return false;
    }
    
    // All gene positions have at least 99% homogeneity
    LOG_INFO("Population reached homogeneity threshold (99%) after {} iterations", m_iterations);
    return true;
}

// Check if average fitness has not changed significantly for last 10 generations
bool TerminationChecker::check_fitness_stability() {
    // Need at least 10 generations to check stability
    if (m_avg_fitness_history.size() < 10) {
        LOG_DEBUG("Not enough fitness history ({} < 10) to check stability", 
                 m_avg_fitness_history.size());
        return false;
    }
    
    // Look at the last 10 values and check if change is less than 0.0001
    const double latest = m_avg_fitness_history.back();
    
    double max_diff = 0.0;
    double max_diff_value = 0.0;
    
    for (auto it = m_avg_fitness_history.begin(); it != m_avg_fitness_history.end() - 1; ++it) {
        double diff = std::abs(*it - latest);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_value = *it;
        }
        
        if (diff > 0.0001) {
            LOG_DEBUG("Fitness change too large: {} vs {} (diff: {})", 
                     *it, latest, diff);
            return false;
        }
    }
    
    LOG_INFO("Fitness stabilized after {} iterations. Max difference: {} (threshold: 0.0001)", 
             m_iterations, max_diff);
    return true;
}

// Check if maximum number of iterations has been reached
bool TerminationChecker::check_max_iterations() {
    bool reached = m_iterations >= m_max_iterations;
    if (reached) {
        LOG_WARN("Maximum iterations ({}) reached without convergence", m_max_iterations);
    }
    return reached;
}

// Get the reason for termination
std::string TerminationChecker::get_termination_reason() const {
    return m_termination_reason;
}

// Get number of iterations performed
unsigned TerminationChecker::get_iterations() const {
    return m_iterations;
}