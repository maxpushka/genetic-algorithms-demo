#include "include/initialization.h"
#include "include/encoding_operator.h"
#include "include/logging.h"
#include <random>
#include <vector>
#include <string>

// Initialize population using binomial distribution with p=0.5
// This means each gene (bit) has a 50% chance of being 0 or 1
pagmo::population initialize_population_binomial(
    const pagmo::problem& prob, 
    pagmo::population::size_type pop_size,
    EncodingMethod encoding_method,
    unsigned seed) {
    
    LOG_INFO("Initializing population of size {} with binomial distribution (p=0.5), seed={}", 
            pop_size, seed);
    
    // Create random engine with the provided seed
    std::mt19937 gen(seed);
    
    // Create a binomial distribution with p=0.5 (equal probability for 0 and 1)
    std::bernoulli_distribution binary_dist(0.5);
    
    // Create an encoding operator for this problem
    EncodingOperator encoder(encoding_method, prob);
    
    // Get problem dimension
    const auto dim = prob.get_nx();
    
    // Get problem bounds
    const auto bounds = prob.get_bounds();
    
    LOG_DEBUG("Problem dimension: {}, bounds: [{}, {}]", 
              dim, bounds.first[0], bounds.second[0]);
    
    // Create an empty population
    pagmo::population pop(prob, 0);
    
    // Calculate bits per dimension based on encoding method
    const unsigned bits_per_dimension = 
        (encoding_method == EncodingMethod::Discretization) ? 10 : 32;
    const unsigned total_bits = bits_per_dimension * dim;
    
    LOG_DEBUG("Using {} bits per dimension, total bits: {}", 
              bits_per_dimension, total_bits);
    
    // Generate individuals
    for (pagmo::population::size_type i = 0; i < pop_size; ++i) {
        // Generate a binary string using binomial distribution
        std::string binary_string;
        binary_string.reserve(total_bits);
        
        for (unsigned j = 0; j < total_bits; ++j) {
            // Each bit has a 50% chance of being 0 or 1
            binary_string.push_back(binary_dist(gen) ? '1' : '0');
        }
        
        // Log a sample of the first individual's bit string
        if (i == 0) {
            LOG_DEBUG("Sample binary string for first individual: {}{}",
                     binary_string.substr(0, std::min(20u, total_bits)),
                     total_bits > 20 ? "..." : "");
        }
        
        // Decode the binary string to a chromosome
        pagmo::vector_double x;
        
        // Depending on encoding method, decode appropriately
        switch (encoding_method) {
            case EncodingMethod::StandardBinary: {
                x.resize(dim);
                for (unsigned j = 0; j < dim; ++j) {
                    const auto start = j * bits_per_dimension;
                    const auto part = binary_string.substr(start, bits_per_dimension);
                    x[j] = decode_binary_to_real(part, bounds.first[j], bounds.second[j]);
                }
                break;
            }
            case EncodingMethod::GrayCode: {
                x.resize(dim);
                for (unsigned j = 0; j < dim; ++j) {
                    const auto start = j * bits_per_dimension;
                    const auto part = binary_string.substr(start, bits_per_dimension);
                    const auto binary = gray_to_binary(part);
                    x[j] = decode_binary_to_real(binary, bounds.first[j], bounds.second[j]);
                }
                break;
            }
            case EncodingMethod::Discretization: {
                x.resize(dim);
                for (unsigned j = 0; j < dim; ++j) {
                    const auto start = j * bits_per_dimension;
                    const auto part = binary_string.substr(start, bits_per_dimension);
                    x[j] = decode_discretization_to_real(part, bounds.first[j], bounds.second[j], 1);
                }
                break;
            }
            default:
                throw std::runtime_error("Unknown encoding method");
        }
        
        // Add the individual to the population
        pop.push_back(x);
        
        // Log the real values for the first individual
        if (i == 0) {
            std::stringstream ss;
            ss << "First individual real values: [";
            for (size_t j = 0; j < std::min(x.size(), (size_t)5); ++j) {
                ss << x[j];
                if (j < std::min(x.size(), (size_t)5) - 1) {
                    ss << ", ";
                }
            }
            if (x.size() > 5) {
                ss << ", ...";
            }
            ss << "]";
            LOG_DEBUG("{}", ss.str());
        }
    }
    
    LOG_INFO("Population initialization complete: {} individuals", pop_size);
    return pop;
}