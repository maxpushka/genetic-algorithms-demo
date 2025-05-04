#include "include/initialization.h"
#include "include/encoding_operator.h"
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
    
    // Create an empty population
    pagmo::population pop(prob, 0);
    
    // Calculate bits per dimension based on encoding method
    const unsigned bits_per_dimension = 
        (encoding_method == EncodingMethod::Discretization) ? 10 : 32;
    const unsigned total_bits = bits_per_dimension * dim;
    
    // Generate individuals
    for (pagmo::population::size_type i = 0; i < pop_size; ++i) {
        // Generate a binary string using binomial distribution
        std::string binary_string;
        binary_string.reserve(total_bits);
        
        for (unsigned j = 0; j < total_bits; ++j) {
            // Each bit has a 50% chance of being 0 or 1
            binary_string.push_back(binary_dist(gen) ? '1' : '0');
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
    }
    
    return pop;
}