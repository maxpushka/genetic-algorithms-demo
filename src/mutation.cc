#include "include/mutation.h"

#include <random>
#include <string>

// Constructor
DensityBasedMutation::DensityBasedMutation(double mutation_prob, unsigned seed)
    : m_mutation_prob(mutation_prob), 
      m_gen(seed),
      m_dist(0.0, 1.0) {}

// Apply density-based mutation to a binary chromosome
std::string DensityBasedMutation::mutate(const std::string& chromosome) const {
    // Create a copy of the chromosome
    std::string result = chromosome;
    
    // Apply mutation to each bit with probability m_mutation_prob
    for (char& bit : result) {
        // Generate a random probability
        const double prob = m_dist(m_gen);
        
        // If random probability is less than mutation probability, flip the bit
        if (prob < m_mutation_prob) {
            bit = (bit == '0') ? '1' : '0';
        }
    }
    
    return result;
}

// Mutate a real-valued chromosome with the corresponding encoding
pagmo::vector_double DensityBasedMutation::mutate(
    const pagmo::vector_double& x,
    const EncodingOperator& encoder) const {
    
    // Encode the real-valued chromosome to binary
    std::string binary = encoder.encode(x);
    
    // Apply mutation to the binary string
    std::string mutated_binary = mutate(binary);
    
    // Decode the mutated binary string back to real values
    return encoder.decode(mutated_binary);
}