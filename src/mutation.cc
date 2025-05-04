#include "include/mutation.h"

#include <random>
#include <string>

#include "include/logging.h"

// Constructor
DensityBasedMutation::DensityBasedMutation(double mutation_prob, unsigned seed)
    : m_mutation_prob(mutation_prob), 
      m_gen(seed),
      m_dist(0.0, 1.0) {
    LOG_DEBUG("Created density-based mutation operator with probability: {}, seed: {}", 
              mutation_prob, seed);
}

// Apply density-based mutation to a binary chromosome
std::string DensityBasedMutation::mutate(const std::string& chromosome) const {
    // Create a copy of the chromosome
    std::string result = chromosome;
    
    // Count how many bits were actually mutated
    unsigned mutated_bits = 0;
    
    // Apply mutation to each bit with probability m_mutation_prob
    for (char& bit : result) {
        // Generate a random probability
        const double prob = m_dist(m_gen);
        
        // If random probability is less than mutation probability, flip the bit
        if (prob < m_mutation_prob) {
            bit = (bit == '0') ? '1' : '0';
            mutated_bits++;
        }
    }
    
    // Only log detailed mutation information at trace level
    LOG_TRACE("Density mutation: mutated {}/{} bits ({:.2f}%), probability: {}", 
             mutated_bits, chromosome.size(), 
             100.0 * mutated_bits / chromosome.size(), m_mutation_prob);
    
    return result;
}

// Mutate a real-valued chromosome with the corresponding encoding
pagmo::vector_double DensityBasedMutation::mutate(
    const pagmo::vector_double& x,
    const EncodingOperator& encoder) const {
    
    // Encode the real-valued chromosome to binary
    std::string binary = encoder.encode(x);
    
    LOG_TRACE("Mutation input: Binary length: {}, example: {}{}", 
             binary.size(), 
             binary.substr(0, std::min(20ul, binary.size())),
             binary.size() > 20 ? "..." : "");
    
    // Apply mutation to the binary string
    std::string mutated_binary = mutate(binary);
    
    // Count differences to original
    unsigned differences = 0;
    for (size_t i = 0; i < binary.size(); i++) {
        if (binary[i] != mutated_binary[i]) {
            differences++;
        }
    }
    
    LOG_TRACE("Mutation complete: {} bits changed out of {} ({:.2f}%)",
             differences, binary.size(), 
             100.0 * differences / binary.size());
    
    // Decode the mutated binary string back to real values
    return encoder.decode(mutated_binary);
}