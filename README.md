# Implementation of Genetic Algorithms for Optimization

This project implements genetic algorithms for finding the maximum of two functions: Ackley's Path function and Deb's function, using the PaGMO library.

## Key Features:

1. **Uses PaGMO's built-in features**:
   - Built-in Ackley function implementation
   - Custom implementation of Deb's function
   - Genetic Algorithm implementation (SGA)
   - Archipelago for parallel execution

2. **Configurable Parameters**:
   - Population sizes: 100, 200, 300, 400
   - Problem dimensions: 1, 2, 3, 5
   - Crossover types: single-point, SBX
   - Crossover probabilities: 0.0, 0.6, 0.8, 1.0
   - Mutation probabilities: varies by dimension

3. **Performance Metrics**:
   - Success rate
   - Number of iterations
   - Number of fitness evaluations
   - Fitness of best solution
   - Population average fitness
   - Convergence metrics
   - Peak and distance accuracy

4. **Output**:
   - Detailed CSV with per-run statistics
   - Summary CSV with aggregate statistics

## Implementation Details

### Custom Deb Function

The Deb function is implemented as a custom User Defined Problem (UDP) following PaGMO's interface requirements. The function is:

```
f(x) = Σ e^(-2(ln 2)((x_i-0.1)/0.8)^2) sin^6(5πx_i)
```

With bounds 0 ≤ x_i < 1.023 and global maximum at x_i = 0.1.

### Archipelago for Parallelization

The implementation uses PaGMO's archipelago model for parallel execution, which significantly improves performance. Multiple islands evolve independently and occasionally share solutions.

### Configuration Space

All combinations of parameters as specified in the task are tested for both functions:

- Different encoding methods (through different crossover operators)
- Crossover types and probabilities
- Mutation probabilities appropriate for each dimension
- Different selection methods (though limited to tournament selection in PaGMO)
- Different population sizes

### Statistical Analysis

For each configuration, multiple runs are performed and statistics are collected on:
- Success rate (percentage of runs that converge to the global optimum)
- Performance metrics (iterations, fitness evaluations)
- Solution quality metrics (fitness values, distance to optimum)

## Usage

The program generates two CSV files:
1. `results/detailed_results.csv` - Contains detailed information for each run
2. `results/summary_results.csv` - Contains aggregate statistics for each configuration

To analyze results, these files can be imported into tools like Excel, Python pandas, or R for visualization and further analysis.