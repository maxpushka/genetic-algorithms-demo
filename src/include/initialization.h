#pragma once

#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <random>
#include <string>
#include <vector>

#include "encoding.h"
#include "encoding_operator.h"

// Custom initialization function that implements binomial distribution with
// p=0.5 as required by TASK.md
pagmo::population initialize_population_binomial(
    const pagmo::problem& prob, pagmo::population::size_type pop_size,
    EncodingMethod encoding_method, unsigned seed);