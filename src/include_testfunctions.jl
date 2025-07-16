# src/include_testfunctions.jl
# Purpose: Includes all test function definitions.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, enables modular function loading.
# Last modified: 16. Juli 2025, 13:21 PM CEST

include("functions/rosenbrock.jl")
include("functions/sphere.jl")
include("functions/ackley.jl")
include("functions/axis_parallel_hyper_ellipsoid.jl")
include("functions/rastrigin.jl")
include("functions/griewank.jl")