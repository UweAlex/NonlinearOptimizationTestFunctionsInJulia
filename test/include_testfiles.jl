# test/include_testfiles.jl
# Purpose: Includes all function-specific test files for NonlinearOptimizationTestFunctionsInJulia.
# Context: Part of the test suite, enables modular test loading for individual test functions.
# Last modified: 16. Juli 2025, 14:26 PM CEST

include("rosenbrock_tests.jl")
include("sphere_tests.jl")
include("ackley_tests.jl")
include("axisparallelhyperellipsoid_tests.jl")
include("rastrigin_tests.jl")
include("griewank_tests.jl")