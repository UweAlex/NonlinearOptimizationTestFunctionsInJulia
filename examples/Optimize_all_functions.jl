# Path: examples/Optimize_all_functions.jl
# Purpose: Optimizes all test functions using Optim.jl's L-BFGS algorithm.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, showcasing TestFunction integration.
# Notes: Minimal output, requires Optim.jl (included in Project.toml).
# Last modified: 11. Juli 2025, 09:13 AM CEST
using NonlinearOptimizationTestFunctionsInJulia, Optim
for tf in values(NonlinearOptimizationTestFunctionsInJulia.TEST_FUNCTIONS)
    result = optimize(tf.f, tf.gradient!, tf.meta[:start], LBFGS())
    println("$(tf.meta[:name]): $(Optim.minimizer(result)), $(Optim.minimum(result))")
end