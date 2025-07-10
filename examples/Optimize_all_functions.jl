# Path: examples/Optimize_all_functions.jl
# Purpose: Demonstrates straightforward optimization of all test functions (Rosenbrock, Sphere) using Optim.jl's L-BFGS algorithm.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, showcasing how TestFunction objects (tf.f, tf.gradient!, tf.start) integrate naturally with Optim.jl for simple and efficient optimization tasks.
# Notes: Minimal output (minimizer and minimum value) to emphasize ease of use, as described in Readme.txt. Requires Optim.jl (included in Project.toml).

using NonlinearOptimizationTestFunctionsInJulia, Optim
for tf in values(NonlinearOptimizationTestFunctionsInJulia.TEST_FUNCTIONS)
    result = optimize(tf.f, tf.gradient!, tf.start, LBFGS())
    println("$(tf.name): $(Optim.minimizer(result)), $(Optim.minimum(result))")
end