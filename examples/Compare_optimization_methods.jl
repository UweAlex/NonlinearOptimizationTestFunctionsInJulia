# Path: examples/Compare_optimization_methods.jl
# Purpose: Compares Gradient Descent and L-BFGS algorithms on the Rosenbrock function to demonstrate the flexibility of TestFunction objects.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, showing how tf.f, tf.gradient!, and tf.start work seamlessly with different Optim.jl algorithms.
# Notes: Simple output to focus on ease of use, as described in Readme.txt. Requires Optim.jl (included in Project.toml).

using NonlinearOptimizationTestFunctionsInJulia, Optim
tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
result_gd = optimize(tf.f, tf.gradient!, tf.start, GradientDescent())
result_lbfgs = optimize(tf.f, tf.gradient!, tf.start, LBFGS())
println("Gradient Descent on $(tf.name): $(Optim.minimizer(result_gd)), $(Optim.minimum(result_gd))")
println("L-BFGS on $(tf.name): $(Optim.minimizer(result_lbfgs)), $(Optim.minimum(result_lbfgs))")