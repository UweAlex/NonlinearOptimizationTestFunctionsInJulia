# Path: examples/Compare_optimization_methods.jl
# Purpose: Compares Gradient Descent and L-BFGS on the Rosenbrock function.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, showing flexibility of TestFunction.
# Notes: Simple output, requires Optim.jl (included in Project.toml).
# Last modified: 11. Juli 2025, 09:13 AM CEST
using NonlinearOptimizationTestFunctionsInJulia, Optim
tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
result_gd = optimize(tf.f, tf.gradient!, tf.meta[:start], GradientDescent())
result_lbfgs = optimize(tf.f, tf.gradient!, tf.meta[:start], LBFGS())
println("Gradient Descent on $(tf.meta[:name]): $(Optim.minimizer(result_gd)), $(Optim.minimum(result_gd))")
println("L-BFGS on $(tf.meta[:name]): $(Optim.minimizer(result_lbfgs)), $(Optim.minimum(result_lbfgs))")