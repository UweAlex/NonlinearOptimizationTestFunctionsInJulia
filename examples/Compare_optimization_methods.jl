# examples/Compare_optimization_methods.jl
using NonlinearOptimizationTestFunctionsInJulia
using Optim

# Demo: Compare two optimization methods (Gradient Descent and L-BFGS) on Rosenbrock
println("Comparing Optimization Methods on Rosenbrock Function:")
println("-"^50)

# Setup Rosenbrock function and gradient
tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
f = tf.f
g! = NonlinearOptimizationTestFunctionsInJulia.gradient!(tf)  # Use generic in-place gradient wrapper
x0 = tf.start  # Starting point [0.0, 0.0]

# Method 1: Gradient Descent
gd_options = Optim.Options(show_trace=false, iterations=10000)
gd_result = optimize(f, g!, x0, GradientDescent(), gd_options)
println("Gradient Descent:")
println("  Minimum: ", Optim.minimizer(gd_result))
println("  Function Value: ", Optim.minimum(gd_result))
println("  Iterations: ", Optim.iterations(gd_result))
println("  Converged: ", Optim.converged(gd_result))
println("-"^50)

# Method 2: L-BFGS
lbfgs_options = Optim.Options(show_trace=false, iterations=10000)
lbfgs_result = optimize(f, g!, x0, LBFGS(), lbfgs_options)
println("L-BFGS:")
println("  Minimum: ", Optim.minimizer(lbfgs_result))
println("  Function Value: ", Optim.minimum(lbfgs_result))
println("  Iterations: ", Optim.iterations(lbfgs_result))
println("  Converged: ", Optim.converged(lbfgs_result))
println("-"^50)