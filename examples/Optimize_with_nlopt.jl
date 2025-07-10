# Path: examples/Optimize_with_nlopt.jl
# Purpose: Demonstrates optimization of the Rosenbrock function using NLopt's LD_LBFGS algorithm to show compatibility with external optimization libraries.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, highlighting how TestFunction objects (tf.f, tf.gradient!, tf.start) integrate naturally with NLopt.
# Notes: Minimal output to emphasize simplicity, as described in Readme.txt. Requires NLopt.jl (optional dependency in Project.toml, install via Pkg.add("NLopt")).

using NonlinearOptimizationTestFunctionsInJulia, NLopt
tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
opt = Opt(:LD_LBFGS, length(tf.start))
NLopt.min_objective!(opt, (x, grad) -> begin
    f = tf.f(x)
    if length(grad) > 0
        tf.gradient!(grad, x)
    end
    f
end)
minf, minx, ret = optimize(opt, tf.start)
println("$(tf.name): $minx, $minf")