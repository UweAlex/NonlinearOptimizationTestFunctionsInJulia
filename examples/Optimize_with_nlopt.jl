# examples/Optimize_with_nlopt.jl
# Purpose: Demonstrates optimization of the Rosenbrock function using NLopt's LD_LBFGS algorithm.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, highlighting TestFunction integration with NLopt.
# Notes: Minimal output for simplicity. Requires NLopt.jl (optional dependency in Project.toml).
# Last modified: 11. Juli 2025, 10:23 AM CEST
using NonlinearOptimizationTestFunctionsInJulia

if isdefined(Main, :NLopt)
    using NLopt
    tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
    opt = Opt(:LD_LBFGS, length(tf.meta[:start]))
    NLopt.ftol_rel!(opt, 1e-6)  # Added tolerance
    NLopt.min_objective!(opt, (x, grad) -> begin
        f = tf.f(x)
        if length(grad) > 0
            tf.gradient!(grad, x)
        end
        f
    end)
    minf, minx, ret = optimize(opt, tf.meta[:start])
    println("$(tf.meta[:name]): $minx, $minf")
else
    println("NLopt.jl is not installed. Please install it to run this example.")
end