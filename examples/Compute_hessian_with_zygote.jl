# Path: examples/Compute_hessian_with_zygote.jl
# Purpose: Shows 3 Newton steps on Rosenbrock using analytical gradients and Zygote's Hessian.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, for experts.
# Notes: Minimal demo, not production-ready (no regularization/convergence checks).
using NonlinearOptimizationTestFunctionsInJulia, Zygote, LinearAlgebra
tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
x = tf.start
for i in 1:3
    x = x - inv(Zygote.hessian(tf.f, x)) * tf.grad(x)
end
println("Nach 3 Newton-Schritten für $(tf.name): $x")