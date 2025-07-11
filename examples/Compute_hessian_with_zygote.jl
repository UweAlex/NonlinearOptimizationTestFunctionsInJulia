# Path: examples/Compute_hessian_with_zygote.jl.disabled
# Purpose: Shows 3 Newton steps on Rosenbrock using analytical gradients and Zygote's Hessian.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, for experts.
# Notes: Temporarily disabled due to MethodError with ForwardDiff.Dual in Zygote.hessian.
#        To be fixed in a future commit by ensuring rosenbrock supports ForwardDiff.Dual types.
# Last modified: 11. Juli 2025, 10:05 AM CEST
# using NonlinearOptimizationTestFunctionsInJulia, Zygote, LinearAlgebra
# tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
# local x = tf.meta[:start]
# x = x - inv(Zygote.hessian(tf.f, x)) * tf.grad(x)  # Schritt 1
# x = x - inv(Zygote.hessian(tf.f, x)) * tf.grad(x)  # Schritt 2
# x = x - inv(Zygote.hessian(tf.f, x)) * tf.grad(x)  # Schritt 3
# println("Nach 3 Newton-Schritten für $(tf.meta[:name]): $x")