NonlinearOptimizationTestFunctionsInJulia

A Julia package for nonlinear optimization test functions with analytical gradients (non-in-place and in-place) and systematic classification (e.g., convexity, multimodality).

Installation

To install, clone the repository and activate the project environment:
using Pkg
Pkg.activate(".")
Pkg.add("LinearAlgebra")
Pkg.add("Test")
Pkg.add("Optim")
Pkg.add("Zygote")  # Optional for Newton demo
Pkg.add("NLopt")   # Optional for NLopt demo
# After registration in the Julia General Registry:
# Pkg.add("NonlinearOptimizationTestFunctionsInJulia")

Requires Julia 1.11.5 or higher.

Available Test Functions

Rosenbrock Function
- Definition: f(x) = Σ_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
- Gradient: Analytically implemented (non-in-place via tf.grad, in-place via tf.gradient!)
- Properties: multimodal, non-convex, non-separable, differentiable, scalable
- Starting Point: [0.0, 0.0, ...]
- Minimum Position: [1.0, 1.0, ...]
- Minimum Value: 0.0

Sphere Function
- Definition: f(x) = Σ_{i=1}^n x_i^2
- Gradient: Analytically implemented (non-in-place via tf.grad, in-place via tf.gradient!)
- Properties: unimodal, convex, separable, differentiable, scalable
- Starting Point: [1.0, 1.0, ...]
- Minimum Position: [0.0, 0.0, ...]
- Minimum Value: 0.0

Usage

Basic Evaluation:
using NonlinearOptimizationTestFunctionsInJulia
julia> rosenbrock([0.5, 0.5])
6.5
julia> sphere([1.0, 1.0])
2.0

Optimization Example:
using Optim
tf = ROSENBROCK_FUNCTION
g = zeros(length(tf.start))
result = optimize(tf.f, (G, x) -> tf.gradient!(G, x), tf.start, LBFGS())
julia> result.minimum
0.0

Demo Scripts:
- examples/Optimize_all_functions.jl: Optimizes all functions with Optim.jl's L-BFGS using in-place gradients, showing minimizer and minimum.
  julia> include("examples/Optimize_all_functions.jl")
- examples/Compare_optimization_methods.jl: Compares Gradient Descent and L-BFGS on Rosenbrock using in-place gradients.
  julia> include("examples/Compare_optimization_methods.jl")
- examples/List_all_available_test_functions_and_their_properties.jl: Lists functions, start points, minima, and properties.
  julia> include("examples/List_all_available_test_functions_and_their_properties.jl")
- examples/Optimize_with_nlopt.jl: Optimizes Rosenbrock with NLopt's LD_LBFGS using in-place gradients (requires NLopt.jl).
  julia> include("examples/Optimize_with_nlopt.jl")
- examples/Compute_hessian_with_zygote.jl: Performs 3 Newton steps on Rosenbrock using non-in-place analytical gradients and Zygote's Hessian (requires Zygote.jl).
  julia> include("examples/Compute_hessian_with_zygote.jl")

Changes (as of July 10, 2025)
- Test Suite: 60 tests passed in test/runtests.jl, covering function values, gradients (non-in-place and in-place), edge cases (NaN, ±Inf).
- Performance: Vectorized implementation for scalability, analytical gradients (~10-100x faster than AD).
- Structure: Test functions in src/functions/, managed via include_testfunctions.jl.
- Documentation: Uses Readme.txt for simplicity (Markdown rendering issues with blank lines).
- Gradient Support: Both non-in-place (tf.grad) and in-place (tf.gradient!) gradients available, compatible with Optim.jl, NLopt, and Hessian-based methods.

Comparison with Other Packages
- CUTEst: Comprehensive but complex setup (SIF, Fortran/C). Our package is Julia-native and simpler.
- Optim.jl: Optimization algorithms, no test functions. Our package provides test functions with analytical gradients for Optim.jl.