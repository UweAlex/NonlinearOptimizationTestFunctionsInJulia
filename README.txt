NonlinearOptimizationTestFunctionsInJulia
Last modified: 11. Juli 2025, 14:20 PM CEST

Purpose
Provides test functions for nonlinear optimization in Julia, including Rosenbrock and Sphere, with analytical gradients and metadata for use with optimization packages like Optim.jl, NLopt, and others.

Installation
- Requires Julia 1.11.5+
- Dependencies: LinearAlgebra, Optim, Test, ForwardDiff, Zygote
- Optional: NLopt for specific demos
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()

Usage
- Load the module: using NonlinearOptimizationTestFunctionsInJulia
- Access test functions: ROSENBROCK_FUNCTION, SPHERE_FUNCTION
- Evaluate functions and gradients:
julia> rosenbrock([0.5, 0.5])
6.5
julia> sphere([1.0, 1.0])
2.0
- Use with optimization libraries:
julia> using Optim
julia> tf = ROSENBROCK_FUNCTION
julia> optimize(tf.f, tf.gradient!, tf.meta[:start](2), LBFGS(), Optim.Options(f_reltol=1e-6))

Test Functions
- Rosenbrock: Multimodal, non-convex, non-separable, differentiable, scalable. Minimum at [1.0, ..., 1.0] with value 0.0.
- Sphere: Unimodal, convex, separable, differentiable, scalable. Minimum at [0.0, ..., 0.0] with value 0.0.
- Access via TEST_FUNCTIONS dictionary:
julia> TEST_FUNCTIONS["Rosenbrock"]
ROSENBROCK_FUNCTION

Metadaten
The metadata :start, :min_position, :lb, and :ub are defined as functions accepting a dimension parameter n (default: 2). Example:
julia> ROSENBROCK_FUNCTION.meta[:lb](3)
3-element Vector{Float64}: [-5.0, -5.0, -5.0]
julia> SPHERE_FUNCTION.meta[:start](4)
4-element Vector{Float64}: [0.0, 0.0, 0.0, 0.0]
- Rosenbrock bounds: [-5.0, 5.0]^n
- Sphere bounds: [-5.12, 5.12]^n

Demos
Five example scripts in examples/ (10-15 lines each):
- Optimize_all_functions.jl: Optimizes all functions with L-BFGS.
- Compare_optimization_methods.jl: Compares Gradient Descent and L-BFGS on Rosenbrock.
- List_all_available_test_functions_and_their_properties.jl: Lists functions, start points, minima, properties.
- Optimize_with_nlopt.jl: Optimizes Rosenbrock with NLopt's LD_LBFGS (requires NLopt.jl).
- Compute_hessian_with_zygote.jl: Performs 3 Newton steps on Rosenbrock using Zygote's Hessian.

Tests
- 64 tests in test/runtests.jl:
  - Function values, gradients, numerical gradient accuracy, edge cases (NaN, Inf, 1e-308).
  - Properties (multimodal, convex, etc.).
  - Scalable metadata for n=2 and n=3.
  - Filtering and property manipulation.
- Run tests:
julia> include("test/runtests.jl")

Features
- Scalable: Functions and metadata support arbitrary dimensions via parameter n.
- Robust: Handles edge cases (NaN, Inf, 1e-308) with appropriate error handling.
- Compatible: Works with Optim.jl, NLopt, and automatic differentiation (ForwardDiff, Zygote).
- Modular: Test functions loaded via src/include_testfunctions.jl and TEST_FUNCTIONS in src/NonlinearOptimizationTestFunctionsInJulia.jl.

Contributing
- Add new test functions in src/functions/ and include them in src/include_testfunctions.jl.
- Ensure new functions provide f, grad, gradient!, and meta with required keys (:name, :start, :min_position, :min_value, :properties, :lb, :ub).
- Run tests to verify compatibility.

License
MIT License