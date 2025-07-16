NonlinearOptimizationTestFunctionsInJulia
Last modified: 16. Juli 2025, 12:25 PM CEST

Purpose
Provides test functions for nonlinear optimization in Julia, including Rosenbrock, Sphere, Ackley, AxisParallelHyperEllipsoid, and Rastrigin, with analytical gradients and metadata for use with optimization packages like Optim.jl, NLopt, and others.

### Ackley Function
The Ackley function uses default bounds `[-5, 5]` for compatibility with the test suite. For standard benchmarks, use `tf.meta[:lb](n, bounds="benchmark")` and `tf.meta[:ub](n, bounds="benchmark")` to set bounds to `[-32.768, 32.768]`.

Installation
- Requires Julia 1.11.5+
- Dependencies: LinearAlgebra, Optim, Test, ForwardDiff, Zygote
- Optional: NLopt for specific demos
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()

Usage
- Load the module:
    using NonlinearOptimizationTestFunctionsInJulia
- Access test functions:
    ROSENBROCK_FUNCTION, SPHERE_FUNCTION, ACKLEY_FUNCTION, AXISPARALLELHYPERELLIPSOID_FUNCTION, RASTRIGIN_FUNCTION
- Evaluate functions and gradients:
    rosenbrock([0.5, 0.5])  # Returns 6.5
    sphere([1.0, 1.0])      # Returns 2.0
    ackley([1.0, 1.0])      # Returns approximately 3.6253849384403627
    axisparallelhyperellipsoid([1.0, 1.0])  # Returns 3.0
    rastrigin([1.0, 1.0])   # Returns 2.0
- Optimize with libraries like Optim.jl:
    using Optim
    tf = ROSENBROCK_FUNCTION
    optimize(tf.f, tf.gradient!, tf.meta[:start](2), LBFGS(), Optim.Options(f_reltol=1e-6))

API Details
- tf.f: Objective function, takes a vector input and returns a scalar.
- tf.grad: Non-in-place gradient function, returns the gradient as a new vector, suitable for ForwardDiff and Zygote.
- tf.gradient!: In-place gradient function, modifies a provided gradient vector, optimized for Optim.jl and NLopt.
- tf.meta: Dictionary containing metadata (e.g., :name, :start, :min_position, :min_value, :properties, :lb, :ub).

Test Functions
- Rosenbrock: Multimodal, non-convex, non-separable, differentiable, scalable, bounded. Minimum at [1.0, ..., 1.0] with value 0.0.
- Sphere: Unimodal, convex, separable, differentiable, scalable, bounded. Minimum at [0.0, ..., 0.0] with value 0.0.
- Ackley: Multimodal, non-convex, non-separable, differentiable, scalable, bounded. Minimum at [0.0, ..., 0.0] with value 0.0.
- AxisParallelHyperEllipsoid: Unimodal, convex, separable, differentiable, scalable. Minimum at [0.0, ..., 0.0] with value 0.0.
- Rastrigin: Multimodal, non-convex, separable, differentiable, scalable, bounded. Minimum at [0.0, ..., 0.0] with value 0.0.
- Access via TEST_FUNCTIONS dictionary:
    TEST_FUNCTIONS["Rosenbrock"]  # Returns ROSENBROCK_FUNCTION

Metadata
The metadata :start, :min_position, :lb, and :ub are defined as functions accepting a dimension parameter n (default: 2). Example:
    ROSENBROCK_FUNCTION.meta[:lb](3)  # Returns [-5.0, -5.0, -5.0]
    SPHERE_FUNCTION.meta[:start](4)   # Returns [0.0, 0.0, 0.0, 0.0]
- Rosenbrock bounds: [-5.0, 5.0]^n
- Sphere bounds: [-5.12, 5.12]^n
- Ackley bounds: [-5.0, 5.0]^n (benchmark: [-32.768, 32.768]^n)
- AxisParallelHyperEllipsoid bounds: [-Inf, Inf]^n
- Rastrigin bounds: [-5.12, 5.12]^n

Demos
Five example scripts in examples/ (10-15 lines each):
- Optimize_all_functions.jl: Optimizes all functions with L-BFGS using tf.gradient!.
- Compare_optimization_methods.jl: Compares Gradient Descent and L-BFGS on Rosenbrock using tf.gradient!.
- List_all_available_test_functions_and_their_properties.jl: Lists functions, start points, minima, properties.
- Optimize_with_nlopt.jl: Optimizes Rosenbrock with NLopt's LD_LBFGS (requires NLopt.jl).
- Compute_hessian_with_zygote.jl: Performs 3 Newton steps on Rosenbrock using Zygote's Hessian.

Tests
- 143 tests in test/runtests.jl:
    - Rosenbrock (23): Function values, gradients, numerical accuracy, edge cases, properties.
    - Sphere (22): Function values, gradients, numerical accuracy, edge cases, properties.
    - Ackley (20): Function values, gradients, numerical accuracy, edge cases, properties.
    - AxisParallelHyperEllipsoid (16): Function values, gradients, numerical accuracy, edge cases, properties.
    - Rastrigin (23): Function values, gradients, numerical accuracy, edge cases, properties.
    - Filtering (4), properties (6), scalable metadata (25).
- Run tests:
    include("test/runtests.jl")

Features
- Scalable: Functions and metadata support arbitrary dimensions via parameter n.
- Robust: Handles edge cases (NaN, Inf, 1e-308) with appropriate error handling.
- Compatible: Works with Optim.jl, NLopt, and automatic differentiation (ForwardDiff, Zygote).
- Modular: Test functions loaded via src/include_testfunctions.jl and TEST_FUNCTIONS in src/NonlinearOptimizationTestFunctionsInJulia.jl.

Contributing
- Add new test functions in src/functions/ and include them in src/include_testfunctions.jl.
- Ensure new functions provide f, grad, gradient!, and meta with required keys (:name, :start, :min_position, :min_value, :properties, :lb, :ub).
- Run tests to verify compatibility.

Note
- Properties are stored in lowercase. Use lowercase when calling has_property, e.g., has_property(tf, "multimodal") instead of has_property(tf, "Multimodal").

License
MIT License