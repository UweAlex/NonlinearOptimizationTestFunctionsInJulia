# NonlinearOptimizationTestFunctionsInJulia
# Last modified: 19 July 2025, 19:37 PM CEST

## Purpose
Provides test functions for nonlinear optimization in Julia, including Rosenbrock, Sphere, Ackley, AxisParallelHyperEllipsoid, Rastrigin, Griewank, Schwefel, Michalewicz, Branin, Goldstein-Price, Shubert, Six-Hump Camelback, Shekel, and Hartmann, with analytical gradients and metadata for use with optimization packages like Optim.jl, NLopt, ForwardDiff, and Zygote. All functions are based on *Test functions for optimization needs* (Molga & Smutnicki, 2005) and include the metadata property :in_molga_smutnicki_2005. The Langermann function is deferred for future implementation.

### Ackley Function
The Ackley function uses default bounds [-5, 5]^n for compatibility with the test suite. For standard benchmarks, use tf.meta[:lb](n, bounds="benchmark") and tf.meta[:ub](n, bounds="benchmark") to set bounds to [-32.768, 32.768]^n.

## Installation
- Requires Julia 1.11.5+
- Dependencies: LinearAlgebra, Optim, Test, ForwardDiff, Zygote
- Optional: NLopt for specific demos
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()

## Usage
- Load the module:
    using NonlinearOptimizationTestFunctionsInJulia
- Access test functions:
    ROSENBROCK_FUNCTION, SPHERE_FUNCTION, ACKLEY_FUNCTION, AXISPARALLELHYPERELLIPSOID_FUNCTION, RASTRIGIN_FUNCTION, GRIEWANK_FUNCTION, SCHWEFEL_FUNCTION, MICHALEWICZ_FUNCTION, BRANIN_FUNCTION, GOLDSTEINPRICE_FUNCTION, SHUBERT_FUNCTION, SIXHUMPCAMELBACK_FUNCTION, SHEKEL_FUNCTION, HARTMANN_FUNCTION
- Evaluate functions and gradients:
    rosenbrock([0.5, 0.5])                # Returns 6.5
    sphere([1.0, 1.0])                    # Returns 2.0
    ackley([1.0, 1.0])                    # Returns approximately 3.6253849384403627
    axisparallelhyperellipsoid([1.0, 1.0]) # Returns 3.0
    rastrigin([1.0, 1.0])                 # Returns 2.0
    griewank([1.0, 1.0])                  # Returns approximately 0.305
    schwefel([420.968746, 420.968746])   # Returns approximately 0.0
    michalewicz([2.20, 1.57])             # Returns approximately -1.8013
    branin([0.0, 0.0])                    # Returns approximately 55.602112642270262
    goldsteinprice([0.0, -1.0])           # Returns 3.0
    shubert([-7.083506, 4.858057])       # Returns approximately -186.7309
    sixhumpcamelback([-0.08984201368301331, 0.7126564032704135]) # Returns -1.031628453489877
    shekel([4.0, 4.0, 4.0, 4.0])         # Returns -10.536409825004505
    hartmann([0.114614, 0.555649, 0.852547]) # Returns -3.86278214782076
- Optimize with libraries like Optim.jl:
    using Optim
    tf = ROSENBROCK_FUNCTION
    optimize(tf.f, tf.gradient!, tf.meta[:start](2), LBFGS(), Optim.Options(f_reltol=1e-6))

## API Details
- tf.f: Objective function, takes a vector input and returns a scalar.
- tf.grad: Non-in-place gradient function, returns the gradient as a new vector, suitable for ForwardDiff and Zygote.
- tf.gradient!: In-place gradient function, automatically generated by the TestFunction constructor, optimized for Optim.jl and NLopt. Do not define explicitly.
- tf.meta: Dictionary containing metadata (e.g., :name, :start, :min_position, :min_value, :properties, :lb, :ub, :in_molga_smutnicki_2005).
- Hinweis: Alle Schlüssel im `TEST_FUNCTIONS`-Dictionary sind in Kleinbuchstaben (lowercase) gespeichert, z. B. `TEST_FUNCTIONS["rosenbrock"]`. Der Zugriff ist case-sensitiv, daher müssen Schlüssel exakt in Kleinbuchstaben angegeben werden.

## Test Functions
- Rosenbrock: Unimodal, non-convex, non-separable, differentiable, scalable, bounded. Minimum at [1.0, ..., 1.0] with value 0.0. Bounds: [-5.0, 5.0]^n.
- Sphere: Unimodal, convex, separable, differentiable, scalable, bounded. Minimum at [0.0, ..., 0.0] with value 0.0. Bounds: [-5.12, 5.12]^n.
- Ackley: Multimodal, non-convex, non-separable, differentiable, scalable, bounded. Minimum at [0.0, ..., 0.0] with value 0.0. Bounds: [-5.0, 5.0]^n (benchmark: [-32.768, 32.768]^n).
- AxisParallelHyperEllipsoid: Unimodal, convex, separable, differentiable, scalable. Minimum at [0.0, ..., 0.0] with value 0.0. Bounds: [-Inf, Inf]^n (restricted to [-100.0, 100.0]^n for gradient tests).
- Rastrigin: Multimodal, non-convex, separable, differentiable, scalable, bounded. Minimum at [0.0, ..., 0.0] with value 0.0. Bounds: [-5.12, 5.12]^n.
- Griewank: Multimodal, non-convex, non-separable, differentiable, scalable, bounded. Minimum at [0.0, ..., 0.0] with value 0.0. Bounds: [-600.0, 600.0]^n.
- Schwefel: Multimodal, non-convex, separable, differentiable, scalable, bounded. Minimum at [420.968746, ..., 420.968746] with value 0.0. Bounds: [-500.0, 500.0]^n.
- Michalewicz: Multimodal, non-convex, separable, differentiable, scalable, bounded. Minimum at approximately [2.20, 1.57, ...] with value approximately -1.8013 (for n=2). Bounds: [0.0, pi]^n.
- Branin: Multimodal, non-convex, non-separable, differentiable, bounded (n=2 only). Minima at [-π, 12.275], [π, 2.275], [9.424778, 2.475] with value approximately 0.397887. Bounds: x₁ ∈ [-5.0, 10.0], x₂ ∈ [0.0, 15.0].
- Goldstein-Price: Multimodal, non-convex, non-separable, differentiable, bounded (n=2 only). Minimum at [0.0, -1.0] with value 3.0. Bounds: [-2.0, 2.0]^2.
- Shubert: Multimodal, non-convex, non-separable, differentiable, bounded (n=2 only). Minimum at multiple points (e.g., [-7.083506, 4.858057]) with value approximately -186.7309. Bounds: [-10.0, 10.0]^2.
- Six-Hump Camelback: Multimodal, non-convex, non-separable, differentiable, bounded (n=2 only). Minima at [-0.08984201368301331, 0.7126564032704135], [0.08984201368301331, -0.7126564032704135] with value -1.031628453489877. Bounds: x₁ ∈ [-3.0, 3.0], x₂ ∈ [-2.0, 2.0].
- Shekel: Multimodal, non-convex, non-separable, differentiable, bounded (n=4 only). Minimum at [4.0, 4.0, 4.0, 4.0] with value -10.536409825004505. Bounds: x₁, x₂, x₃, x₄ ∈ [0.0, 10.0].
- Hartmann: Multimodal, non-convex, non-separable, differentiable, bounded (n=3 only). Minimum at [0.114614, 0.555649, 0.852547] with value -3.86278214782076. Bounds: x₁, x₂, x₃ ∈ [0.0, 1.0].
- Access via TEST_FUNCTIONS dictionary:
    TEST_FUNCTIONS["rosenbrock"]  # Returns ROSENBROCK_FUNCTION
    TEST_FUNCTIONS["sixhumpcamelback"]  # Returns SIXHUMPCAMELBACK_FUNCTION
    TEST_FUNCTIONS["shekel"]  # Returns SHEKEL_FUNCTION
    TEST_FUNCTIONS["hartmann"]  # Returns HARTMANN_FUNCTION

## Metadata
The metadata :start, :min_position, :lb, and :ub are defined as functions accepting a dimension parameter n (default: 2, 3, or 4 depending on the function). Example:
    ROSENBROCK_FUNCTION.meta[:lb](3)  # Returns [-5.0, -5.0, -5.0]
    SIXHUMPCAMELBACK_FUNCTION.meta[:min_position](2)  # Returns [-0.08984201368301331, 0.7126564032704135]
    SHEKEL_FUNCTION.meta[:start](4)  # Returns [5.0, 5.0, 5.0, 5.0]
    HARTMANN_FUNCTION.meta[:ub](3)  # Returns [1.0, 1.0, 1.0]

## Numerische Toleranzen
Gradiententests verwenden eine Toleranz von `atol=1e-3` aufgrund numerischer Instabilitäten bei Funktionen wie Schwefel, Ackley und Rastrigin. Dies betrifft die Prüfung, ob der Gradient im Optimum numerisch null ist, sowie Vergleiche mit numerischen und `ForwardDiff`-Gradienten. Details finden sich in `test/runtests.jl`.

## Demos
Five example scripts in examples/ (10-15 lines each):
- Optimize_all_functions.jl: Optimizes all functions with L-BFGS using tf.gradient!.
- Compare_optimization_methods.jl: Compares Gradient Descent and L-BFGS on Rosenbrock using tf.gradient!.
- List_all_available_test_functions_and_their_properties.jl: Lists functions, start points, minima, properties.
- Optimize_with_nlopt.jl: Optimizes Rosenbrock with NLopt's LD_LBFGS (requires NLopt.jl).
- Compute_hessian_with_zygote.jl: Performs 3 Newton steps on Rosenbrock using Zygote's Hessian.

## Tests
- All tests are consolidated in test/runtests.jl, including function-specific tests (via test/include_testfiles.jl) and cross-function tests. The former file test/test_project.jl was removed to avoid redundancies. Gradient tests, including verification that the gradient is numerically zero at the known optimum (with tolerance atol=1e-3 due to numerical instability of functions like Schwefel, Ackley, Rastrigin), and comparisons with numerical and ForwardDiff gradients at 20 random points (with tolerance atol=1e-3), are exclusively in test/runtests.jl. Current test counts:
    - Cross-Function Tests: 551/551 (including gradient comparisons and verification of zero gradient at the optimum for all differentiable functions)
    - Rosenbrock: 23/23 (function values, metadata, edge cases, optimization)
    - Sphere: 22/22 (function values, metadata, edge cases, optimization)
    - Ackley: 20/20 (function values, metadata, edge cases, optimization)
    - AxisParallelHyperEllipsoid: 16/16 (function values, metadata, edge cases, optimization)
    - Rastrigin: 23/23 (function values, metadata, edge cases, optimization)
    - Griewank: 7/7 (function values, metadata, edge cases, optimization)
    - Schwefel: 27/27 (function values, metadata, edge cases, optimization)
    - Michalewicz: 21/21 (function values, metadata, edge cases, optimization)
    - Branin: 17/17 (function values, metadata, edge cases, optimization, multiple minima)
    - Goldstein-Price: 17/17 (function values, metadata, edge cases, optimization)
    - Shubert: 17/17 (function values, metadata, edge cases, optimization, multiple minima)
    - Six-Hump Camelback: 17/17 (function values, metadata, edge cases, optimization, multiple minima)
    - Shekel: 13/13 (function values, metadata, edge cases, optimization)
    - Hartmann: 13/13 (function values, metadata, edge cases, optimization)
- Run tests:
    cd /c/Users/uweal/NonlinearOptimizationTestFunctionsInJulia
    julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'

## Features
- Scalable: Functions and metadata support arbitrary dimensions via parameter n (where applicable).
- Robust: Handles edge cases (NaN, Inf, 1e-308) with appropriate error handling.
- Compatible: Works with Optim.jl, NLopt, and automatic differentiation (ForwardDiff, Zygote).
- Modular: Test functions loaded via src/include_testfunctions.jl and TEST_FUNCTIONS in src/NonlinearOptimizationTestFunctionsInJulia.jl.
- Numerically Stable Tests: Gradient tests use a tolerance of atol=1e-3 to account for numerical instability in functions like Schwefel, Ackley, and Rastrigin.
- Precise Minima: Minima positions and values are sourced from multiple reliable references (e.g., al-roomi.org, sfu.ca) to avoid errors from rounded literature values.

## Contributing
- Add new test functions in src/functions/ and include them in src/include_testfunctions.jl.
- Ensure new functions provide f, grad, and meta with required keys (:name, :start, :min_position, :min_value, :properties, :lb, :ub, :in_molga_smutnicki_2005).
- **WICHTIGES VERBOT**: Do not define the `gradient!` function explicitly in any test function. It is automatically generated by the `TestFunction` constructor in src/NonlinearOptimizationTestFunctionsInJulia.jl as `(G, x) -> copyto!(G, grad(x))`. Explicit definitions lead to redundancies and inconsistencies.
- Run tests to verify compatibility.
- Use lowercase symbols for function names and files, and avoid debugging outputs.
- Research minima positions and values from multiple sources (e.g., al-roomi.org, sfu.ca, geatbx.com) to ensure precision, as literature values (e.g., Molga & Smutnicki, 2005) are often rounded.
- Note: The Langermann function is deferred for future implementation and should not be added until explicitly requested.

## Note
- Properties are stored in lowercase. Use lowercase when calling has_property, e.g., has_property(tf, "multimodal") instead of has_property(tf, "Multimodal").
- The metadata property :in_molga_smutnicki_2005 is included for all functions to indicate their origin from *Test functions for optimization needs* (Molga & Smutnicki, 2005).
- Code examples in this documentation use 4-space indentation instead of triple backticks (```) to avoid rendering issues in some browsers.

## License
MIT License