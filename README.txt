NonlinearOptimizationTestFunctionsInJulia

A Julia package for nonlinear optimization test functions with analytical gradients and systematic classification (e.g., convexity, multimodality).

Installation

To install, clone the repository and activate the project environment:
using Pkg
Pkg.activate(".")
Pkg.add("LinearAlgebra")
Pkg.add("Test")
Pkg.add("Optim")
# After registration in the Julia General Registry:
# Pkg.add("NonlinearOptimizationTestFunctionsInJulia")

Requires Julia 1.11.5 or higher.

Available Test Functions

Rosenbrock Function
- Definition: f(x) = Σ_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
- Gradient: Analytically implemented.
- Properties: multimodal, non-convex, non-separable, differentiable, scalable
- Starting Point: [0.0, 0.0, ...]
- Minimum Position: [1.0, 1.0, ...]
- Minimum Value: 0.0

Sphere Function
- Definition: f(x) = Σ_{i=1}^n x_i^2
- Gradient: Analytically implemented.
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
function optimize_rosenbrock()
    x0 = ROSENBROCK_FUNCTION.start
    result = optimize(rosenbrock, rosenbrock_gradient!, x0, LBFGS(), Optim.Options(iterations=100))
    println("Minimum found at: ", Optim.minimizer(result))
    println("Objective value: ", Optim.minimum(result))
end
julia> optimize_rosenbrock()

Demo Scripts:
- examples/Optimize_all_functions.jl: Optimizes all functions with BFGS, showing minimizer, minimum value, iterations, and convergence status.
  julia> include("examples/Optimize_all_functions.jl")
- examples/Compare_optimization_methods.jl: Compares Gradient Descent and L-BFGS on the Rosenbrock function.
  julia> include("examples/Compare_optimization_methods.jl")
- examples/List_all_available_test_functions_and_their_properties.jl: Lists all functions and their properties.
  julia> include("examples/List_all_available_test_functions_and_their_properties.jl")

Changes (as of July 10, 2025)
- Test Suite: 60 tests passed successfully (28 for Rosenbrock, 28 for Sphere, 4 for Filter) in test/runtests.jl, covering function values, gradients, edge cases, properties, and filtering.
- Performance: Vectorized implementation for scalability.
- Structure: Test functions modularized in src/functions/, included via include_testfunctions.jl.

Comparison with Other Packages
- Opfunu: Provides >500 benchmark functions (incl. CEC 2005–2022) in Python/NumPy, without analytical gradients. Our library offers Julia-native analytical gradients for higher performance and integration with Optim.jl.
- CUTEst: Comprehensive collection (>1000 problems), but complex setup (SIF format, Fortran/C dependencies). Our library is Julia-native and more accessible.
- Optim.jl: Optimization algorithms (e.g., L-BFGS, Gradient Descent), no test functions. Our library complements it with standardized test functions.
- NLopt: C-based algorithms with Julia wrapper, no test functions. Our library provides compatible test functions.

Contributing
Contributions are welcome! Submit issues or pull requests at:
https://github.com/UweAlex/NonlinearOptimizationTestFunctionsInJulia

Adding New Functions
1. Create a file in src/functions/ (e.g., new_function.jl).
2. Define function, gradient, and const NEW_FUNCTION = TestFunction(...).
3. Add include("functions/new_function.jl") to src/include_testfunctions.jl.