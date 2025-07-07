NonlinearOptimizationTestFunctionsInJulia

A Julia package providing test functions for nonlinear optimization, with analytical gradients, systematic classification (e.g., convexity, multimodality), and a flexible API for auto-discovery and filtering.

Installation
using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("Test")
Pkg.add("Optim")  # For optimization examples
# Clone the repository
# ] add https://github.com/UweAlex/NonlinearOptimizationTestFunctionsInJulia
Requires Julia 1.11.5 or later.

Available Test Functions
Rosenbrock Function
Mathematical Definition:
f(x) = sum from i=1 to n-1 [100(x_(i+1) - x_i^2)^2 + (1 - x_i)^2]

Gradient:
nabla f(x)_i = -400 x_i (x_(i+1) - x_i^2) - 2(1 - x_i) for i=1,...,n-1
nabla f(x)_(i+1) += 200 (x_(i+1) - x_i^2)

Properties:
- Convex: No
- Multimodal: Yes
- Differentiable: Yes
- Separable: No
- Scalable: Yes
- Constraints: None
- Starting Point: [0.0, 0.0, ...]

Sphere Function
Mathematical Definition:
f(x) = sum from i=1 to n x_i^2

Gradient:
nabla f(x)_i = 2 x_i

Properties:
- Convex: Yes
- Multimodal: No
- Differentiable: Yes
- Separable: Yes
- Scalable: Yes
- Constraints: None
- Starting Point: [0.0, 0.0, ...]

Usage
Basic Evaluation
using NonlinearOptimizationTestFunctionsInJulia

# Evaluate Rosenbrock function
julia> rosenbrock([0.5, 0.5])
6.5
julia> rosenbrock_gradient([0.5, 0.5])
2-element Vector{Float64}: [-51.0, 50.0]

# Evaluate Sphere function
julia> sphere([1.0, 1.0])
2.0
julia> sphere_gradient([1.0, 1.0])
2-element Vector{Float64}: [2.0, 2.0]

# Filter convex functions
julia> convex_funcs = filter_testfunctions(TEST_FUNCTIONS, tf -> tf.is_convex)
1-element Vector{TestFunction}: [SPHERE_FUNCTION]

Optimization Example
Below is an example of optimizing the Rosenbrock function using the BFGS algorithm from Optim.jl:

using NonlinearOptimizationTestFunctionsInJulia
using Optim

function optimize_rosenbrock()
    x0 = ROSENBROCK_FUNCTION.start
    function objective(x::Vector{Float64})
        return rosenbrock(x)
    end
    function gradient!(G::Vector{Float64}, x::Vector{Float64})
        G .= rosenbrock_gradient(x)
    end
    result = optimize(objective, gradient!, x0, BFGS(), Optim.Options(iterations=100, show_trace=false))
    println("Optimization results for Rosenbrock function:")
    println("Minimum found at: ", Optim.minimizer(result))
    println("Objective value: ", Optim.minimum(result))
    println("Convergence: ", Optim.converged(result))
    println("Iterations: ", Optim.iterations(result))
    return result
end

# Run optimization
julia> optimize_rosenbrock()
Optimization results for Rosenbrock function:
Minimum found at: [0.9999999999373614, 0.999999999868622]
Objective value: 7.645684e-21
Convergence: true
Iterations: 16

Features
- Analytical Gradients: Each test function provides an analytical gradient for optimization algorithms.
- Systematic Classification: Functions are classified by properties like convexity, multimodality, and separability.
- Auto-Discovery: Use TEST_FUNCTIONS to access all available test functions.
- Flexible API: Functions like rosenbrock and rosenbrock_gradient provide direct access to function and gradient evaluations.
- Filtering: filter_testfunctions allows filtering by properties (e.g., is_convex, is_multimodal).

Documentation
See docs/ for interim reports and detailed information, including development progress and test results.

Comparison with Other Packages
Unlike other packages, this project emphasizes a lightweight, Julia-based environment with systematic classification and auto-discovery:
- CUTEst: Offers hundreds of Fortran-based test problems for constrained/unconstrained optimization but lacks systematic classification and requires a more complex setup.
- BenchmarkFunctions.jl: Provides 50+ multi-objective functions but no analytical gradients or classification.
- Optim.jl: Includes ~10 test functions with manual gradients, without systematic classification.
- NLopt: Offers a wide range of optimization algorithms but lacks systematic test function classification and auto-discovery.
- OptimizationTestFunctions.jl: Less active, with limited functionality.

Contributing
Contributions are welcome! Please submit issues or pull requests to https://github.com/UweAlex/NonlinearOptimizationTestFunctionsInJulia.