NonlinearOptimizationTestFunctionsInJulia

A Julia package for nonlinear optimization test functions with analytical gradients, systematic classification (e.g., convexity, multimodality), and a user-friendly API for auto-discovery and filtering. Ideal for researchers and developers testing optimization algorithms.

Installation

Clone the repository and activate the project environment:
using Pkg
Pkg.activate(".")
Pkg.add("LinearAlgebra")
Pkg.add("Test")
Pkg.add("Optim")  # For optimization examples
# Once registered in the Julia General Registry:
# Pkg.add("NonlinearOptimizationTestFunctionsInJulia")

Requires Julia 1.11.5 or higher.

Available Test Functions

Rosenbrock Function
- Definition: f(x) = Σ_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
- Gradient: Analytically implemented.
- Properties:
  - Convex: No
  - Multimodal: Yes
  - Differentiable: Yes
  - Separable: No
  - Scalable: Yes
  - Starting Point: [0.0, 0.0, ...]
  - Minimum Position: [1.0, 1.0, ...]
  - Minimum Value: 0.0

Sphere Function
- Definition: f(x) = Σ_{i=1}^n x_i^2
- Gradient: Analytically implemented.
- Properties:
  - Convex: Yes
  - Multimodal: No
  - Differentiable: Yes
  - Separable: Yes
  - Scalable: Yes
  - Starting Point: [0.0, 0.0, ...]
  - Minimum Position: [0.0, 0.0, ...]
  - Minimum Value: 0.0

Usage

Basic Evaluation
using NonlinearOptimizationTestFunctionsInJulia
julia> rosenbrock([0.5, 0.5])
6.5
julia> sphere([1.0, 1.0])
2.0

Optimization Example
using Optim
function optimize_rosenbrock()
    x0 = ROSENBROCK_FUNCTION.start
    result = optimize(rosenbrock, rosenbrock_gradient, x0, BFGS(), Optim.Options(iterations=100))
    println("Minimum found at: ", Optim.minimizer(result))
    println("Objective value: ", Optim.minimum(result))
end
julia> optimize_rosenbrock()
Minimum found at: [0.9999999999373614, 0.999999999868622]
Objective value: 7.645684e-21

Changes (as of July 8, 2025)
- Test Suite: 60 tests passed successfully (28 for Rosenbrock, 28 for Sphere, 4 for Filter).
- New Fields: min_position and min_value added to the TestFunction structure.
- Edge-Case Handling: Support for NaN, ±Inf, overflow, and multidimensional inputs implemented.
- Bug Fixes: Corrected MethodError for empty vector and description of SPHERE_FUNCTION.
- Performance: Vectorized implementation of rosenbrock and rosenbrock_gradient for better scalability.

Contributing
Contributions are welcome! Submit issues or pull requests at:
https://github.com/UweAlex/NonlinearOptimizationTestFunctionsInJulia