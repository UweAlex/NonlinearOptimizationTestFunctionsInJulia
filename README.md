# README.md
# NonlinearOptimizationTestFunctionsInJulia

A Julia package for nonlinear optimization test functions with analytical gradients and systematic classification.

## Rosenbrock Function
**Mathematical Definition**:
\[ f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2] \]

**Gradient**:
\[ \nabla f(x)_i = -400 x_i (x_{i+1} - x_i^2) - 2(1 - x_i) \text{ for } i=1,\dots,n-1 \]
\[ \nabla f(x)_{i+1} += 200 (x_{i+1} - x_i^2) \]

**Properties**:
- Convex: No
- Multimodal: Yes
- Differentiable: Yes
- Separable: No
- Scalable: Yes
- Constraints: None
- Starting Point: `[0.0, 0.0, ...]`

**Usage**:
```julia
julia> rosenbrock([1.0, 1.0])
0.0
julia> rosenbrock_gradient([1.0, 1.0])
2-element Vector{Float64}: [0.0, 0.0]