# src/functions/sphere.jl
# Purpose: Implements the Sphere test function with its gradient for nonlinear optimization.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, used in optimization demos and tests.
# Last modified: 11. Juli 2025, 09:55 AM CEST

function sphere(x::Vector{T}) where {T<:Real}
    if any(isnan.(x)) return T(NaN) end
    if any(isinf.(x)) return T(Inf) end
    return sum(x.^2)
end

function sphere_gradient(x::Vector{T}) where {T<:Real}
    if any(isnan.(x)) return fill(T(NaN), length(x)) end
    if any(isinf.(x)) return fill(T(Inf), length(x)) end
    return 2.0 * x
end

const SPHERE_FUNCTION = TestFunction(
    sphere,
    sphere_gradient,
    Dict(
        :name => "Sphere",
        :start => [0.0, 0.0],
        :min_position => [0.0, 0.0],
        :min_value => 0.0,
        :properties => Set(["unimodal", "convex", "separable", "differentiable", "scalable"]),
        :description => "Sphere function: f(x) = Σ x_i^2",
        :math => "f(x) = \\sum_{i=1}^n x_i^2"
    )
)