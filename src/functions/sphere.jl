# src/functions/sphere.jl
# Purpose: Implements the Sphere test function with its gradient for nonlinear optimization.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, used in optimization demos and tests.
# Last modified: 11. Juli 2025, 10:23 AM CEST

"""
    sphere(x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}} -> T
Computes the Sphere function: f(x) = Σ x_i^2.
Returns NaN for NaN inputs and Inf for Inf inputs.
"""
function sphere(x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}}
    any(isnan.(x)) && return T(NaN)
    any(isinf.(x)) && return T(Inf)
    return sum(x.^2)
end

"""
    sphere_gradient(x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}} -> Vector{T}
Computes the gradient of the Sphere function: ∇f(x) = 2x.
Returns a vector of NaN for NaN inputs and Inf for Inf inputs.
"""
function sphere_gradient(x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}}
    any(isnan.(x)) && return fill(T(NaN), length(x))
    any(isinf.(x)) && return fill(T(Inf), length(x))
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
        :lb => fill(-5.12, 2),
        :ub => fill(5.12, 2),
        :description => "Sphere function: f(x) = Σ x_i^2",
        :math => "f(x) = \\sum_{i=1}^n x_i^2"
    )
)