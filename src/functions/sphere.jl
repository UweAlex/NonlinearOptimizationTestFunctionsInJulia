    # src/functions/sphere.jl
    # Purpose: Implements the Sphere test function with its gradient for nonlinear optimization.
    # Context: Part of NonlinearOptimizationTestFunctionsInJulia, used in optimization demos and tests.
    # Last modified: 13. Juli 2025, 22:53 PM CEST

    function sphere(x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}}
        any(isnan.(x)) && return T(NaN)
        any(isinf.(x)) && return T(Inf)
        return sum(x.^2)
    end

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
            :start => (n::Int=2) -> fill(0.0, n),
            :min_position => (n::Int=2) -> fill(0.0, n),
            :min_value => 0.0,
            :properties => Set(["unimodal", "convex", "separable", "differentiable", "scalable", "bounded"]),
            :lb => (n::Int=2) -> fill(-5.12, n),
            :ub => (n::Int=2) -> fill(5.12, n),
            :description => "Sphere function: f(x) = Σ x_i^2",
            :math => "f(x) = \\sum_{i=1}^n x_i^2"
        )
    )