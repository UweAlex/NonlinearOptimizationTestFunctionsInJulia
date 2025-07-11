# src/functions/rosenbrock.jl
# Purpose: Implements the Rosenbrock test function with its gradient for nonlinear optimization.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, used in optimization demos and tests.
# Last modified: 11. Juli 2025, 09:55 AM CEST

function rosenbrock(x::Vector{T}) where {T<:Real}
    @assert length(x) >= 2 "Rosenbrock requires at least 2 dimensions"
    n = length(x)
    if any(isnan.(x)) return T(NaN) end
    if any(isinf.(x)) return T(Inf) end
    sum(100.0 * (x[2:n] .- x[1:n-1].^2).^2 .+ (1 .- x[1:n-1]).^2)
end

function rosenbrock_gradient(x::Vector{T}) where {T<:Real}
    @assert length(x) >= 2 "Rosenbrock requires at least 2 dimensions"
    n = length(x)
    if any(isnan.(x)) return fill(T(NaN), n) end
    if any(isinf.(x)) return fill(T(Inf), n) end
    grad = zeros(T, n)
    grad[1:n-1] .= -400.0 .* x[1:n-1] .* (x[2:n] .- x[1:n-1].^2) .- 2.0 .* (1 .- x[1:n-1])
    grad[2:n] .+= 200.0 .* (x[2:n] .- x[1:n-1].^2)
    return grad
end

const ROSENBROCK_FUNCTION = TestFunction(
    rosenbrock,
    rosenbrock_gradient,
    Dict(
        :name => "Rosenbrock",
        :start => [0.0, 0.0],
        :min_position => [1.0, 1.0],
        :min_value => 0.0,
        :properties => Set(["multimodal", "non-convex", "non-separable", "differentiable", "scalable"]),
        :description => "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2",
        :math => "f(x) = \\sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]"
    )
)