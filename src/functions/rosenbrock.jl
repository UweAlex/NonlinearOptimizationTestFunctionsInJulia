# src/functions/rosenbrock.jl
# Purpose: Implements the Rosenbrock test function with its gradient for nonlinear optimization.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, used in optimization demos and tests.
# Last modified: 13. Juli 2025, 10:45 AM CEST
function rosenbrock(x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}}
    length(x) >= 2 || throw(ArgumentError("Rosenbrock requires at least 2 dimensions"))
    n = length(x)
    any(isnan.(x)) && return T(NaN)
    any(isinf.(x)) && return T(Inf)
    sum(100.0 * (x[2:n] .- x[1:n-1].^2).^2 .+ (1 .- x[1:n-1]).^2)
end
function rosenbrock_gradient(x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}}
    length(x) >= 2 || throw(ArgumentError("Rosenbrock requires at least 2 dimensions"))
    n = length(x)
    any(isnan.(x)) &&

 return fill(T(NaN), n)
    any(isinf.(x)) && return fill(T(Inf), n)
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
        :start => (n::Int=2) -> begin
            n >= 2 || throw(ArgumentError("Rosenbrock requires at least 2 dimensions"))
            fill(0.0, n)
        end,
        :min_position => (n::Int=2) -> begin
            n >= 2 || throw(ArgumentError("Rosenbrock requires at least 2 dimensions"))
            fill(1.0, n)
        end,
        :min_value => 0.0,
        :properties => Set(["multimodal", "non-convex", "non-separable", "differentiable", "scalable"]),
        :lb => (n::Int=2) -> begin
            n >= 2 || throw(ArgumentError("Rosenbrock requires at least 2 dimensions"))
            fill(-5.0, n)
        end,
        :ub => (n::Int=2) -> begin
            n >= 2 || throw(ArgumentError("Rosenbrock requires at least 2 dimensions"))
            fill(5.0, n)
        end,
        :description => "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2",
        :math => "f(x) = \\sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]"
    )
)