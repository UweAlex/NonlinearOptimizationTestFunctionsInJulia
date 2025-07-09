function rosenbrock(x::Vector{Float64})
    @assert length(x) >= 2 "Rosenbrock requires at least 2 dimensions"
    n = length(x)
    if any(isnan.(x)) return NaN end
    if any(isinf.(x)) return Inf end
    return sum(100.0 * (x[2:n] .- x[1:n-1].^2).^2 .+ (1 .- x[1:n-1]).^2)
end

function rosenbrock_gradient(x::Vector{Float64})
    @assert length(x) >= 2 "Rosenbrock requires at least 2 dimensions"
    n = length(x)
    if any(isnan.(x)) return fill(NaN, n) end
    if any(isinf.(x)) return fill(Inf, n) end
    grad = zeros(Float64, n)
    grad[1:n-1] .= -400.0 .* x[1:n-1] .* (x[2:n] .- x[1:n-1].^2) .- 2.0 .* (1 .- x[1:n-1])
    grad[2:n] .+= 200.0 .* (x[2:n] .- x[1:n-1].^2)
    return grad
end

const ROSENBROCK_FUNCTION = TestFunction(
    rosenbrock,
    rosenbrock_gradient,
    [0.0, 0.0],
    [1.0, 1.0],
    0.0,
    Dict(
        :description => "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2",
        :math => "f(x) = \\sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]"
    ),
    "Rosenbrock",
    Set(["multimodal", "differentiable", "scalable"])
)