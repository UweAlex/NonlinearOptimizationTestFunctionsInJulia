function sphere(x::Vector{Float64})
    if any(isnan.(x)) return NaN end
    if any(isinf.(x)) return Inf end
    return sum(x.^2)
end

function sphere_gradient(x::Vector{Float64})
    if any(isnan.(x)) return fill(NaN, length(x)) end
    if any(isinf.(x)) return fill(Inf, length(x)) end
    return 2.0 * x
end

const SPHERE_FUNCTION = TestFunction(
    sphere,
    sphere_gradient,
    [0.0, 0.0],
    [0.0, 0.0],
    0.0,
    Dict(
        :description => "Sphere function: f(x) = Σ x_i^2",
        :math => "f(x) = \\sum_{i=1}^n x_i^2"
    ),
    "Sphere",
    Set(["convex", "separable", "differentiable", "scalable"])
)