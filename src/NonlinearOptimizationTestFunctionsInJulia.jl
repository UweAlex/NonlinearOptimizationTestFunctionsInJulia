module NonlinearOptimizationTestFunctionsInJulia

using LinearAlgebra

# TestFunction Struktur mit Klassifikationsmerkmalen
struct TestFunction
    f::Function
    grad::Function
    start::Vector{Float64}
    info::Dict
    name::String
    is_convex::Bool
    is_concave::Bool
    has_constraints::Bool
    is_multimodal::Bool
    is_differentiable::Bool
    is_separable::Bool
    is_scalable::Bool
end

# Strukturierte Nutzung (makrofrei)
function use_testfunction(tf::TestFunction, x::Vector{Float64})
    return (f=tf.f(x), grad=tf.grad(x))
end

# Rosenbrock-Funktion
function rosenbrock(x::Vector{Float64})
    return sum(100.0 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:length(x)-1)
end

function rosenbrock_gradient(x::Vector{Float64})
    n = length(x)
    grad = zeros(Float64, n)
    for i in 1:n-1
        grad[i] = -400.0 * x[i] * (x[i+1] - x[i]^2) - 2 * (1 - x[i])
        grad[i+1] += 200.0 * (x[i+1] - x[i]^2)
    end
    return grad
end

const ROSENBROCK_FUNCTION = TestFunction(
    rosenbrock,
    rosenbrock_gradient,
    [0.0, 0.0],
    Dict(:description => "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2", :math => "f(x) = \\sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]"),
    "Rosenbrock",
    false, # is_convex
    false, # is_concave
    false, # has_constraints
    true,  # is_multimodal
    true,  # is_differentiable
    false, # is_separable
    true   # is_scalable
)

# Sphere-Funktion
function sphere(x::Vector{Float64})
    return sum(x.^2)
end

function sphere_gradient(x::Vector{Float64})
    return 2.0 .* x
end

const SPHERE_FUNCTION = TestFunction(
    sphere,
    sphere_gradient,
    [0.0, 0.0],
    Dict(:description => "Sphere function: f(x) = Σx_i^2", :math => "f(x) = \\sum_{i=1}^n x_i^2"),
    "Sphere",
    true,  # is_convex
    false, # is_concave
    false, # has_constraints
    false, # is_multimodal
    true,  # is_differentiable
    true,  # is_separable
    true   # is_scalable
)

# Auto-Discovery
const TEST_FUNCTIONS = TestFunction[ROSENBROCK_FUNCTION, SPHERE_FUNCTION]

export rosenbrock, rosenbrock_gradient, ROSENBROCK_FUNCTION, sphere, sphere_gradient, SPHERE_FUNCTION, TEST_FUNCTIONS, use_testfunction

end # module
