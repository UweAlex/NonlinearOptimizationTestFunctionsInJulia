module NonlinearOptimizationTestFunctionsInJulia

using LinearAlgebra

# Struktur für Testfunktionen mit Klassifikationsmerkmalen
struct TestFunction
    f::Function                  # Die Zielfunktion
    grad::Function               # Die Gradientenfunktion
    start::Vector{Float64}       # Startpunkt für die Optimierung
    min_position::Vector{Float64} # Position des Minimums
    min_value::Float64           # Wert am Minimum
    info::Dict                   # Zusätzliche Metadaten (z.B. Beschreibung)
    name::String                 # Name der Funktion
    is_convex::Bool              # Ist die Funktion konvex?
    is_concave::Bool             # Ist die Funktion konkav?
    has_constraints::Bool        # Hat die Funktion Nebenbedingungen?
    is_multimodal::Bool          # Ist die Funktion multimodal?
    is_differentiable::Bool      # Ist die Funktion differenzierbar?
    is_separable::Bool           # Ist die Funktion separierbar?
    is_scalable::Bool            # Ist die Funktion skalierbar?
end

# Hilfsfunktion zur Nutzung einer Testfunktion
"""
    use_testfunction(tf::TestFunction, x::Vector{Float64})

Evaluates the test function and its gradient at a given point `x`.

# Arguments
- `tf`: The `TestFunction` instance to evaluate.
- `x`: The point at which to evaluate the function and gradient.

# Returns
A named tuple with fields `f` (function value) and `grad` (gradient vector).
"""
function use_testfunction(tf::TestFunction, x::Vector{Float64})
    @assert !isempty(x) "Input vector x must not be empty"
    return (f=tf.f(x), grad=tf.grad(x))
end

# Hilfsfunktion zum Filtern von Testfunktionen
"""
    filter_testfunctions(test_functions::Vector{TestFunction}, predicate::Function)

Filters a list of test functions based on a predicate function.

# Arguments
- `test_functions`: A vector of `TestFunction` instances.
- `predicate`: A function that takes a `TestFunction` and returns a `Bool`.

# Returns
A vector of `TestFunction` instances that satisfy the predicate.
"""
function filter_testfunctions(test_functions::Vector{TestFunction}, predicate::Function)
    return [tf for tf in test_functions if predicate(tf)]
end

# Rosenbrock-Funktion (vektorisierte Version für bessere Performance)
"""
    rosenbrock(x::Vector{Float64})

Computes the Rosenbrock function value at point `x`.

# Arguments
- `x`: Input vector.

# Returns
The function value as a `Float64`.
"""
function rosenbrock(x::Vector{Float64})
    @assert length(x) >= 2 "Rosenbrock requires at least 2 dimensions"
    n = length(x)
    if any(isnan.(x)) return NaN end
    if any(isinf.(x)) return Inf end
    return sum(100.0 * (x[2:n] .- x[1:n-1].^2).^2 .+ (1 .- x[1:n-1]).^2)
end

# Gradient der Rosenbrock-Funktion (vektorisierte Version)
"""
    rosenbrock_gradient(x::Vector{Float64})

Computes the gradient of the Rosenbrock function at point `x`.

# Arguments
- `x`: Input vector.

# Returns
The gradient as a `Vector{Float64}`.
"""
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

# Instanz der Rosenbrock-Funktion
const ROSENBROCK_FUNCTION = TestFunction(
    rosenbrock,
    rosenbrock_gradient,
    [0.0, 0.0],           # Startpunkt
    [1.0, 1.0],           # Minimum-Position
    0.0,                  # Minimum-Wert
    Dict(
        :description => "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2",
        :math => "f(x) = \\sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]"
    ),
    "Rosenbrock",
    false,  # is_convex
    false,  # is_concave
    false,  # has_constraints
    true,   # is_multimodal
    true,   # is_differentiable
    false,  # is_separable
    true    # is_scalable
)

# Sphärenfunktion
"""
    sphere(x::Vector{Float64})

Computes the Sphere function value at point `x`.

# Arguments
- `x`: Input vector.

# Returns
The function value as a `Float64`.
"""
function sphere(x::Vector{Float64})
    if any(isnan.(x)) return NaN end
    if any(isinf.(x)) return Inf end
    return sum(x.^2)
end

# Gradient der Sphärenfunktion
"""
    sphere_gradient(x::Vector{Float64})

Computes the gradient of the Sphere function at point `x`.

# Arguments
- `x`: Input vector.

# Returns
The gradient as a `Vector{Float64}`.
"""
function sphere_gradient(x::Vector{Float64})
    if any(isnan.(x)) return fill(NaN, length(x)) end
    if any(isinf.(x)) return fill(Inf, length(x)) end
    return 2.0 * x
end

# Instanz der Sphärenfunktion
const SPHERE_FUNCTION = TestFunction(
    sphere,
    sphere_gradient,
    [0.0, 0.0],           # Startpunkt
    [0.0, 0.0],           # Minimum-Position
    0.0,                  # Minimum-Wert
    Dict(
        :description => "Sphere function: f(x) = Σ x_i^2",
        :math => "f(x) = \\sum_{i=1}^n x_i^2"
    ),
    "Sphere",
    true,   # is_convex
    false,  # is_concave
    false,  # has_constraints
    false,  # is_multimodal
    true,   # is_differentiable
    true,   # is_separable
    true    # is_scalable
)

# Liste aller Testfunktionen
const TEST_FUNCTIONS = TestFunction[ROSENBROCK_FUNCTION, SPHERE_FUNCTION]

# Exportierte Symbole
export rosenbrock, rosenbrock_gradient, ROSENBROCK_FUNCTION,
       sphere, sphere_gradient, SPHERE_FUNCTION,
       TEST_FUNCTIONS, use_testfunction, filter_testfunctions, TestFunction

end # module