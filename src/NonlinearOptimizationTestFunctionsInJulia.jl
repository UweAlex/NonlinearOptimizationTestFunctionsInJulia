# NonlinearOptimizationTestFunctionsInJulia: A Julia module providing test functions for nonlinear optimization,
# including properties like modality, convexity, and separability for filtering and analysis.
module NonlinearOptimizationTestFunctionsInJulia
using LinearAlgebra

# Valid properties for test functions, describing characteristics like modality, convexity, separability.
const VALID_PROPERTIES = Set([
    "unimodal", "multimodal", "highly multimodal", "deceptive",
    "convex", "non-convex", "quasi-convex", "strongly convex",
    "separable", "non-separable", "partially separable", "fully non-separable",
    "differentiable", "scalable", "continuous", "bounded", "has_constraints"
])

# TestFunction struct: Represents an optimization test function with its objective function, gradient, etc.
struct TestFunction
    f::Function
    grad::Function
    start::Vector{Float64}
    min_position::Vector{Float64}
    min_value::Float64
    info::Dict
    name::String
    properties::Set{String}
    function TestFunction(f, grad, start, min_position, min_value, info, name, properties)
        @assert all(p in VALID_PROPERTIES for p in properties) "Invalid properties: $(setdiff(properties, VALID_PROPERTIES))."
        new(f, grad, start, min_position, min_value, info, name, properties)
    end
end

# Checks if a test function has a specific property (case-insensitive).
function has_property(tf::TestFunction, prop::String)
    return lowercase(prop) in tf.properties
end

# Adds a new property to a test function, ensuring it is valid.
function add_property(tf::TestFunction, prop::String)
    lprop = lowercase(prop)
    @assert lprop in VALID_PROPERTIES "Invalid property: $lprop."
    new_properties = union(tf.properties, [lprop])
    return TestFunction(tf.f, tf.grad, tf.start, tf.min_position, tf.min_value, tf.info, tf.name, new_properties)
end

# Evaluates a test function at point x, returning its value and gradient.
function use_testfunction(tf::TestFunction, x::Vector{Float64})
    @assert !isempty(x) "Input vector x must not be empty"
    return (f=tf.f(x), grad=tf.grad(x))
end

# Filters test functions based on a predicate (e.g., has_property).
function filter_testfunctions(predicate::Function)
    return [tf for tf in values(TEST_FUNCTIONS) if predicate(tf)]
end

function filter_testfunctions(test_functions::Dict{String, TestFunction}, predicate::Function)
    return [tf for tf in values(test_functions) if predicate(tf)]
end

# In-place gradient wrapper for Optim.jl compatibility.
function gradient!(tf::TestFunction)
    return (G, x) -> copyto!(G, tf.grad(x))
end

# Dictionary holding all test functions, keyed by their names.
const TEST_FUNCTIONS = Dict{String, TestFunction}()

# Include test function definitions (e.g., rosenbrock.jl, sphere.jl).
include("include_testfunctions.jl")

# Populate TEST_FUNCTIONS with all TestFunction constants.
for name in names(@__MODULE__, all=true)
    if endswith(string(name), "_FUNCTION")
        tf = getfield(@__MODULE__, name)
        if tf isa TestFunction
            TEST_FUNCTIONS[tf.name] = tf
        end
    end
end

# Export test functions, their gradients, and constants dynamically.
for tf in values(TEST_FUNCTIONS)
    export_name = Symbol(lowercase(tf.name))
    export_gradient = Symbol(lowercase(tf.name) * "_gradient")
    export_constant = Symbol(uppercase(tf.name) * "_FUNCTION")
    @eval export $export_name
    @eval export $export_gradient
    @eval export $export_constant
end

# Export core module components.
export TEST_FUNCTIONS, filter_testfunctions, TestFunction, use_testfunction, has_property, add_property, gradient!

end # module