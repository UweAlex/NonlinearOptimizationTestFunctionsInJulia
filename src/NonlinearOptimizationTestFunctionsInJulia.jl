# src/NonlinearOptimizationTestFunctionsInJulia.jl
# Purpose: Defines the core module for nonlinear optimization test functions.
# Context: Provides TestFunction structure, metadata validation, and function registry.
# Last modified: 11. Juli 2025, 10:14 AM CEST

module NonlinearOptimizationTestFunctionsInJulia

using LinearAlgebra
using ForwardDiff

# Zulässige Eigenschaften
const VALID_PROPERTIES = Set([
    "unimodal", "multimodal", "highly multimodal", "deceptive",
    "convex", "non-convex", "quasi-convex", "strongly convex",
    "separable", "non-separable", "partially separable", "fully non-separable",
    "differentiable", "scalable", "continuous", "bounded", "has_constraints"
])

# Struktur für Testfunktionen
struct TestFunction
    f::Function
    grad::Function
    gradient!::Function
    meta::Dict{Symbol, Any}
    function TestFunction(f, grad, meta)
        required_keys = [:name, :start, :min_position, :min_value, :properties, :lb, :ub]
        missing_keys = setdiff(required_keys, keys(meta))
        isempty(missing_keys) || throw(ArgumentError("Missing required meta keys: $missing_keys"))
        meta[:properties] isa Set || throw(ArgumentError("meta[:properties] must be a Set"))
        all(p in VALID_PROPERTIES for p in meta[:properties]) || throw(ArgumentError("Invalid properties: $(setdiff(meta[:properties], VALID_PROPERTIES))"))
        gradient! = (G, x) -> copyto!(G, grad(x))
        new(f, grad, gradient!, meta)
    end
end

# Prüft, ob eine Eigenschaft vorhanden ist
function has_property(tf::TestFunction, prop::String)
    lprop = lowercase(prop)
    lprop in VALID_PROPERTIES || throw(ArgumentError("Invalid property: $lprop"))
    return lprop in tf.meta[:properties]
end

# Fügt eine Eigenschaft hinzu
function add_property(tf::TestFunction, prop::String)
    lprop = lowercase(prop)
    lprop in VALID_PROPERTIES || throw(ArgumentError("Invalid property: $lprop"))
    new_meta = copy(tf.meta)
    new_meta[:properties] = union(tf.meta[:properties], [lprop])
    return TestFunction(tf.f, tf.grad, new_meta)
end

# Hilfsfunktion zum Evaluieren
function use_testfunction(tf::TestFunction, x::Vector{T}) where {T<:Union{Real, ForwardDiff.Dual}}
    isempty(x) && throw(ArgumentError("Input vector x must not be empty"))
    return (f=tf.f(x), grad=tf.grad(x))
end

# Hilfsfunktion zum Filtern
function filter_testfunctions(predicate::Function)
    return [tf for tf in values(TEST_FUNCTIONS) if predicate(tf)]
end

function filter_testfunctions(test_functions::Dict{String, TestFunction}, predicate::Function)
    return [tf for tf in values(test_functions) if predicate(tf)]
end

# Dictionary für alle Testfunktionen
const TEST_FUNCTIONS = Dict{String, TestFunction}()

# Einzige Include-Anweisung
include("include_testfunctions.jl")

# Sammle alle TestFunction-Konstanten mit Try-Catch
for name in names(@__MODULE__, all=true)
    try
        if endswith(string(name), "_FUNCTION")
            tf = getfield(@__MODULE__, name)
            if tf isa TestFunction
                TEST_FUNCTIONS[tf.meta[:name]] = tf
            end
        end
    catch e
        @warn "Failed to load TestFunction for $name: $e"
    end
end

# Sicherer Export von Funktionen, Gradienten und Konstanten
for tf in values(TEST_FUNCTIONS)
    export_name = Symbol(lowercase(tf.meta[:name]))
    export_gradient = Symbol(lowercase(tf.meta[:name]) * "_gradient")
    export_constant = Symbol(uppercase(tf.meta[:name]) * "_FUNCTION")
    isdefined(@__MODULE__, export_name) && @eval export $export_name
    isdefined(@__MODULE__, export_gradient) && @eval export $export_gradient
    isdefined(@__MODULE__, export_constant) && @eval export $export_constant
end

export TEST_FUNCTIONS, filter_testfunctions, TestFunction, use_testfunction, has_property, add_property

end # module