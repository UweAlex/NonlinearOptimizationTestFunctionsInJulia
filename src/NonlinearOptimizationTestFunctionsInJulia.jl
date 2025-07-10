# src\NonlinearOptimizationTestFunctionsInJulia.jl
module NonlinearOptimizationTestFunctionsInJulia

using LinearAlgebra

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
    gradient!::Function  # Neues Feld für in-place Gradientenberechnung
    start::Vector{Float64}
    min_position::Vector{Float64}
    min_value::Float64
    info::Dict
    name::String
    properties::Set{String}
    function TestFunction(f, grad, start, min_position, min_value, info, name, properties)
        @assert all(p in VALID_PROPERTIES for p in properties) "Invalid properties: $(setdiff(properties, VALID_PROPERTIES))."
        # Erstelle die gradient!-Funktion, die tf.grad(x) in G kopiert
        gradient! = (G, x) -> copyto!(G, grad(x))
        new(f, grad, gradient!, start, min_position, min_value, info, name, properties)
    end
end

# Prüft, ob eine Eigenschaft vorhanden ist
function has_property(tf::TestFunction, prop::String)
    return lowercase(prop) in tf.properties
end

# Fügt eine Eigenschaft hinzu
function add_property(tf::TestFunction, prop::String)
    lprop = lowercase(prop)
    @assert lprop in VALID_PROPERTIES "Invalid property: $lprop."
    new_properties = union(tf.properties, [lprop])
    return TestFunction(tf.f, tf.grad, tf.start, tf.min_position, tf.min_value, tf.info, tf.name, new_properties)
end

# Hilfsfunktion zum Evaluieren
function use_testfunction(tf::TestFunction, x::Vector{Float64})
    @assert !isempty(x) "Input vector x must not be empty"
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

# Sammle alle TestFunction-Konstanten
for name in names(@__MODULE__, all=true)
    if endswith(string(name), "_FUNCTION")
        tf = getfield(@__MODULE__, name)
        if tf isa TestFunction
            TEST_FUNCTIONS[tf.name] = tf
        end
    end
end

# Exportiere Funktionen, Gradienten und Konstanten
for tf in values(TEST_FUNCTIONS)
    export_name = Symbol(lowercase(tf.name))
    export_gradient = Symbol(lowercase(tf.name) * "_gradient")
    export_constant = Symbol(uppercase(tf.name) * "_FUNCTION")
    @eval export $export_name
    @eval export $export_gradient
    @eval export $export_constant
end

export TEST_FUNCTIONS, filter_testfunctions, TestFunction, use_testfunction, has_property, add_property

end # module