module NonlinearOptimizationTestFunctionsInJulia
using LinearAlgebra

# Struktur für Testfunktionen
struct TestFunction
    f::Function
    grad::Function
    start::Vector{Float64}
    min_position::Vector{Float64}
    min_value::Float64
    info::Dict
    name::String
    properties::Set{String}  # Eigenschaften als Set, kleingeschrieben
end

# Prüft, ob eine Eigenschaft vorhanden ist (kleingeschrieben)
function has_property(tf::TestFunction, prop::String)
    return lowercase(prop) in tf.properties
end

# Fügt eine Eigenschaft hinzu und gibt ein neues TestFunction zurück
function add_property(tf::TestFunction, prop::String)
    new_properties = union(tf.properties, [lowercase(prop)])
    return TestFunction(
        tf.f,
        tf.grad,
        tf.start,
        tf.min_position,
        tf.min_value,
        tf.info,
        tf.name,
        new_properties
    )
end

# Hilfsfunktion zum Evaluieren einer Testfunktion
function use_testfunction(tf::TestFunction, x::Vector{Float64})
    @assert !isempty(x) "Input vector x must not be empty"
    return (f=tf.f(x), grad=tf.grad(x))
end

# Hilfsfunktion zum Filtern von Testfunktionen
function filter_testfunctions(predicate::Function)
    return [tf for tf in values(TEST_FUNCTIONS) if predicate(tf)]
end

function filter_testfunctions(test_functions::Dict{String, TestFunction}, predicate::Function)
    return [tf for tf in values(test_functions) if predicate(tf)]
end

# Dictionary für alle Testfunktionen
const TEST_FUNCTIONS = Dict{String, TestFunction}()

# Einzige Include-Anweisung für die zentrale Datei
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