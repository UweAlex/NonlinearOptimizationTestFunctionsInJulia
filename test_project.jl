# test_project.jl
# Purpose: Test NonlinearOptimizationTestFunctionsInJulia with Zygote, without NLopt
# Last modified: 11. Juli 2025, 12:16 PM CEST

# Schritt 1: Bestätige Julia-Version
println("Julia-Version: ", VERSION)
@assert VERSION >= v"1.11.5" "Julia 1.11.5+ ist erforderlich! Installation bestätigt."

# Schritt 2: Aktiviere das Projekt
using Pkg
Pkg.activate(".")  # Aktiviere das Projektverzeichnis
Pkg.instantiate()  # Stelle sicher, dass LinearAlgebra, Optim, Test, ForwardDiff, Zygote installiert sind

# Schritt 3: Lade das Modul
using NonlinearOptimizationTestFunctionsInJulia
using Test

# Schritt 4: Führe alle Tests aus (runtests.jl)
println("\nFühre alle Tests aus...")
include("test/runtests.jl")  # Sollte 60 Tests bestehen
# Erwartete Ausgabe: "Test Summary: NonlinearOptimizationTestFunctionsInJulia Tests | 60 passed, 60 total"

# Schritt 5: Manuelle Tests für kritische Funktionen
println("\nManuelle Tests für Rosenbrock und Sphere...")

# Rosenbrock-Funktion
@test rosenbrock([1.0, 1.0]) ≈ 0.0  # Minimum
@test rosenbrock([0.5, 0.5]) ≈ 6.5  # Funktionswert
@test rosenbrock_gradient([1.0, 1.0]) ≈ [0.0, 0.0]  # Gradient im Minimum
println("Rosenbrock: Funktionswerte und Gradienten OK")

# Sphere-Funktion
@test sphere([0.0, 0.0]) ≈ 0.0  # Minimum
@test sphere([1.0, 1.0]) ≈ 2.0  # Funktionswert
@test sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]  # Gradient
println("Sphere: Funktionswerte und Gradienten OK")

# Randfälle
x_underflow = fill(1e-308, 2)
@test isfinite(rosenbrock(x_underflow))  # Unterflow
@test all(isfinite, rosenbrock_gradient(x_underflow))
@test isnan(rosenbrock([NaN, 1.0]))  # NaN
@test isinf(rosenbrock([Inf, 1.0]))  # Inf
@test isfinite(sphere(x_underflow))
@test isnan(sphere([NaN, 1.0]))
@test isinf(sphere([Inf, 1.0]))
println("Randfälle (NaN, Inf, 1e-308): OK")

# ForwardDiff-Kompatibilität
using ForwardDiff
x_dual = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, 0.0)]
@test isfinite(rosenbrock(x_dual))
@test all(isfinite, rosenbrock_gradient(x_dual))
@test isfinite(sphere(x_dual))
@test all(isfinite, sphere_gradient(x_dual))
println("ForwardDiff-Kompatibilität: OK")

# Zygote Hessian
using Zygote, LinearAlgebra
@test size(Zygote.hessian(rosenbrock, [0.5, 0.5])) == (2, 2)
@test size(Zygote.hessian(sphere, [1.0, 1.0])) == (2, 2)
println("Zygote Hessian: OK")

# Metadaten
@test ROSENBROCK_FUNCTION.meta[:name] == "Rosenbrock"
@test ROSENBROCK_FUNCTION.meta[:lb] == fill(-5.0, 2)
@test SPHERE_FUNCTION.meta[:name] == "Sphere"
@test SPHERE_FUNCTION.meta[:ub] == fill(5.12, 2)
println("Metadaten: OK")

# Filter und Eigenschaften
@test length(filter_testfunctions(tf -> has_property(tf, "multimodal"))) == 1  # Nur Rosenbrock
@test length(filter_testfunctions(tf -> has_property(tf, "convex"))) == 1  # Nur Sphere
tf_modified = add_property(ROSENBROCK_FUNCTION, "bounded")
@test has_property(tf_modified, "bounded")
println("Filter und Eigenschaften: OK")

# Fehlerbehandlung
println("\nTeste Fehlerbehandlung...")
try
    rosenbrock([1.0])  # Ungültige Dimension
    println("Fehler: Rosenbrock mit einer Dimension sollte fehlschlagen!")
catch e
    @test e isa ArgumentError
    println("Fehlerbehandlung (Dimensionen): OK")
end

try
    has_property(ROSENBROCK_FUNCTION, "invalid_property")  # Ungültige Eigenschaft
    println("Fehler: Ungültige Eigenschaft sollte fehlschlagen!")
catch e
    @test e isa ArgumentError
    println("Fehlerbehandlung (Eigenschaften): OK")
end

# Schritt 6: Demos ausführen (ohne NLopt)
println("\nFühre Demos aus (ohne NLopt)...")
include("examples/Optimize_all_functions.jl")  # Optimiert alle Funktionen mit L-BFGS
include("examples/Compare_optimization_methods.jl")  # Gradient Descent vs. L-BFGS
include("examples/List_all_available_test_functions_and_their_properties.jl")  # Listet Funktionen
include("examples/Compute_hessian_with_zygote.jl")  # Newton-Schritte mit Zygote
println("Demos (ohne NLopt): OK")

# Schritt 7: Zusammenfassung
println("\nAlle Tests abgeschlossen. Überprüfe die Ausgabe auf Fehler.")