# test/test_project.jl
# Purpose: Test NonlinearOptimizationTestFunctionsInJulia with Zygote, without NLopt
# Context: Ensures functionality without optional dependencies
# Last modified: 11. Juli 2025, 14:10 PM CEST
using Pkg
println("Julia-Version: ", VERSION)
Pkg.activate(".")
Pkg.instantiate()
using NonlinearOptimizationTestFunctionsInJulia
using Test
println("\nFühre alle Tests aus...")
include("test/runtests.jl")
println("\nManuelle Tests für Rosenbrock und Sphere...")
@test rosenbrock([1.0, 1.0]) ≈ 0.0
@test rosenbrock([0.5, 0.5]) ≈ 6.5
@test rosenbrock_gradient([1.0, 1.0]) ≈ [0.0, 0.0]
println("Rosenbrock: Funktionswerte und Gradienten OK")
@test sphere([0.0, 0.0]) ≈ 0.0
@test sphere([1.0, 1.0]) ≈ 2.0
@test sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]
println("Sphere: Funktionswerte und Gradienten OK")
x_underflow = fill(1e-308, 2)
@test isfinite(rosenbrock(x_underflow))
@test all(isfinite, rosenbrock_gradient(x_underflow))
@test isnan(rosenbrock([NaN, 1.0]))
@test isinf(rosenbrock([Inf, 1.0]))
@test isfinite(sphere(x_underflow))
@test isnan(sphere([NaN, 1.0]))
@test isinf(sphere([Inf, 1.0]))
println("Randfälle (NaN, Inf, 1e-308): OK")
using ForwardDiff
x_dual = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, 0.0)]
@test isfinite(rosenbrock(x_dual))
@test all(isfinite, rosenbrock_gradient(x_dual))
@test isfinite(sphere(x_dual))
@test all(isfinite, sphere_gradient(x_dual))
println("ForwardDiff-Kompatibilität: OK")
using Zygote, LinearAlgebra
@test size(Zygote.hessian(rosenbrock, [0.5, 0.5])) == (2, 2)
@test size(Zygote.hessian(sphere, [1.0, 1.0])) == (2, 2)
println("Zygote Hessian: OK")
n = 2
@test ROSENBROCK_FUNCTION.meta[:name] == "Rosenbrock"
@test ROSENBROCK_FUNCTION.meta[:lb](n) == fill(-5.0, n)
@test SPHERE_FUNCTION.meta[:name] == "Sphere"
@test SPHERE_FUNCTION.meta[:ub](n) == fill(5.12, n)
println("Metadaten: OK")
@test length(filter_testfunctions(tf -> has_property(tf, "multimodal"))) == 1
@test length(filter_testfunctions(tf -> has_property(tf, "convex"))) == 1
tf_modified = add_property(ROSENBROCK_FUNCTION, "bounded")
@test has_property(tf_modified, "bounded")
println("Filter und Eigenschaften: OK")
println("\nTeste Fehlerbehandlung...")
try
    rosenbrock([1.0])
    println("Fehler: Rosenbrock mit einer Dimension sollte fehlschlagen!")
catch e
    @test e isa ArgumentError
    println("Fehlerbehandlung (Dimensionen): OK")
end
try
    has_property(ROSENBROCK_FUNCTION, "invalid_property")
    println("Fehler: Ungültige Eigenschaft sollte fehlschlagen!")
catch e
    @test e isa ArgumentError
    println("Fehlerbehandlung (Eigenschaften): OK")
end
println("\nFühre Demos aus (ohne NLopt)...")
include("examples/Optimize_all_functions.jl")
include("examples/Compare_optimization_methods.jl")
include("examples/List_all_available_test_functions_and_their_properties.jl")
include("examples/Compute_hessian_with_zygote.jl")
println("Demos (ohne NLopt): OK")
println("\nAlle Tests abgeschlossen. Überprüfe die Ausgabe auf Fehler.")