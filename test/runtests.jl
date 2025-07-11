# test/runtests.jl
# Purpose: Tests for NonlinearOptimizationTestFunctionsInJulia, covering function values, gradients, Hessians, and metadata.
# Context: Ensures correctness of Rosenbrock, Sphere, and filtering functionality.
# Last modified: 11. Juli 2025, 12:15 PM CEST
using Test, ForwardDiff, Zygote, LinearAlgebra
using NonlinearOptimizationTestFunctionsInJulia

# Hilfsfunktion für numerische Gradienten
function finite_difference_gradient(f, x, h=1e-6)
    n = length(x)
    grad = zeros(n)
    for i in 1:n
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2h)
    end
    return grad
end

@testset "NonlinearOptimizationTestFunctionsInJulia Tests" begin
    @testset "Rosenbrock Tests" begin
        # Funktionswerte (2D)
        @test rosenbrock([1.0, 1.0]) ≈ 0.0
        @test rosenbrock([0.5, 0.5]) ≈ 6.5
        @test rosenbrock([-1.0, 1.0]) ≈ 4.0
        # Funktionswerte (3D)
        x_3d = [0.5, 0.5, 0.5]
        @test rosenbrock(x_3d) ≈ 13.0
        # Gradienten (2D)
        @test rosenbrock_gradient([1.0, 1.0]) ≈ [0.0, 0.0]
        @test rosenbrock_gradient([0.5, 0.5])[1] ≈ -51.0 atol=1e-6
        @test rosenbrock_gradient([0.5, 0.5])[2] ≈ 50.0 atol=1e-6
        # Gradienten (3D)
        @test length(rosenbrock_gradient(x_3d)) == 3
        @test rosenbrock_gradient(x_3d)[1] ≈ -51.0 atol=1e-6
        # Randfälle
        x_underflow = fill(1e-308, 2)
        @test isfinite(rosenbrock(x_underflow))
        @test all(isfinite, rosenbrock_gradient(x_underflow))
        @test isnan(rosenbrock([NaN, 1.0]))
        @test isinf(rosenbrock([Inf, 1.0]))
        x_large = fill(1e308, 2)
        @test isinf(rosenbrock(x_large))
        # ForwardDiff Kompatibilität
        x_dual = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, 0.0)]
        @test isfinite(rosenbrock(x_dual))
        @test all(isfinite, rosenbrock_gradient(x_dual))
        # Numerische Gradientengenauigkeit
        x = [0.5, 0.5]
        @test rosenbrock_gradient(x) ≈ finite_difference_gradient(rosenbrock, x) atol=1e-6
        x_3d = [0.5, 0.5, 0.5]
        @test rosenbrock_gradient(x_3d) ≈ finite_difference_gradient(rosenbrock, x_3d) atol=1e-6
        # Eigenschaften
        @test has_property(ROSENBROCK_FUNCTION, "multimodal")
        @test has_property(ROSENBROCK_FUNCTION, "non-convex")
        @test has_property(ROSENBROCK_FUNCTION, "non-separable")
        @test has_property(ROSENBROCK_FUNCTION, "differentiable")
        @test has_property(ROSENBROCK_FUNCTION, "scalable")
        # Startpunkt und Minimum
        @test ROSENBROCK_FUNCTION.meta[:start] == [0.0, 0.0]
        @test ROSENBROCK_FUNCTION.meta[:min_position] == [1.0, 1.0]  # Korrigiert von 1EPA0
        @test ROSENBROCK_FUNCTION.meta[:min_value] ≈ 0.0
        # Hessian mit Zygote
        H = Zygote.hessian(rosenbrock, [0.5, 0.5])
        @test size(H) == (2, 2)
        @test H[1, 1] ≈ 102.0 atol=1e-6
    end

    @testset "Sphere Tests" begin
        # Funktionswerte (2D)
        @test sphere([0.0, 0.0]) ≈ 0.0
        @test sphere([1.0, 1.0]) ≈ 2.0
        @test sphere([-1.0, -1.0]) ≈ 2.0
        # Funktionswerte (3D)
        x_3d = [1.0, 1.0, 1.0]
        @test sphere(x_3d) ≈ 3.0
        # Gradienten (2D)
        @test sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]
        @test sphere_gradient([0.0, 0.0]) ≈ [0.0, 0.0]
        @test sphere_gradient([-1.0, -1.0]) ≈ [-2.0, -2.0]
        # Gradienten (3D)
        @test length(sphere_gradient(x_3d)) == 3
        @test sphere_gradient(x_3d) ≈ [2.0, 2.0, 2.0]
        # Randfälle
        x_underflow = fill(1e-308, 2)
        @test isfinite(sphere(x_underflow))
        @test all(isfinite, sphere_gradient(x_underflow))
        @test isnan(sphere([NaN, 1.0]))
        @test isinf(sphere([Inf, 1.0]))
        x_large = fill(1e308, 2)
        @test isinf(sphere(x_large))
        # ForwardDiff Kompatibilität
        x_dual = [ForwardDiff.Dual(1.0, 1.0), ForwardDiff.Dual(1.0, 0.0)]
        @test isfinite(sphere(x_dual))
        @test all(isfinite, sphere_gradient(x_dual))
        # Numerische Gradientengenauigkeit
        x = [1.0, 1.0]
        @test sphere_gradient(x) ≈ finite_difference_gradient(sphere, x) atol=1e-6
        x_3d = [1.0, 1.0, 1.0]
        @test sphere_gradient(x_3d) ≈ finite_difference_gradient(sphere, x_3d) atol=1e-6
        # Eigenschaften
        @test has_property(SPHERE_FUNCTION, "unimodal")
        @test has_property(SPHERE_FUNCTION, "convex")
        @test has_property(SPHERE_FUNCTION, "separable")
        @test has_property(SPHERE_FUNCTION, "differentiable")
        @test has_property(SPHERE_FUNCTION, "scalable")
        # Startpunkt und Minimum
        @test SPHERE_FUNCTION.meta[:start] == [0.0, 0.0]
        @test SPHERE_FUNCTION.meta[:min_position] == [0.0, 0.0]
        @test SPHERE_FUNCTION.meta[:min_value] ≈ 0.0
        # Hessian mit Zygote
        H = Zygote.hessian(sphere, [1.0, 1.0])
        @test size(H) == (2, 2)
        @test H ≈ [2.0 0.0; 0.0 2.0] atol=1e-6
    end

    @testset "Metadata Tests" begin
        @test ROSENBROCK_FUNCTION.meta[:name] == "Rosenbrock"
        @test ROSENBROCK_FUNCTION.meta[:lb] == fill(-5.0, 2)
        @test ROSENBROCK_FUNCTION.meta[:ub] == fill(5.0, 2)
        @test SPHERE_FUNCTION.meta[:name] == "Sphere"
        @test SPHERE_FUNCTION.meta[:lb] == fill(-5.12, 2)
        @test SPHERE_FUNCTION.meta[:ub] == fill(5.12, 2)  # Korrigiert von SPHERE_FUNCTION[:ub]
    end

    @testset "Filter and Properties Tests" begin
        @test length(filter_testfunctions(tf -> has_property(tf, "multimodal"))) == 1
        @test length(filter_testfunctions(tf -> has_property(tf, "convex"))) == 1
        @test length(filter_testfunctions(tf -> has_property(tf, "differentiable"))) == 2
        @test has_property(add_property(ROSENBROCK_FUNCTION, "bounded"), "bounded")
    end
end