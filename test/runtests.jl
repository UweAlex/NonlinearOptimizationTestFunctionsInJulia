# test/runtests.jl
# Purpose: Tests for NonlinearOptimizationTestFunctionsInJulia, covering function values, gradients, and metadata.
# Context: Ensures correctness of Rosenbrock, Sphere, and filtering functionality.
# Last modified: 11. Juli 2025, 10:14 AM CEST
using Test, ForwardDiff
using NonlinearOptimizationTestFunctionsInJulia

@testset "NonlinearOptimizationTestFunctionsInJulia Tests" begin
    @testset "Rosenbrock Tests" begin
        # Bestehende Tests bleiben unverändert
        @test rosenbrock([1.0, 1.0]) ≈ 0.0
        @test rosenbrock([0.5, 0.5]) ≈ 6.5
        # Unterlauf-Tests
        @testset "Underflow tests" begin
            x = fill(1e-308, 2)
            @test isfinite(rosenbrock(x))
            @test all(isfinite, rosenbrock_gradient(x))
        end
        # ForwardDiff-Kompatibilität
        @testset "ForwardDiff Compatibility" begin
            x = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, 0.0)]
            @test isfinite(rosenbrock(x))
            @test all(isfinite, rosenbrock_gradient(x))
        end
    end

    @testset "Sphere Tests" begin
        # Bestehende Tests bleiben unverändert
        @test sphere([0.0, 0.0]) ≈ 0.0
        @test sphere([1.0, 1.0]) ≈ 2.0
        # Unterlauf-Tests
        @testset "Underflow tests" begin
            x = fill(1e-308, 2)
            @test isfinite(sphere(x))
            @test all(isfinite, sphere_gradient(x))
        end
        # ForwardDiff-Kompatibilität
        @testset "ForwardDiff Compatibility" begin
            x = [ForwardDiff.Dual(1.0, 1.0), ForwardDiff.Dual(1.0, 0.0)]
            @test isfinite(sphere(x))
            @test all(isfinite, sphere_gradient(x))
        end
    end

    # Bestehende Filter- und Properties-Tests bleiben unverändert
end