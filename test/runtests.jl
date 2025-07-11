# test/runtests.jl
# Purpose: Tests for NonlinearOptimizationTestFunctionsInJulia, covering function values, gradients, and metadata.
# Context: Ensures correctness of Rosenbrock, Sphere, and filtering functionality.
# Last modified: 11. Juli 2025, 10:23 AM CEST
using Test, ForwardDiff
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
        @test rosenbrock([1.0, 1.0]) ≈ 0.0
        @test rosenbrock([0.5, 0.5]) ≈ 6.5
        @test rosenbrock_gradient([1.0, 1.0]) ≈ [0.0, 0.0]
        @testset "Underflow tests" begin
            x = fill(1e-308, 2)
            @test isfinite(rosenbrock(x))
            @test all(isfinite, rosenbrock_gradient(x))
        end
        @testset "ForwardDiff Compatibility" begin
            x = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, 0.0)]
            @test isfinite(rosenbrock(x))
            @test all(isfinite, rosenbrock_gradient(x))
        end
        @testset "Numerical Gradient Accuracy" begin
            x = [0.5, 0.5]
            @test rosenbrock_gradient(x) ≈ finite_difference_gradient(rosenbrock, x) atol=1e-6
        end
    end

    @testset "Sphere Tests" begin
        @test sphere([0.0, 0.0]) ≈ 0.0
        @test sphere([1.0, 1.0]) ≈ 2.0
        @test sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]
        @testset "Underflow tests" begin
            x = fill(1e-308, 2)
            @test isfinite(sphere(x))
            @test all(isfinite, sphere_gradient(x))
        end
        @testset "ForwardDiff Compatibility" begin
            x = [ForwardDiff.Dual(1.0, 1.0), ForwardDiff.Dual(1.0, 0.0)]
            @test isfinite(sphere(x))
            @test all(isfinite, sphere_gradient(x))
        end
        @testset "Numerical Gradient Accuracy" begin
            x = [1.0, 1.0]
            @test sphere_gradient(x) ≈ finite_difference_gradient(sphere, x) atol=1e-6
        end
    end

    @testset "Metadata Tests" begin
        @test ROSENBROCK_FUNCTION.meta[:name] == "Rosenbrock"
        @test ROSENBROCK_FUNCTION.meta[:lb] == fill(-5.0, 2)
        @test SPHERE_FUNCTION.meta[:name] == "Sphere"
        @test SPHERE_FUNCTION.meta[:ub] == fill(5.12, 2)
    end

    @testset "Filter and Properties Tests" begin
        @test length(filter_testfunctions(tf -> has_property(tf, "multimodal"))) == 1
        @test length(filter_testfunctions(tf -> has_property(tf, "convex"))) == 1
        tf = add_property(ROSENBROCK_FUNCTION, "bounded")
        @test has_property(tf, "bounded")
    end
end