# test/runtests.jl
# Purpose: Tests for NonlinearOptimizationTestFunctionsInJulia, covering function values, gradients, and metadata.
# Context: Ensures correctness of Rosenbrock, Sphere, and filtering functionality.
# Last modified: 11. Juli 2025, 09:36 AM CEST
using Test
using NonlinearOptimizationTestFunctionsInJulia

@testset "NonlinearOptimizationTestFunctionsInJulia Tests" begin
    @testset "Rosenbrock Tests" begin
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock([1.0, 1.0]) ≈ 0.0
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock([0.5, 0.5]) ≈ 6.5
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock([0.0, 0.0]) ≈ 1.0
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock_gradient([0.5, 0.5]) ≈ [-51.0, 50.0] atol=1e-6
        @test NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.meta[:start] ≈ [0.0, 0.0]
        @test NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.meta[:name] == "Rosenbrock"
        @test NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.meta[:description] == "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2"
        @test NonlinearOptimizationTestFunctionsInJulia.use_testfunction(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, [0.5, 0.5]).f ≈ 6.5
        @test NonlinearOptimizationTestFunctionsInJulia.use_testfunction(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, [0.5, 0.5]).grad ≈ [-51.0, 50.0] atol=1e-6
        @test isnan(NonlinearOptimizationTestFunctionsInJulia.rosenbrock([NaN, 0.5]))
        @test isinf(NonlinearOptimizationTestFunctionsInJulia.rosenbrock([Inf, 0.5]))
        @test isinf(NonlinearOptimizationTestFunctionsInJulia.rosenbrock([-Inf, 0.5]))
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock_gradient([NaN, 0.5]) |> x -> all(isnan.(x))
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock_gradient([Inf, 0.5]) |> x -> all(isinf.(x))
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock_gradient([-Inf, 0.5]) |> x -> all(isinf.(x))
        @test isinf(NonlinearOptimizationTestFunctionsInJulia.rosenbrock([1e308, 1e308]))
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock_gradient([1e308, 1e308]) |> x -> all(isinf.(x))
        x_10d = rand(10)
        @test NonlinearOptimizationTestFunctionsInJulia.rosenbrock(x_10d) isa Float64
        @test length(NonlinearOptimizationTestFunctionsInJulia.rosenbrock_gradient(x_10d)) == 10
        @test_throws AssertionError NonlinearOptimizationTestFunctionsInJulia.rosenbrock([1.0])
        @test_throws AssertionError NonlinearOptimizationTestFunctionsInJulia.use_testfunction(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, Float64[])
        @test NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.meta[:min_position] ≈ [1.0, 1.0]
        @test NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.meta[:min_value] ≈ 0.0
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "multimodal")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "non-convex")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "non-separable")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "differentiable")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "scalable")
        @test !NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "convex")
        @test !NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "separable")
        @test !NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "has_constraints")
        @test :gradient! in fieldnames(TestFunction)
        G = zeros(2)
        NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.gradient!(G, [0.5, 0.5])
        @test G ≈ [-51.0, 50.0] atol=1e-6
        G = zeros(2)
        NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.gradient!(G, [NaN, 0.5])
        @test all(isnan.(G))
        G = zeros(2)
        NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION.gradient!(G, [Inf, 0.5])
        @test all(isinf.(G))
    end

    @testset "Sphere Tests" begin
        @test NonlinearOptimizationTestFunctionsInJulia.sphere([0.0, 0.0]) ≈ 0.0
        @test NonlinearOptimizationTestFunctionsInJulia.sphere([1.0, 1.0]) ≈ 2.0
        @test NonlinearOptimizationTestFunctionsInJulia.sphere([-1.0, -1.0]) ≈ 2.0
        @test NonlinearOptimizationTestFunctionsInJulia.sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]
        @test NonlinearOptimizationTestFunctionsInJulia.sphere_gradient([0.0, 0.0]) ≈ [0.0, 0.0]
        @test NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.meta[:start] ≈ [0.0, 0.0]
        @test NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.meta[:name] == "Sphere"
        @test NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.meta[:description] == "Sphere function: f(x) = Σ x_i^2"
        @test NonlinearOptimizationTestFunctionsInJulia.use_testfunction(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, [1.0, 1.0]).f ≈ 2.0
        @test NonlinearOptimizationTestFunctionsInJulia.use_testfunction(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, [1.0, 1.0]).grad ≈ [2.0, 2.0] atol=1e-6
        @test isnan(NonlinearOptimizationTestFunctionsInJulia.sphere([NaN, 0.5]))
        @test isinf(NonlinearOptimizationTestFunctionsInJulia.sphere([Inf, 0.5]))
        @test isinf(NonlinearOptimizationTestFunctionsInJulia.sphere([-Inf, 0.5]))
        @test NonlinearOptimizationTestFunctionsInJulia.sphere_gradient([NaN, 0.5]) |> x -> all(isnan.(x))
        @test NonlinearOptimizationTestFunctionsInJulia.sphere_gradient([Inf, 0.5]) |> x -> all(isinf.(x))
        @test NonlinearOptimizationTestFunctionsInJulia.sphere_gradient([-Inf, 0.5]) |> x -> all(isinf.(x))
        @test isinf(NonlinearOptimizationTestFunctionsInJulia.sphere([1e308, 1e308]))
        @test NonlinearOptimizationTestFunctionsInJulia.sphere_gradient([1e308, 1e308]) |> x -> all(isinf.(x))
        x_10d = rand(10)
        @test NonlinearOptimizationTestFunctionsInJulia.sphere(x_10d) isa Float64
        @test length(NonlinearOptimizationTestFunctionsInJulia.sphere_gradient(x_10d)) == 10
        @test_throws AssertionError NonlinearOptimizationTestFunctionsInJulia.use_testfunction(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, Float64[])
        @test NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.meta[:min_position] ≈ [0.0, 0.0]
        @test NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.meta[:min_value] ≈ 0.0
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "unimodal")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "convex")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "separable")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "differentiable")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "scalable")
        @test !NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "multimodal")
        @test !NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "non-convex")
        @test !NonlinearOptimizationTestFunctionsInJulia.has_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "has_constraints")
        @test :gradient! in fieldnames(TestFunction)
        G = zeros(2)
        NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.gradient!(G, [1.0, 1.0])
        @test G ≈ [2.0, 2.0] atol=1e-6
        G = zeros(2)
        NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.gradient!(G, [NaN, 0.5])
        @test all(isnan.(G))
        G = zeros(2)
        NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION.gradient!(G, [Inf, 0.5])
        @test all(isinf.(G))
    end

    @testset "Filter Tests" begin
        konvexe_funktionen = NonlinearOptimizationTestFunctionsInJulia.filter_testfunctions(tf -> NonlinearOptimizationTestFunctionsInJulia.has_property(tf, "convex"))
        @test length(konvexe_funktionen) == 1
        @test konvexe_funktionen[1].meta[:name] == "Sphere"
        multimodale_funktionen = NonlinearOptimizationTestFunctionsInJulia.filter_testfunctions(tf -> NonlinearOptimizationTestFunctionsInJulia.has_property(tf, "multimodal"))
        @test length(multimodale_funktionen) == 1
        @test multimodale_funktionen[1].meta[:name] == "Rosenbrock"
    end

    @testset "Properties Manipulation Tests" begin
        new_tf = NonlinearOptimizationTestFunctionsInJulia.add_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "continuous")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(new_tf, "continuous")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(new_tf, "multimodal")
        new_tf2 = NonlinearOptimizationTestFunctionsInJulia.add_property(NonlinearOptimizationTestFunctionsInJulia.SPHERE_FUNCTION, "bounded")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(new_tf2, "bounded")
        @test NonlinearOptimizationTestFunctionsInJulia.has_property(new_tf2, "convex")
        @test_throws AssertionError NonlinearOptimizationTestFunctionsInJulia.add_property(NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION, "invalid")
    end

    @testset "All Functions Properties" begin
        for tf in values(NonlinearOptimizationTestFunctionsInJulia.TEST_FUNCTIONS)
            @test all(p in NonlinearOptimizationTestFunctionsInJulia.VALID_PROPERTIES for p in tf.meta[:properties])
        end
    end
end