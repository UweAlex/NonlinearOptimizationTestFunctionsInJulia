using Test
using NonlinearOptimizationTestFunctionsInJulia

@testset "NonlinearOptimizationTestFunctionsInJulia Tests" begin
    @testset "Rosenbrock Tests" begin
        @test rosenbrock([1.0, 1.0]) ≈ 0.0
        @test rosenbrock([0.5, 0.5]) ≈ 6.5
        @test rosenbrock([0.0, 0.0]) ≈ 1.0
        @test rosenbrock_gradient([0.5, 0.5]) ≈ [-51.0, 50.0] atol=1e-6
        @test ROSENBROCK_FUNCTION.start ≈ [0.0, 0.0]
        @test ROSENBROCK_FUNCTION.name == "Rosenbrock"
        @test ROSENBROCK_FUNCTION.info[:description] == "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2"
        @test use_testfunction(ROSENBROCK_FUNCTION, [0.5, 0.5]).f ≈ 6.5
        @test use_testfunction(ROSENBROCK_FUNCTION, [0.5, 0.5]).grad ≈ [-51.0, 50.0] atol=1e-6
        @test isnan(rosenbrock([NaN, 0.5]))
        @test isinf(rosenbrock([Inf, 0.5]))
        @test isinf(rosenbrock([-Inf, 0.5]))
        @test rosenbrock_gradient([NaN, 0.5]) |> x -> all(isnan.(x))
        @test rosenbrock_gradient([Inf, 0.5]) |> x -> all(isinf.(x))
        @test rosenbrock_gradient([-Inf, 0.5]) |> x -> all(isinf.(x))
        @test isinf(rosenbrock([1e308, 1e308]))
        @test rosenbrock_gradient([1e308, 1e308]) |> x -> all(isinf.(x))
        x_10d = rand(10)
        @test rosenbrock(x_10d) isa Float64
        @test length(rosenbrock_gradient(x_10d)) == 10
        @test_throws AssertionError rosenbrock([1.0])
        @test_throws AssertionError use_testfunction(ROSENBROCK_FUNCTION, Float64[])
        @test ROSENBROCK_FUNCTION.min_position ≈ [1.0, 1.0]
        @test ROSENBROCK_FUNCTION.min_value ≈ 0.0
        @test has_property(ROSENBROCK_FUNCTION, "multimodal")
        @test has_property(ROSENBROCK_FUNCTION, "non-convex")
        @test has_property(ROSENBROCK_FUNCTION, "non-separable")
        @test has_property(ROSENBROCK_FUNCTION, "differentiable")
        @test has_property(ROSENBROCK_FUNCTION, "scalable")
        @test !has_property(ROSENBROCK_FUNCTION, "convex")
        @test !has_property(ROSENBROCK_FUNCTION, "separable")
        @test !has_property(ROSENBROCK_FUNCTION, "has_constraints")
    end

    @testset "Sphere Tests" begin
        @test sphere([0.0, 0.0]) ≈ 0.0
        @test sphere([1.0, 1.0]) ≈ 2.0
        @test sphere([-1.0, -1.0]) ≈ 2.0
        @test sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]
        @test sphere_gradient([0.0, 0.0]) ≈ [0.0, 0.0]
        @test SPHERE_FUNCTION.start ≈ [0.0, 0.0]
        @test SPHERE_FUNCTION.name == "Sphere"
        @test SPHERE_FUNCTION.info[:description] == "Sphere function: f(x) = Σ x_i^2"
        @test use_testfunction(SPHERE_FUNCTION, [1.0, 1.0]).f ≈ 2.0
        @test use_testfunction(SPHERE_FUNCTION, [1.0, 1.0]).grad ≈ [2.0, 2.0] atol=1e-6
        @test isnan(sphere([NaN, 0.5]))
        @test isinf(sphere([Inf, 0.5]))
        @test isinf(sphere([-Inf, 0.5]))
        @test sphere_gradient([NaN, 0.5]) |> x -> all(isnan.(x))
        @test sphere_gradient([Inf, 0.5]) |> x -> all(isinf.(x))
        @test sphere_gradient([-Inf, 0.5]) |> x -> all(isinf.(x))
        @test isinf(sphere([1e308, 1e308]))
        @test sphere_gradient([1e308, 1e308]) |> x -> all(isinf.(x))
        x_10d = rand(10)
        @test sphere(x_10d) isa Float64
        @test length(sphere_gradient(x_10d)) == 10
        @test_throws AssertionError use_testfunction(SPHERE_FUNCTION, Float64[])
        @test SPHERE_FUNCTION.min_position ≈ [0.0, 0.0]
        @test SPHERE_FUNCTION.min_value ≈ 0.0
        @test has_property(SPHERE_FUNCTION, "unimodal")
        @test has_property(SPHERE_FUNCTION, "convex")
        @test has_property(SPHERE_FUNCTION, "separable")
        @test has_property(SPHERE_FUNCTION, "differentiable")
        @test has_property(SPHERE_FUNCTION, "scalable")
        @test !has_property(SPHERE_FUNCTION, "multimodal")
        @test !has_property(SPHERE_FUNCTION, "non-convex")
        @test !has_property(SPHERE_FUNCTION, "has_constraints")
    end

    @testset "Filter Tests" begin
        konvexe_funktionen = filter_testfunctions(tf -> has_property(tf, "convex"))
        @test length(konvexe_funktionen) == 1
        @test konvexe_funktionen[1].name == "Sphere"
        multimodale_funktionen = filter_testfunctions(tf -> has_property(tf, "multimodal"))
        @test length(multimodale_funktionen) == 1
        @test multimodale_funktionen[1].name == "Rosenbrock"
    end

    @testset "Properties Manipulation Tests" begin
        new_tf = add_property(ROSENBROCK_FUNCTION, "continuous")
        @test has_property(new_tf, "continuous")
        @test has_property(new_tf, "multimodal")
        new_tf2 = add_property(SPHERE_FUNCTION, "bounded")
        @test has_property(new_tf2, "bounded")
        @test has_property(new_tf2, "convex")
        @test_throws AssertionError add_property(ROSENBROCK_FUNCTION, "invalid")
    end

    @testset "All Functions Properties" begin
        for tf in values(TEST_FUNCTIONS)
            @test all(p in NonlinearOptimizationTestFunctionsInJulia.VALID_PROPERTIES for p in tf.properties)
        end
    end
end