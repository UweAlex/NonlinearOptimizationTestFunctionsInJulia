using Test
using NonlinearOptimizationTestFunctionsInJulia

@testset "Rosenbrock Tests" begin
    @test rosenbrock([1.0, 1.0]) ≈ 0.0
    @test rosenbrock([0.5, 0.5]) ≈ 6.5
    @test rosenbrock_gradient([0.5, 0.5]) ≈ [-51.0, 50.0] atol=1e-6
    @test ROSENBROCK_FUNCTION.start ≈ [0.0, 0.0]
    @test ROSENBROCK_FUNCTION.is_convex == false
    @test ROSENBROCK_FUNCTION.is_multimodal == true
    @test ROSENBROCK_FUNCTION.is_differentiable == true
    @test ROSENBROCK_FUNCTION.is_separable == false
    @test ROSENBROCK_FUNCTION.is_scalable == true
    @test ROSENBROCK_FUNCTION.name == "Rosenbrock"
    @test ROSENBROCK_FUNCTION.info[:description] == "Rosenbrock function: f(x) = Σ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2"
    @test use_testfunction(ROSENBROCK_FUNCTION, [0.5, 0.5]).f ≈ 6.5
    @test use_testfunction(ROSENBROCK_FUNCTION, [0.5, 0.5]).grad ≈ [-51.0, 50.0] atol=1e-6
end

@testset "Sphere Tests" begin
    @test sphere([0.0, 0.0]) ≈ 0.0
    @test sphere([1.0, 1.0]) ≈ 2.0
    @test sphere([-1.0, -1.0]) ≈ 2.0
    @test sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]
    @test sphere_gradient([0.0, 0.0]) ≈ [0.0, 0.0]
    @test SPHERE_FUNCTION.start ≈ [0.0, 0.0]
    @test SPHERE_FUNCTION.is_convex == true
    @test SPHERE_FUNCTION.is_multimodal == false
    @test SPHERE_FUNCTION.is_differentiable == true
    @test SPHERE_FUNCTION.is_separable == true
    @test SPHERE_FUNCTION.is_scalable == true
    @test SPHERE_FUNCTION.name == "Sphere"
    @test SPHERE_FUNCTION.info[:description] == "Sphere function: f(x) = Σx_i^2"
    @test use_testfunction(SPHERE_FUNCTION, [1.0, 1.0]).f ≈ 2.0
    @test use_testfunction(SPHERE_FUNCTION, [1.0, 1.0]).grad ≈ [2.0, 2.0] atol=1e-6
end

@testset "Filter Tests" begin
    konvexe_funktionen = filter_testfunctions(TEST_FUNCTIONS, tf -> tf.is_convex)
    @test length(konvexe_funktionen) == 1
    @test konvexe_funktionen[1].name == "Sphere"

    multimodale_funktionen = filter_testfunctions(TEST_FUNCTIONS, tf -> tf.is_multimodal)
    @test length(multimodale_funktionen) == 1
    @test multimodale_funktionen[1].name == "Rosenbrock"
end

@testset "Apply Tests" begin
    # Dummy-Optimierungsmethode, die den Startpunkt zurückgibt
    dummy_method = tf -> tf.start
    # Teste ohne Callback
    apply_to_testfunctions(dummy_method, TEST_FUNCTIONS)
    # Teste mit Callback
    results = []
    apply_to_testfunctions(dummy_method, TEST_FUNCTIONS, callback=(tf, res) -> push!(results, (tf.name, res)))
    @test length(results) == 2
    @test results[1] == ("Rosenbrock", [0.0, 0.0])
    @test results[2] == ("Sphere", [0.0, 0.0])
end
