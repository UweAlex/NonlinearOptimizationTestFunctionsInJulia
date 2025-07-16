# test/test_project.jl
# Purpose: Manual tests for NonlinearOptimizationTestFunctionsInJulia to verify function values, gradients, Hessians, and metadata.
# Context: Ensures correctness of all test functions.
# Last modified: 16. Juli 2025, 14:22 PM CEST

using Test, ForwardDiff, Zygote, LinearAlgebra
using NonlinearOptimizationTestFunctionsInJulia
using Optim

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

@testset "Manual Tests for NonlinearOptimizationTestFunctionsInJulia" begin
    @testset "Rosenbrock Manual Tests" begin
        @test rosenbrock([1.0, 1.0]) ≈ 0.0
        @test rosenbrock([0.5, 0.5]) ≈ 6.5
        @test rosenbrock_gradient([1.0, 1.0]) ≈ [0.0, 0.0]
        @test rosenbrock_gradient([0.5, 0.5])[1] ≈ -51.0 atol=1e-6
        @test rosenbrock_gradient([0.5, 0.5])[2] ≈ 50.0 atol=1e-6
        @test rosenbrock_gradient([0.5, 0.5]) ≈ finite_difference_gradient(rosenbrock, [0.5, 0.5]) atol=1e-6
        @test has_property(ROSENBROCK_FUNCTION, "differentiable")
        @test ROSENBROCK_FUNCTION.meta[:min_position](2) == [1.0, 1.0]
    end
    @testset "Sphere Manual Tests" begin
        @test sphere([0.0, 0.0]) ≈ 0.0
        @test sphere([1.0, 1.0]) ≈ 2.0
        @test sphere_gradient([1.0, 1.0]) ≈ [2.0, 2.0]
        @test sphere_gradient([0.0, 0.0]) ≈ [0.0, 0.0]
        @test sphere_gradient([1.0, 1.0]) ≈ finite_difference_gradient(sphere, [1.0, 1.0]) atol=1e-6
        @test has_property(SPHERE_FUNCTION, "convex")
        @test SPHERE_FUNCTION.meta[:min_position](2) == [0.0, 0.0]
    end
    @testset "Ackley Manual Tests" begin
        @test ackley([0.0, 0.0]) ≈ 0.0 atol=1e-6
        @test ackley([1.0, 1.0]) ≈ 3.6253849384403627 atol=1e-6
        @test ackley_gradient([0.0, 0.0]) ≈ [0.0, 0.0] atol=1e-6
        @test ackley_gradient([1.0, 1.0]) ≈ finite_difference_gradient(ackley, [1.0, 1.0]) atol=1e-6
        @test has_property(ACKLEY_FUNCTION, "differentiable")
        @test ACKLEY_FUNCTION.meta[:min_position](2) == [0.0, 0.0]
    end
    @testset "AxisParallelHyperEllipsoid Manual Tests" begin
        @test axisparallelhyperellipsoid([0.0, 0.0]) ≈ 0.0
        @test axisparallelhyperellipsoid([1.0, 1.0]) ≈ 3.0
        @test axisparallelhyperellipsoid_gradient([0.0, 0.0]) ≈ [0.0, 0.0]
        @test axisparallelhyperellipsoid_gradient([1.0, 1.0]) ≈ [2.0, 4.0]
        @test axisparallelhyperellipsoid_gradient([1.0, 1.0]) ≈ finite_difference_gradient(axisparallelhyperellipsoid, [1.0, 1.0]) atol=1e-6
        @test has_property(AXISPARALLELHYPERELLIPSOID_FUNCTION, "convex")
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:min_position](2) == [0.0, 0.0]
    end
    @testset "Rastrigin Manual Tests" begin
        @test rastrigin([0.0, 0.0]) ≈ 0.0 atol=1e-6
        @test rastrigin([1.0, 1.0]) ≈ 2.0 atol=1e-6
        @test rastrigin_gradient([0.0, 0.0]) ≈ [0.0, 0.0] atol=1e-6
        @test rastrigin_gradient([1.0, 1.0]) ≈ [2.0, 2.0] atol=1e-6
        @test rastrigin_gradient([1.0, 1.0]) ≈ finite_difference_gradient(rastrigin, [1.0, 1.0]) atol=1e-6
        @test has_property(RASTRIGIN_FUNCTION, "differentiable")
        @test RASTRIGIN_FUNCTION.meta[:min_position](2) == [0.0, 0.0]
    end
    @testset "Griewank Manual Tests" begin
        @test griewank([0.0, 0.0]) ≈ 0.0 atol=1e-6
        @test griewank([1.0, 1.0]) ≈ 0.5899802569229068 atol=2.5e-4
        @test griewank_gradient([0.0, 0.0]) ≈ [0.0, 0.0] atol=1e-6
        @test griewank_gradient([1.0, 1.0]) ≈ [0.34614348486240547, 0.18918869330539688] atol=1e-5
        @test has_property(GRIEWANK_FUNCTION, "differentiable")
        @test GRIEWANK_FUNCTION.meta[:min_position](2) == [0.0, 0.0]
        @test GRIEWANK_FUNCTION.meta[:name] == "griewank"
    end
    @testset "Edge Cases" begin
        @test isnan(rosenbrock([NaN, 1.0]))
        @test isinf(rosenbrock([Inf, 1.0]))
        @test isfinite(rosenbrock([1e-308, 1e-308]))
        @test isnan(sphere([NaN, 1.0]))
        @test isinf(sphere([Inf, 1.0]))
        @test isfinite(sphere([1e-308, 1e-308]))
        @test isnan(ackley([NaN]))
        @test isinf(ackley([Inf]))
        @test isfinite(ackley([1e-308]))
        @test isnan(axisparallelhyperellipsoid([NaN]))
        @test isinf(axisparallelhyperellipsoid([Inf]))
        @test isfinite(axisparallelhyperellipsoid([1e-308]))
        @test isnan(rastrigin([NaN]))
        @test isinf(rastrigin([Inf]))
        @test isfinite(rastrigin([1e-308]))
        @test isnan(griewank([NaN]))
        @test isinf(griewank([Inf]))
        @test isfinite(griewank([1e-308]))
    end
    @testset "ForwardDiff Compatibility" begin
        x_dual = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, 0.0)]
        @test isfinite(rosenbrock(x_dual))
        @test all(isfinite, rosenbrock_gradient(x_dual))
        @test isfinite(sphere(x_dual))
        @test all(isfinite, sphere_gradient(x_dual))
        @test isfinite(ackley(x_dual))
        @test all(isfinite, ackley_gradient(x_dual))
        @test isfinite(axisparallelhyperellipsoid(x_dual))
        @test all(isfinite, axisparallelhyperellipsoid_gradient(x_dual))
        @test isfinite(rastrigin(x_dual))
        @test all(isfinite, rastrigin_gradient(x_dual))
        @test isfinite(griewank(x_dual))
        @test all(isfinite, griewank_gradient(x_dual))
    end
    @testset "Zygote Hessian" begin
        H_rosenbrock = Zygote.hessian(rosenbrock, [0.5, 0.5])
        @test size(H_rosenbrock) == (2, 2)
        @test H_rosenbrock[1, 1] ≈ 102.0 atol=1e-6
        H_sphere = Zygote.hessian(sphere, [1.0, 1.0])
        @test H_sphere ≈ [2.0 0.0; 0.0 2.0] atol=1e-6
    end
    @testset "Metadata and Filter Tests" begin
        @test ROSENBROCK_FUNCTION.meta[:name] == "rosenbrock"
        @test SPHERE_FUNCTION.meta[:name] == "sphere"
        @test ACKLEY_FUNCTION.meta[:name] == "ackley"
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:name] == "axisparallelhyperellipsoid"
        @test RASTRIGIN_FUNCTION.meta[:name] == "rastrigin"
        @test GRIEWANK_FUNCTION.meta[:name] == "griewank"
        @test length(filter_testfunctions(tf -> has_property(tf, "convex"))) == 2
        @test length(filter_testfunctions(tf -> has_property(tf, "differentiable"))) == 6
    end
    @testset "Optimization Tests" begin
        for tf in values(TEST_FUNCTIONS)
            if tf.meta[:name] in ["ackley", "griewank"]
                start_points = [tf.meta[:start](2), zeros(2), rand(2) .* 2 .- 1]
                best_result = nothing
                best_f = Inf
                for start in start_points
                    result = optimize(tf.f, tf.gradient!, start, LBFGS(), Optim.Options(f_reltol=1e-6))
                    if Optim.minimum(result) < best_f
                        best_f = Optim.minimum(result)
                        best_result = result
                    end
                end
                @test Optim.minimum(best_result) < 1e-5
                @test Optim.minimizer(best_result) ≈ tf.meta[:min_position](2) atol=1e-3
            else
                result = optimize(tf.f, tf.gradient!, tf.meta[:start](2), LBFGS(), Optim.Options(f_reltol=1e-6))
                @test Optim.minimum(result) < 1e-5
                @test Optim.minimizer(result) ≈ tf.meta[:min_position](2) atol=1e-3
            end
        end
        result_n10 = optimize(ROSENBROCK_FUNCTION.f, ROSENBROCK_FUNCTION.gradient!, ROSENBROCK_FUNCTION.meta[:start](10), LBFGS(), Optim.Options(f_reltol=1e-6))
        @test Optim.minimum(result_n10) < 1e-5
        @test Optim.minimizer(result_n10) ≈ ROSENBROCK_FUNCTION.meta[:min_position](10) atol=1e-3
        result_n100 = optimize(ROSENBROCK_FUNCTION.f, ROSENBROCK_FUNCTION.gradient!, ROSENBROCK_FUNCTION.meta[:start](100), LBFGS(), Optim.Options(f_reltol=1e-6))
        @test Optim.minimum(result_n100) < 1e-5
        @test Optim.minimizer(result_n100) ≈ ROSENBROCK_FUNCTION.meta[:min_position](100) atol=1e-3
        result_n10_sphere = optimize(SPHERE_FUNCTION.f, SPHERE_FUNCTION.gradient!, SPHERE_FUNCTION.meta[:start](10), LBFGS(), Optim.Options(f_reltol=1e-6))
        @test Optim.minimum(result_n10_sphere) < 1e-5
        @test Optim.minimizer(result_n10_sphere) ≈ SPHERE_FUNCTION.meta[:min_position](10) atol=1e-3
        result_n100_sphere = optimize(SPHERE_FUNCTION.f, SPHERE_FUNCTION.gradient!, SPHERE_FUNCTION.meta[:start](100), LBFGS(), Optim.Options(f_reltol=1e-6))
        @test Optim.minimum(result_n100_sphere) < 1e-5
        @test Optim.minimizer(result_n100_sphere) ≈ SPHERE_FUNCTION.meta[:min_position](100) atol=1e-3
    end
end