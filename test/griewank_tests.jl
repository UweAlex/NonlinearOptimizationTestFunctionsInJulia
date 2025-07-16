# test/griewank_tests.jl
# Purpose: Tests for the Griewank function in NonlinearOptimizationTestFunctionsInJulia.
# Context: Verifies function values, gradients, and optimization.
# Last modified: 16. Juli 2025, 14:48 PM CEST

using Test, Optim, ForwardDiff, LinearAlgebra
using NonlinearOptimizationTestFunctionsInJulia: GRIEWANK_FUNCTION, griewank, griewank_gradient

@testset "Griewank Tests" begin
    tf = GRIEWANK_FUNCTION
    @test_throws ArgumentError griewank(Float64[])
    @test isnan(griewank([NaN]))
    @test isinf(griewank([Inf]))
    @test isfinite(griewank([1e-308]))
    @test griewank([0.0]) ≈ 0.0 atol=1e-6
    @test griewank([0.0, 0.0]) ≈ 0.0 atol=1e-6
    @test griewank([1.0, 1.0]) ≈ 0.5899802569229068 atol=2.5e-4
    @test griewank_gradient([0.0]) ≈ [0.0] atol=1e-6
    @test griewank_gradient([1.0, 1.0]) ≈ [0.34614348486240547, 0.18918869330539688] atol=1e-5
    x = tf.meta[:start](2)
    G = zeros(length(x))
    tf.gradient!(G, x)
    @test G ≈ griewank_gradient(x) atol=1e-6
    @test tf.meta[:name] == "griewank"
    @test tf.meta[:start](1) == [1.0]
    @test tf.meta[:min_position](1) == [0.0]
    @test tf.meta[:min_value] ≈ 0.0 atol=1e-6
    @test tf.meta[:lb](1) == [-5.0]
    @test tf.meta[:ub](1) == [5.0]
    @test tf.meta[:lb](2, bounds="benchmark") == [-600.0, -600.0]
    @test tf.meta[:ub](2, bounds="benchmark") == [600.0, 600.0]
    @testset "High-Precision Validation" begin
        setprecision(BigFloat, 256)
        @test griewank(BigFloat.([1.0, 1.0])) ≈ BigFloat("0.5899802569229068") atol=2.5e-4
    end
    @testset "Optimization Tests" begin
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
    end
    @testset "ForwardDiff Compatibility" begin
        x_dual = [ForwardDiff.Dual(0.5, 1.0), ForwardDiff.Dual(0.5, 0.0)]
        @test isfinite(griewank(x_dual))
        @test all(isfinite, griewank_gradient(x_dual))
    end
end