# test/runtests.jl
# Purpose: Entry point for running all tests in NonlinearOptimizationTestFunctionsInJulia.
# Context: Contains cross-function tests and includes function-specific tests via include_testfiles.jl.
# Last modified: 16. Juli 2025, 13:21 PM CEST

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

@testset "NonlinearOptimizationTestFunctionsInJulia Cross-Function Tests" begin
    @testset "Metadata Tests" begin
        @test ROSENBROCK_FUNCTION.meta[:name] == "rosenbrock"
        @test ROSENBROCK_FUNCTION.meta[:lb](2) == fill(-5.0, 2)
        @test ROSENBROCK_FUNCTION.meta[:ub](2) == fill(5.0, 2)
        @test SPHERE_FUNCTION.meta[:name] == "sphere"
        @test SPHERE_FUNCTION.meta[:lb](2) == fill(-5.12, 2)
        @test SPHERE_FUNCTION.meta[:ub](2) == fill(5.12, 2)
        @test ACKLEY_FUNCTION.meta[:name] == "ackley"
        @test ACKLEY_FUNCTION.meta[:lb](2) == fill(-5.0, 2)
        @test ACKLEY_FUNCTION.meta[:ub](2) == fill(5.0, 2)
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:name] == "axisparallelhyperellipsoid"
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:lb](2) == fill(-Inf, 2)
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:ub](2) == fill(Inf, 2)
        @test RASTRIGIN_FUNCTION.meta[:name] == "rastrigin"
        @test RASTRIGIN_FUNCTION.meta[:lb](2) == fill(-5.12, 2)
        @test RASTRIGIN_FUNCTION.meta[:ub](2) == fill(5.12, 2)
        @test GRIEWANK_FUNCTION.meta[:name] == "griewank"
        @test GRIEWANK_FUNCTION.meta[:lb](2) == fill(-5.0, 2)
        @test GRIEWANK_FUNCTION.meta[:ub](2) == fill(5.0, 2)
    end

    @testset "Scalable Metadata Tests" begin
        @test ROSENBROCK_FUNCTION.meta[:lb](3) == fill(-5.0, 3)
        @test ROSENBROCK_FUNCTION.meta[:ub](3) == fill(5.0, 3)
        @test ROSENBROCK_FUNCTION.meta[:start](3) == fill(0.0, 3)
        @test ROSENBROCK_FUNCTION.meta[:min_position](3) == fill(1.0, 3)
        @test SPHERE_FUNCTION.meta[:lb](3) == fill(-5.12, 3)
        @test SPHERE_FUNCTION.meta[:ub](3) == fill(5.12, 3)
        @test SPHERE_FUNCTION.meta[:start](3) == fill(0.0, 3)
        @test SPHERE_FUNCTION.meta[:min_position](3) == fill(0.0, 3)
        @test ACKLEY_FUNCTION.meta[:lb](3) == fill(-5.0, 3)
        @test ACKLEY_FUNCTION.meta[:ub](3) == fill(5.0, 3)
        @test ACKLEY_FUNCTION.meta[:start](3) == fill(1.0, 3)
        @test ACKLEY_FUNCTION.meta[:min_position](3) == fill(0.0, 3)
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:lb](3) == fill(-Inf, 3)
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:ub](3) == fill(Inf, 3)
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:start](3) == fill(1.0, 3)
        @test AXISPARALLELHYPERELLIPSOID_FUNCTION.meta[:min_position](3) == fill(0.0, 3)
        @test RASTRIGIN_FUNCTION.meta[:lb](3) == fill(-5.12, 3)
        @test RASTRIGIN_FUNCTION.meta[:ub](3) == fill(5.12, 3)
        @test RASTRIGIN_FUNCTION.meta[:start](3) == fill(1.0, 3)
        @test RASTRIGIN_FUNCTION.meta[:min_position](3) == fill(0.0, 3)
        @test GRIEWANK_FUNCTION.meta[:lb](3) == fill(-5.0, 3)
        @test GRIEWANK_FUNCTION.meta[:ub](3) == fill(5.0, 3)
        @test GRIEWANK_FUNCTION.meta[:start](3) == fill(1.0, 3)
        @test GRIEWANK_FUNCTION.meta[:min_position](3) == fill(0.0, 3)
    end

    @testset "Filter and Properties Tests" begin
        @test length(filter_testfunctions(tf -> has_property(tf, "multimodal"))) == 4
        @test length(filter_testfunctions(tf -> has_property(tf, "convex"))) == 2
        @test length(filter_testfunctions(tf -> has_property(tf, "differentiable"))) == 6
        @test has_property(add_property(ROSENBROCK_FUNCTION, "bounded"), "bounded")
    end
end

include("include_testfiles.jl")