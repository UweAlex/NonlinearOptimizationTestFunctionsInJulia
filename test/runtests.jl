using Test
using NonlinearOptimizationTestFunctionsInJulia

@testset "Rosenbrock Tests" begin
    @test rosenbrock([1.0, 1.0]) ≈ 0.0
    @test rosenbrock([0.0, 0.0]) ≈ 1.0
    @test rosenbrock([-1.2, 1.0]) ≈ 24.2 atol=1e-6
    @test rosenbrock([0.5, 0.5]) ≈ 6.5
end