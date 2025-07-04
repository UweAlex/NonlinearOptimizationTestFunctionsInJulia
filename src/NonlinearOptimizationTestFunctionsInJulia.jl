module NonlinearOptimizationTestFunctionsInJulia
export rosenbrock
function rosenbrock(x::Vector)
    sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:length(x)-1)
end
end