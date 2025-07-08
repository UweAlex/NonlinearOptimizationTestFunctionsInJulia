module OptimizeAllTestFunctions
using Optim
using ..NonlinearOptimizationTestFunctionsInJulia: TEST_FUNCTIONS, TestFunction

function optimize_all_testfunctions()
    results = Dict{String, Any}()
    for tf in TEST_FUNCTIONS
        try
            result = optimize(tf.f, tf.grad, tf.start, BFGS(), Optim.Options(iterations=100))
            results[tf.name] = (minimizer=Optim.minimizer(result), minimum=Optim.minimum(result))
        catch e
            @error "Fehler bei Optimierung von $(tf.name): $e"
        end
    end
    return results
end

export optimize_all_testfunctions
end