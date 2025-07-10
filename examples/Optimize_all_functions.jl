# examples/Optimize_all_functions.jl
using NonlinearOptimizationTestFunctionsInJulia
using Optim

# Optimizes all test functions using BFGS and returns results
function optimize_all_testfunctions()
    results = Dict{String, NamedTuple}()
    for tf in values(NonlinearOptimizationTestFunctionsInJulia.TEST_FUNCTIONS)
        try
            result = optimize(
                tf.f,
                tf.gradient!,  # Nutzt das gradient!-Feld der TestFunction-Instanz
                tf.start,
                BFGS(),
                Optim.Options(show_trace=false, iterations=10000)
            )
            results[tf.name] = (minimizer=Optim.minimizer(result), minimum=Optim.minimum(result))
        catch e
            @error "Error optimizing $(tf.name): $e"
        end
    end
    return results
end

# Run the optimization and print results
println("Optimizing All Test Functions with BFGS:")
println("-"^50)
results = optimize_all_testfunctions()
for (name, result) in results
    println("Function: ", name)
    println("  Minimizer: ", result.minimizer)
    println("  Minimum Value: ", result.minimum)
    println("-"^50)
end