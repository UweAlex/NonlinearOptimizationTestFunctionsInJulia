function optimize_all_testfunctions()
    println("Optimizing all test functions:")
    for tf in TEST_FUNCTIONS
        isempty(tf.start) && error("Startpunkt für $(tf.name) ist leer")
        println("Running optimization for: ", tf.name)
        function grad!(G, x)
            G .= tf.grad(x)
        end
        try
            result = optimize(tf.f, grad!, tf.start, BFGS(), Optim.Options(iterations=1000, show_trace=false))
            println("Minimum found at: ", Optim.minimizer(result))
            println("Objective value: ", Optim.minimum(result))
            println("Convergence: ", Optim.converged(result))
            println("Iterations: ", Optim.iterations(result))
            println()
        catch e
            println("Fehler bei Optimierung von $(tf.name): ", e)
            println()
        end
    end
end