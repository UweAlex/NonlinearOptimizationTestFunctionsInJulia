# examples/demo_dimension_complexity.jl
# This demo script illustrates how the optimization complexity increases with dimension 
# for scalable test functions in the NonlinearOptimizationTestFunctions package. 
# It uses the Rosenbrock function (a classic scalable, ill-conditioned problem) and 
# optimizes it with L-BFGS for increasing dimensions. The script measures the average 
# number of function + gradient evaluations over multiple runs as a proxy for complexity.
# 
# Why Rosenbrock? It's unimodal but has a narrow, curved valley that becomes harder 
# to navigate in higher dimensions due to ill-conditioning (high condition number of Hessian).
# 
# The demo runs quickly by limiting dimensions (up to 64) and using analytical gradients 
# for fast convergence. Extend dimensions if needed, but higher n will take longer.
# 
# Requirements: Install Optimization.jl and OptimizationOptimJL.jl if not present 
# (via using Pkg; Pkg.add(["Optimization", "OptimizationOptimJL"])).
# 
# Expected output: Printed table showing evaluations growing with n, e.g., 
# from ~50 for n=2 to several hundred for n=64, demonstrating the "curse of dimensionality".
# 
# Last modified: December 21, 2025.

# Import the core package for test functions and utilities (fixed, optimization_problem, etc.).
using NonlinearOptimizationTestFunctions

# Import Optimization.jl for a unified solver interface and OptimJL backend for LBFGS.
using Optimization, OptimizationOptimJL

# Import Statistics for averaging over runs.
using Statistics

# Select a scalable function: Rosenbrock (valley-shaped, ill-conditioned in high dim).
tf_orig = ROSENBROCK_FUNCTION

# Print introductory message.
println("Demo: Increasing optimization complexity with dimension for scalable functions")
println("Function: ", name(tf_orig), " (scalable, unimodal but ill-conditioned)")
println("Optimizer: L-BFGS with analytical gradients")
println("Measure: Average function + gradient evaluations over 3 runs\n")

# Define dimensions to test: powers of 2, kept small for quick runtime (~seconds total).
dimensions = [2, 4, 8, 16, 32, 64]

# Dictionary to store average evaluations per dimension.
results = Dict{Int, Float64}()

# Loop over dimensions.
for n in dimensions
    # Convert the scalable function to a fixed-dimension instance.
    # fixed(tf_orig; n=n) binds metadata (start, min_position, etc.) to this n.
    tf = fixed(tf_orig; n=n)
    
    # Collect evaluations over 3 independent runs for averaging (reduces noise).
    evals = Float64[]
    for _ in 1:3
        # Reset counters before each optimization.
        reset_counts!(tf)
        
        # Solve using L-BFGS via the unified interface.
        # optimization_problem(tf) provides f, grad, start point, etc.
        sol = solve(
            optimization_problem(tf),
            LBFGS();
            maxiters=10000,    # Sufficient for convergence in these dimensions
            reltol=1e-6        # Reasonable tolerance
        )
        
        # Total evaluations: function calls + gradient calls.
        total_calls = get_f_count(tf) + get_grad_count(tf)
        push!(evals, total_calls)
    end
    
    # Compute and store average.
    avg_calls = mean(evals)
    results[n] = avg_calls
    
    # Print result for this dimension.
    println("n = $n: Average evaluations = ", round(avg_calls, digits=1))
end

# Summary message highlighting the trend.
println("\nSummary: As dimension (n) increases, the number of evaluations grows due to ")
println("the 'curse of dimensionality' and worsening conditioning. For Rosenbrock, ")
println("this manifests as slower convergence in the elongated valley. Try higher n ")
println("for more pronounced effects, but note longer runtimes.")