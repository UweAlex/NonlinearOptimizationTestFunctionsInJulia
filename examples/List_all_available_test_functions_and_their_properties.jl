# Path: examples/List_all_available_test_functions_and_their_properties.jl
# Purpose: Lists all test functions (Rosenbrock, Sphere) with their start points, minima, and properties to show easy access to TestFunction metadata.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, enabling users to inspect function characteristics for optimization tasks.
# Notes: Minimal output for clarity and natural usage, as described in Readme.txt. No external dependencies required beyond the package itself.

using NonlinearOptimizationTestFunctionsInJulia
for tf in values(NonlinearOptimizationTestFunctionsInJulia.TEST_FUNCTIONS)
    println("$(tf.name): Start at $(tf.start), Minimum at $(tf.min_position), Value $(tf.min_value), Properties: $(join(tf.properties, ", "))")
end