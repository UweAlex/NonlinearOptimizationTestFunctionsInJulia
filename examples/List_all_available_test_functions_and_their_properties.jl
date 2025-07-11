# Path: examples/List_all_available_test_functions_and_their_properties.jl
# Purpose: Lists all test functions with their start points, minima, and properties.
# Context: Part of NonlinearOptimizationTestFunctionsInJulia, enabling inspection of function characteristics.
# Notes: Minimal output for clarity, no external dependencies required.
# Last modified: 11. Juli 2025, 09:13 AM CEST
using NonlinearOptimizationTestFunctionsInJulia
for tf in values(NonlinearOptimizationTestFunctionsInJulia.TEST_FUNCTIONS)
    println("$(tf.meta[:name]): Start at $(tf.meta[:start]), Minimum at $(tf.meta[:min_position]), Value $(tf.meta[:min_value]), Properties: $(join(tf.meta[:properties], ", "))")
end