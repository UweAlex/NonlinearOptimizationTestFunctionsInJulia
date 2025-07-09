# examples/List_all_available_test_functions_and_their_properties.js
using NonlinearOptimizationTestFunctionsInJulia

# Demo 1: List all available test functions and their properties
println("Available Test Functions and Their Properties:")
println("-"^50)
for tf in values(TEST_FUNCTIONS)
    println("Function: ", tf.name)
    println("Properties: ", join(sort(collect(tf.properties)), ", "))
    println("-"^50)
end