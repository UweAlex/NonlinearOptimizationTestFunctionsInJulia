# examples/property_statistics.jl
# This script generates statistics about the properties of all test functions 
# in the NonlinearOptimizationTestFunctions package. It counts the occurrences 
# of each property across all functions, sorts them by frequency, and prints 
# a formatted summary. The script is designed to run reliably without any 
# additional dependencies beyond the core package (no Printf, Crayons, etc.).
# It demonstrates how to iterate over the TEST_FUNCTIONS dictionary, use 
# accessors like properties(tf) to retrieve metadata safely, and perform 
# basic data aggregation and sorting in Julia.

# Import the package which provides the TEST_FUNCTIONS dictionary and 
# accessor functions like properties(tf).
using NonlinearOptimizationTestFunctions

# Initialize a dictionary to count the occurrences of each property.
# Properties are case-insensitively normalized to lowercase for consistency.
counter = Dict{String, Int}()

# Iterate over all TestFunction objects in the package's registry.
# TEST_FUNCTIONS is a global dictionary where values are the TestFunction instances.
for tf in values(TEST_FUNCTIONS)
    # Retrieve the list of properties for this function using the accessor.
    # properties(tf) returns a sorted array of strings (e.g., ["continuous", "differentiable"]).
    # If no properties are defined, it defaults to an empty array, avoiding errors.
    props = properties(tf)
    
    # Loop through each property in the list.
    for p in props
        # Normalize the property to lowercase to handle any case variations 
        # (though properties are typically standardized, this ensures robustness).
        key = lowercase(p)
        
        # Increment the count for this property key. If the key doesn't exist, 
        # start from 0.
        counter[key] = get(counter, key, 0) + 1
    end
end

# Sort the collected properties by their count in descending order.
# Convert the dictionary to a list of pairs (prop => count), then sort by the count value.
sorted = sort(collect(counter), by = x -> x[2], rev = true)

# Calculate the total number of test functions for percentage computations.
total = length(TEST_FUNCTIONS)

# Print a header for the statistics output.
# Uses string interpolation for the total count.
println("\n=== PROPERTY STATISTICS ($total Functions) ===\n")

# Print the table header.
println("  Count   | Property")
println("----------+----------------------------------------")

# Enumerate through the sorted list to print each row.
for (i, (prop, count)) in enumerate(sorted)
    # Calculate the percentage of functions with this property.
    # Round to one decimal place for readability.
    percent = round(100 * count / total, digits=1)
    
    # Print the row: left-pad the count to 7 characters for alignment,
    # then the property name and percentage in parentheses.
    println(lpad(count, 7), "  | $prop  ($percent%)  ")
end

# Print the total number of unique properties found.
println("\nTotal unique properties: $(length(sorted))")

# Check if there are any properties at all before accessing the top one.
if !isempty(sorted)
    # Get the most frequent property (first in the sorted list).
    top = first(sorted)
    
    # Print information about the most frequent property.
    println("Most frequent property: \"$(top[1])\" with $(top[2]) occurrences")
end