# Test Functions

## Overview
The package includes test functions accessible via TEST_FUNCTIONS, designed for nonlinear optimization with scalable metadata.

## Rosenbrock
Properties: Multimodal, non-convex, non-separable, differentiable, scalable
Minimum: [1.0, ..., 1.0] with value 0.0
Bounds: [-5.0, 5.0]^n
Example:

    rosenbrock([0.5, 0.5]) # Returns 6.5

## Sphere
Properties: Unimodal, convex, separable, differentiable, scalable
Minimum: [0.0, ..., 0.0] with value 0.0
Bounds: [-5.12, 5.12]^n
Example:

    sphere([1.0, 1.0]) # Returns 2.0

## Scalable Metadata
Access metadata for arbitrary dimensions:

    ROSENBROCK_FUNCTION.meta[:lb](3) # Returns [-5.0, -5.0, -5.0]
    SPHERE_FUNCTION.meta[:start](4) # Returns [0.0, 0.0, 0.0, 0.0]