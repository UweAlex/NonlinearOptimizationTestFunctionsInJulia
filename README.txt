NonlinearOptimizationTestFunctionsInJulia
Last modified: 11. Juli 2025, 10:23 AM CEST

Purpose:
This Julia package provides test functions for nonlinear optimization, including Rosenbrock and Sphere, with analytical gradients and metadata. It is designed to be simple, scalable, and compatible with Optim.jl, NLopt, and Hessian-based methods.

Features:
- Modular structure: Test functions in src/functions/ (rosenbrock.jl, sphere.jl), each under 100 lines.
- TestFunction structure: Contains f (function), grad (non-in-place gradient), gradient! (in-place gradient), and meta (Dict with name, start, min_position, min_value, properties, lb, ub, description).
- Search domains: Defined in meta[:lb] and meta[:ub], e.g., [-5, 5]^n for Rosenbrock, [-5.12, 5.12]^n for Sphere.
- Compatibility: Pure Julia, supports Julia 1.11.5+, Optim.jl, NLopt, and Hessian methods (Zygote.hessian with ForwardDiff.Dual support).
- Tests: 60/60 tests in test/runtests.jl, covering function values, gradients, edge cases (NaN, Inf, 1e-308), and properties (e.g., multimodal, convex).
- Demos: Five minimal scripts (10-15 lines) in examples/, e.g., Optimize_all_functions.jl (L-BFGS), Compare_optimization_methods.jl (Gradient Descent vs. L-BFGS), Optimize_with_nlopt.jl, Compute_hessian_with_zygote.jl (fixed for ForwardDiff.Dual).
- Load time: Target < 0.5 seconds for two functions.
- Error handling: Replaced @assert with throw(ArgumentError(...)) for robustness.

Usage:
1. Install: Clone from GitHub (Julia General Registry planned).
2. Load: using NonlinearOptimizationTestFunctionsInJulia
3. Example (L-BFGS optimization):
   tf = NonlinearOptimizationTestFunctionsInJulia.ROSENBROCK_FUNCTION
   result = optimize(tf.f, tf.gradient!, tf.meta[:start], LBFGS())
   println("$(tf.meta[:name]): $(Optim.minimizer(result)), $(Optim.minimum(result))")
4. List functions: Run examples/List_all_available_test_functions_and_their_properties.jl.

Planned Improvements:
- Optimize load time with FUNCTION_REGISTRY in src/registry.jl.
- Add numerical gradient tests in test/runtests.jl.
- Enhance documentation with docstrings and Documenter.jl.
- Add tolerances (e.g., ftol_rel=1e-6) in demos.

Dependencies:
- Required: LinearAlgebra, Optim, Zygote, Test
- Optional: NLopt (for examples/Optimize_with_nlopt.jl)

Notes:
- No Markdown used to avoid rendering issues.
- File comments in src/, examples/, test/ include path, purpose, context, and timestamp.
- Comparison with Opfunu (Python, 500+ functions, no gradients) is internal, not public.
- Future: Julia Discourse post (deferred).

Contact:
GitHub repository (link TBD), Julia Discourse (deferred).