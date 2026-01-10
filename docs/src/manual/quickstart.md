# bench.jl – Vergleich SHGO.jl vs SciPy mit PyCall (stabiler Fallback)

using Pkg
Pkg.activate(".")

# === SciPy-Installation sicherstellen ===
println("Prüfe/Installiere SciPy in PyCall-Umgebung...")
try
    using Conda
    if !Conda.exists("scipy")
        println("SciPy wird installiert...")
        Conda.add("scipy")
        println("SciPy installiert!")
    else
        println("SciPy bereits vorhanden.")
    end
catch e
    @warn "Conda-Installation fehlgeschlagen – versuche manuell: Conda.add(\"scipy\")" exception=e
end

# === Pakete laden ===
using PyCall
using SHGO
using NonlinearOptimizationTestFunctions
using Printf
using BenchmarkTools

println("Initialisiere PyCall...")
const sp_opt = pyimport("scipy.optimize")   # SciPy laden – sollte jetzt funktionieren

# Setup
fn_name = "sixhumpcamelback"
tf = fixed(TEST_FUNCTIONS[fn_name]; n=2)
n_div = 12
scipy_n = (n_div + 1)^2   # SciPy n entspricht ungefähr (n_div + 1)^2 Punkten

println("="^60)
println("STUFE 1: DIFFERENZ-ANALYSE - $fn_name")
println("="^60)

# --- FUNKTION: Julia Lauf ---
function run_julia(tf, n_div)
    res = analyze(tf; n_div=n_div, use_gradient_pruning=true)
    return res
end

# --- FUNKTION: SciPy Lauf ---
function run_scipy(tf, scipy_n)
    # Python-Funktion Wrapper
    py_fn(x) = begin
        julia_x = pyconvert(Vector{Float64}, x)
        tf.f(julia_x)
    end
    
    py_grad(x) = begin
        julia_x = pyconvert(Vector{Float64}, x)
        tf.grad(julia_x)
    end
    
    # Bounds konvertieren – ROBUST FÜR DICT-STRUKTUR
    bounds = try
        if hasproperty(tf, :bounds) && tf.bounds isa Dict
            # Für Dict: über values iterieren (unabhängig von Keys)
            [(b.lb, b.ub) for b in values(tf.bounds)]
        elseif hasproperty(tf, :bounds) && tf.bounds isa Vector
            [(b.lb, b.ub) for b in tf.bounds]
        elseif hasproperty(tf, :lb) && hasproperty(tf, :ub)
            [(tf.lb, tf.ub)]
        else
            error("Unbekannte Bounds-Struktur: $(typeof(tf.bounds))")
        end
    catch e
        @error "Bounds-Extraktion fehlgeschlagen!" exception=e
        error("Kann Bounds nicht extrahieren – überprüfe tf.bounds-Struktur.")
    end
    
    # SciPy shgo aufrufen
    res_py = sp_opt.shgo(py_fn, bounds, n=scipy_n, iters=1, jac=py_grad)
    return res_py
end

# --- Julia Benchmarking ---
println("\n[1/2] Starte Julia Benchmarking...")
println("      Warm-up läuft...")
run_julia(tf, n_div)  # Warm-up

println("      Messung läuft...")
t_jl = @belapsed run_julia($tf, $n_div) samples=3 evals=1

# Julia Ergebnis sammeln
res_jl = run_julia(tf, n_div)

# --- SciPy Benchmarking ---
println("\n[2/2] Starte SciPy Benchmarking...")
println("      Warm-up läuft...")
try
    run_scipy(tf, scipy_n)  # Warm-up
catch e
    @warn "SciPy Warm-up fehlgeschlagen" exception=e
end

println("      Messung läuft...")
t_py_start = time()
try
    res_py = run_scipy(tf, scipy_n)
    t_py = time() - t_py_start
catch e
    @error "SciPy Messung fehlgeschlagen!" exception=e
    t_py = NaN
    res_py = nothing
end

# --- Ergebnisse extrahieren ---
println("\n" * "="^60)
println("ERGEBNIS-VERGLEICH")
println("="^60)

# Julia Ergebnisse
jl_n_basins = res_jl.num_basins
jl_best_obj = isempty(res_jl.local_minima) ? NaN : minimum(m.objective for m in res_jl.local_minima)

# SciPy Ergebnisse (mit Fehlerbehandlung)
py_n_basins = try
    length(pyconvert(Vector, res_py.xl))
catch
    0
end

py_best_obj = try
    pyconvert(Float64, res_py.fun)
catch
    NaN
end

# Tabelle ausgeben
println("\n" * "-"^60)
@printf "%-25s | %-15s | %-15s\n" "Metrik" "SHGO.jl" "SciPy"
println("-"^60)
@printf "%-25s | %-15d | %-15d\n" "Gefundene Basins" jl_n_basins py_n_basins
@printf "%-25s | %-15.8f | %-15.8f\n" "Globales Minimum" jl_best_obj py_best_obj
@printf "%-25s | %-15.6fs | %-15.6fs\n" "Wall-clock Time" t_jl t_py
println("-"^60)

# Speedup berechnen
if isfinite(t_jl) && isfinite(t_py) && t_jl > 0 && t_py > 0
    speedup = t_py / t_jl
    if speedup > 1
        @printf "\n✓ SHGO.jl ist %.2fx schneller als SciPy\n" speedup
    else
        @printf "\n⚠ SciPy ist %.2fx schneller als SHGO.jl\n" (1/speedup)
    end
else
    println("\n⚠ Zeitmessung ungültig (NaN oder Fehler)")
end

println("\n" * "="^60)



NonlinearOptimizationTestFunctions.jl – Detailed Quick Reference (English)

1. Property Queries – Most Frequently Used

properties(tf)                                  
→ Returns all properties as a sorted Vector{String}
Example: ["continuous", "differentiable", "multimodal", "scalable", "non-separable"]

has_property(tf, "scalable")                    
→ true / false

has_property(tf, "has_noise")                   
→ true for noisy functions (e.g. Quartic)

has_property(tf, ["multimodal", "non-convex", "bounded"])  
→ true only if ALL requested properties are present

Common shortcuts (internally using has_property – this is the recommended way):
scalable(tf)       === has_property(tf, "scalable")
is_bounded(tf)     === has_property(tf, "bounded")
is_noisy(tf)       === has_property(tf, "has_noise")

2. Core Metadata Accessors

name(tf)               
→ Clean lowercase function name
Example: "rosenbrock", "rastrigin", "ackley"

dim(tf)                
→ -1 = arbitrarily scalable
→ ≥ 2 = fixed dimension (e.g. 2 for Himmelblau)

default_n(tf)          
→ Recommended default dimension for scalable functions
→ Only defined when the function is scalable (most often 2, sometimes 4+)

description(tf)        
→ Human-readable description (often includes origin & special remarks)

math(tf)               
→ LaTeX formula as raw string

source(tf)             
→ Scientific/literature reference
Example: "Jamil & Yang (2013, p. 29)"

3. Starting Point, Known Solutions & Bounds

start(tf)          or   start(tf, n)  
→ Recommended starting point (usually far from optimum)

min_position(tf)   or   min_position(tf, n)  
→ Position(s) of the global minimum/minima

min_value(tf)      or   min_value(tf, n)  
→ Function value at the global minimum

lb(tf)   or   lb(tf, n)     → lower bounds (Vector{Float64})
ub(tf)   or   ub(tf, n)     → upper bounds

4. Function Evaluation & Analytical Gradient

tf.f(x)                
→ Evaluate objective function → returns Float64

tf.grad(x)             
→ Compute gradient (out-of-place) → returns Vector{Float64}

tf.gradient!(g, x)     
→ In-place gradient computation (usually faster & more memory efficient)

5. Function & Gradient Call Counting (very useful for profiling/benchmarking)

get_f_count(tf)        
→ How many times has the objective function been called so far?

get_grad_count(tf)     
→ How many gradient evaluations so far?
(counts both grad() and gradient!())

reset_counts!(tf)      
→ Reset both counters to zero
→ Practically the only important mutating function!
→ Always call this before each new optimization experiment

6. Typical Usage Example (copy-paste friendly)

# Get a function (two equivalent ways)
tf = TEST_FUNCTIONS["rosenbrock"]
tf = ROSENBROCK_FUNCTION           # uppercase exported constant

# Basic information
println("Function: ", name(tf))
println("Dimension: ", dim(tf), " (scalable = ", scalable(tf), ")")
println("Properties: ", join(properties(tf), ", "))
println("Source: ", source(tf))

# Prepare experiment
reset_counts!(tf)                  # very important!

n = dim(tf) < 0 ? default_n(tf) : dim(tf)
x₀ = start(tf, n)                  # safe for both fixed and scalable

println("Start point: ", x₀)
println("Known global minimum: ", min_value(tf, n), " at ", min_position(tf, n))

# Evaluate once
f₀ = tf.f(x₀)
∇f₀ = tf.grad(x₀)

println("f(x₀) = ", f₀)
println("Calls so far → f: ", get_f_count(tf), ", ∇: ", get_grad_count(tf))