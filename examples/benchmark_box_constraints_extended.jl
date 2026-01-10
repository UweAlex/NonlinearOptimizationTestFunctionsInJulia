# examples/benchmark_box_constraints_projected.jl
# =============================================================================
# Purpose: 
#   Benchmark comparing FOUR box-constraint handling methods including
#   a new Projected Gradient approach that prevents steps outside bounds.
#
#   Comparison:
#     1. Fminbox + LBFGS (standard)
#     2. L1 Domain-safe Wrapper (with_box_constraints) + LBFGS BackTracking
#     3. L1 + Projected Gradient (neu: begrenzte Schrittweite + Gradient-Projektion)
#     4. Pure Projected Gradient (ohne L1 Penalty)
#
#   Die Idee bei Projected Gradient:
#     - Linesearch darf nie außerhalb der Bounds schreiten
#     - Gradient-Komponenten die "nach außen" zeigen werden auf 0 gesetzt
#     - Kein Zigzagging an den Bounds
# =============================================================================

using NonlinearOptimizationTestFunctions
using Optim
using LineSearches
using Random
using Statistics
using LinearAlgebra  # für norm, dot

# --- Configuration ---
const NUM_START_POINTS = 20
const MAX_ITERATIONS   = 20_000
const ABS_TOL          = 1e-8
const RHO_L1           = 1e6
# ---------------------

# -------------------------------------------------------------------------
# Projected Gradient Wrapper
# -------------------------------------------------------------------------
"""
Erstellt einen Wrapper der:
1. Gradient-Komponenten auf 0 setzt wenn sie am Bound "nach außen" zeigen
2. Eine max_step Funktion bereitstellt für begrenzte Linesearch
"""
struct ProjectedWrapper{F,G}
    f::F
    grad!::G
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function create_projected_wrapper(tf, lower::Vector{Float64}, upper::Vector{Float64})
    # Projizierte Gradient-Funktion
    function projected_grad!(g, x)
        tf.gradient!(g, x)
        # Projektion: Gradient-Komponenten die nach außen zeigen → 0
        @inbounds for i in eachindex(x)
            if x[i] <= lower[i] && g[i] > 0
                g[i] = 0.0
            elseif x[i] >= upper[i] && g[i] < 0
                g[i] = 0.0
            end
        end
        return g
    end
    
    return ProjectedWrapper(tf.f, projected_grad!, lower, upper)
end

"""
Berechnet maximale Schrittweite α so dass x + α*d innerhalb der Bounds bleibt
"""
function max_feasible_step(x::Vector{Float64}, d::Vector{Float64}, 
                           lower::Vector{Float64}, upper::Vector{Float64})
    α_max = Inf
    @inbounds for i in eachindex(x)
        if d[i] > 1e-12
            α_max = min(α_max, (upper[i] - x[i]) / d[i])
        elseif d[i] < -1e-12
            α_max = min(α_max, (lower[i] - x[i]) / d[i])
        end
    end
    return max(α_max, 0.0)
end

# -------------------------------------------------------------------------
# L1 + Projected Wrapper (kombiniert beide Ansätze)
# Mit Toleranz: Projektion greift schon NAHE den Bounds
# -------------------------------------------------------------------------
struct L1ProjectedWrapper{F,G}
    f::F
    grad!::G
    lower::Vector{Float64}
    upper::Vector{Float64}
    rho::Float64
end

function create_l1_projected_wrapper(tf, lower::Vector{Float64}, upper::Vector{Float64}; 
                                      rho=RHO_L1, tol=1e-4)
    n = length(lower)
    
    # L1 penalisierte Zielfunktion mit Clamping
    function f_l1(x)
        x_clamped = clamp.(x, lower, upper)
        penalty = rho * sum(i -> max(0.0, lower[i] - x[i]) + max(0.0, x[i] - upper[i]), 1:n)
        return tf.f(x_clamped) + penalty
    end
    
    # L1 Gradient mit Toleranz-basierter Projektion
    function grad_l1_projected!(g, x)
        x_clamped = clamp.(x, lower, upper)
        tf.gradient!(g, x_clamped)
        
        # L1 Subgradient addieren
        @inbounds for i in 1:n
            if x[i] < lower[i]
                g[i] -= rho
            elseif x[i] > upper[i]
                g[i] += rho
            end
        end
        
        # Toleranz-basierte Projektion: NAHE am Bound und Gradient zeigt nach außen → 0
        @inbounds for i in eachindex(x)
            if x[i] <= lower[i] + tol && g[i] > 0
                g[i] = 0.0
            elseif x[i] >= upper[i] - tol && g[i] < 0
                g[i] = 0.0
            end
        end
        
        return g
    end
    
    return L1ProjectedWrapper(f_l1, grad_l1_projected!, lower, upper, rho)
end

# -------------------------------------------------------------------------
# Custom Linesearch mit Schrittweitenbegrenzung
# -------------------------------------------------------------------------
"""
BackTracking Linesearch mit maximaler Schrittweite begrenzt auf feasible region
"""
struct BoundedBackTracking{T}
    lower::Vector{Float64}
    upper::Vector{Float64}
    base_ls::T
end

function BoundedBackTracking(lower, upper)
    BoundedBackTracking(lower, upper, LineSearches.BackTracking())
end

# -------------------------------------------------------------------------
# Helper: Generate strictly feasible random start point
# -------------------------------------------------------------------------
function generate_random_start(lower::Vector{Float64}, upper::Vector{Float64}, rng::MersenneTwister)
    n = length(lower)
    x0 = Vector{Float64}(undef, n)
    epsilon = 1e-8

    @inbounds for i in 1:n
        l_eff = lower[i] + epsilon
        u_eff = upper[i] - epsilon
        if u_eff <= l_eff
            x0[i] = (lower[i] + upper[i]) / 2
        else
            x0[i] = l_eff + rand(rng) * (u_eff - l_eff)
        end
    end
    return x0
end

# -------------------------------------------------------------------------
# Manuelle Optimierung mit projiziertem Gradient
# -------------------------------------------------------------------------
function optimize_projected(f, grad!, x0, lower, upper; 
                           maxiter=MAX_ITERATIONS, gtol=1e-8, ftol=ABS_TOL)
    x = copy(x0)
    g = similar(x)
    n = length(x)
    
    f_val = f(x)
    grad!(g, x)
    
    f_calls = 1
    g_calls = 1
    
    α = 1.0
    
    for iter in 1:maxiter
        # Prüfe Konvergenz
        gnorm = norm(g)
        if gnorm < gtol
            return (converged=true, minimum=f_val, minimizer=x, 
                    f_calls=f_calls, g_calls=g_calls, iterations=iter)
        end
        
        # Suchrichtung (Gradient Descent)
        d = -g
        
        # Maximale feasible Schrittweite
        α_max = max_feasible_step(x, d, lower, upper)
        if α_max < 1e-12
            # Keine Bewegung möglich
            return (converged=true, minimum=f_val, minimizer=x,
                    f_calls=f_calls, g_calls=g_calls, iterations=iter)
        end
        
        # Backtracking Linesearch mit begrenzter Schrittweite
        α = min(1.0, 0.99 * α_max)
        c1 = 1e-4
        ρ = 0.5
        
        f_new = f(x + α * d)
        f_calls += 1
        
        while f_new > f_val + c1 * α * dot(g, d)
            α *= ρ
            if α < 1e-12
                break
            end
            f_new = f(x + α * d)
            f_calls += 1
        end
        
        # Update
        x .+= α .* d
        
        # Clamp zur Sicherheit (numerische Fehler)
        x .= clamp.(x, lower, upper)
        
        f_old = f_val
        f_val = f_new
        grad!(g, x)
        g_calls += 1
        
        # Konvergenz nach Funktionswert
        if abs(f_old - f_val) < ftol * (1 + abs(f_val))
            return (converged=true, minimum=f_val, minimizer=x,
                    f_calls=f_calls, g_calls=g_calls, iterations=iter)
        end
    end
    
    return (converged=false, minimum=f_val, minimizer=x,
            f_calls=f_calls, g_calls=g_calls, iterations=maxiter)
end

# -------------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------------
bounded_tfs = filter_testfunctions(tf -> "bounded" in tf.meta[:properties])

println("Benchmark: Fminbox vs L1 vs L1+Projected vs PureProjected")
println("Found $(length(bounded_tfs)) bounded benchmark functions.")
println("Running each with $NUM_START_POINTS random strictly feasible start points\n")

results = NamedTuple[]
rng = MersenneTwister(42)

for tf_orig in bounded_tfs
    # Fix dimension
    tf = if scalable(tf_orig)
        fixed(tf_orig; n=tf_orig.meta[:default_n])
    else
        tf_orig
    end

    name = tf.name
    n = dim(tf)

    if n > 50
        println("Skipping $name (n=$n – too large)")
        continue
    end

    lower = Float64.(lb(tf))
    upper = Float64.(ub(tf))

    # Wrapper erstellen
    tf_l1 = with_box_constraints(tf)
    l1_proj = create_l1_projected_wrapper(tf, lower, upper)
    proj = create_projected_wrapper(tf, lower, upper)

    # Zähler
    sum_fminbox_calls = 0.0
    sum_l1_calls = 0.0
    sum_l1proj_calls = 0.0
    sum_proj_calls = 0.0
    
    fminbox_success = 0
    l1_success = 0
    l1proj_success = 0
    proj_success = 0

    lbfgs_stable = LBFGS(linesearch=LineSearches.BackTracking())
    lbfgs_standard = LBFGS()

    for _ in 1:NUM_START_POINTS
        x0 = generate_random_start(lower, upper, rng)

        # --- 1. Fminbox ---
        reset_counts!(tf)
        try
            res = optimize(tf.f, tf.gradient!, lower, upper, x0,
                          Fminbox(lbfgs_standard),
                          Optim.Options(iterations=MAX_ITERATIONS, f_reltol=ABS_TOL))
            calls = get_f_count(tf) + get_grad_count(tf)
            sum_fminbox_calls += calls
            Optim.converged(res) && (fminbox_success += 1)
        catch e
            @warn "Fminbox failed for $name: $e"
        end

        # --- 2. L1 wrapper (standard) ---
        reset_counts!(tf_l1)
        try
            res = optimize(tf_l1.f, tf_l1.gradient!, x0, lbfgs_stable,
                          Optim.Options(iterations=MAX_ITERATIONS, f_reltol=ABS_TOL))
            calls = get_f_count(tf_l1) + get_grad_count(tf_l1)
            sum_l1_calls += calls
            Optim.converged(res) && (l1_success += 1)
        catch e
            @warn "L1 failed for $name: $e"
        end

        # --- 3. L1 + Projected ---
        reset_counts!(tf)
        try
            res = optimize(l1_proj.f, l1_proj.grad!, x0, lbfgs_stable,
                          Optim.Options(iterations=MAX_ITERATIONS, f_reltol=ABS_TOL))
            calls = get_f_count(tf) + get_grad_count(tf)
            sum_l1proj_calls += calls
            Optim.converged(res) && (l1proj_success += 1)
        catch e
            @warn "L1+Proj failed for $name: $e"
        end

        # --- 4. Pure Projected (custom optimizer) ---
        reset_counts!(tf)
        try
            res = optimize_projected(tf.f, proj.grad!, x0, lower, upper)
            calls = res.f_calls + res.g_calls
            sum_proj_calls += calls
            res.converged && (proj_success += 1)
        catch e
            @warn "Projected failed for $name: $e"
        end
    end

    avg_fminbox = fminbox_success > 0 ? sum_fminbox_calls / fminbox_success : Inf
    avg_l1 = l1_success > 0 ? sum_l1_calls / l1_success : Inf
    avg_l1proj = l1proj_success > 0 ? sum_l1proj_calls / l1proj_success : Inf
    avg_proj = proj_success > 0 ? sum_proj_calls / proj_success : Inf

    succ_fminbox = fminbox_success / NUM_START_POINTS
    succ_l1 = l1_success / NUM_START_POINTS
    succ_l1proj = l1proj_success / NUM_START_POINTS
    succ_proj = proj_success / NUM_START_POINTS

    push!(results, (
        name = name, n = n,
        avg_fminbox = avg_fminbox, avg_l1 = avg_l1, 
        avg_l1proj = avg_l1proj, avg_proj = avg_proj,
        succ_fminbox = succ_fminbox, succ_l1 = succ_l1,
        succ_l1proj = succ_l1proj, succ_proj = succ_proj
    ))

    # Status
    rates = [succ_fminbox, succ_l1, succ_l1proj, succ_proj]
    names = ["Fminbox", "L1", "L1+Proj", "Proj"]
    max_rate = maximum(rates)
    winners = [names[i] for i in eachindex(rates) if rates[i] == max_rate]
    
    status = if length(winners) == 4 || max_rate == 0.0
        "Alle gleich ($(round(max_rate*100,digits=1))%)"
    elseif length(winners) > 1
        join(winners, "/") * " ($(round(max_rate*100,digits=1))%)"
    else
        "$(winners[1]) best ($(round(max_rate*100,digits=1))%)"
    end
    
    println("$(lpad(name, 35)) (n=$(lpad(n,2))) → $status")
end

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
println("\n")
println("="^70)
println("BENCHMARK SUMMARY ($NUM_START_POINTS starts per function)")
println("="^70)

# Paarweise Vergleiche L1 vs L1+Proj
l1_beats_l1proj = count(r -> r.succ_l1 > r.succ_l1proj, results)
l1proj_beats_l1 = count(r -> r.succ_l1proj > r.succ_l1, results)
l1_l1proj_tie = count(r -> r.succ_l1 == r.succ_l1proj, results)

# L1 vs Proj
l1_beats_proj = count(r -> r.succ_l1 > r.succ_proj, results)
proj_beats_l1 = count(r -> r.succ_proj > r.succ_l1, results)

# L1+Proj vs Proj
l1proj_beats_proj = count(r -> r.succ_l1proj > r.succ_proj, results)
proj_beats_l1proj = count(r -> r.succ_proj > r.succ_l1proj, results)

# Fminbox Vergleiche
l1_beats_fminbox = count(r -> r.succ_l1 > r.succ_fminbox, results)
fminbox_beats_l1 = count(r -> r.succ_fminbox > r.succ_l1, results)

l1proj_beats_fminbox = count(r -> r.succ_l1proj > r.succ_fminbox, results)
fminbox_beats_l1proj = count(r -> r.succ_fminbox > r.succ_l1proj, results)

proj_beats_fminbox = count(r -> r.succ_proj > r.succ_fminbox, results)
fminbox_beats_proj = count(r -> r.succ_fminbox > r.succ_proj, results)

println("\n--- PAARWEISE VERGLEICHE (Robustheit) ---")
println()
println("L1 vs L1+Projected:")
println("  L1 gewinnt:      $l1_beats_l1proj")
println("  L1+Proj gewinnt: $l1proj_beats_l1")
println("  Unentschieden:   $l1_l1proj_tie")
println()
println("L1 vs Fminbox:")
println("  L1 gewinnt:      $l1_beats_fminbox")
println("  Fminbox gewinnt: $fminbox_beats_l1")
println()
println("L1+Proj vs Fminbox:")
println("  L1+Proj gewinnt: $l1proj_beats_fminbox")
println("  Fminbox gewinnt: $fminbox_beats_l1proj")
println()
println("Pure Projected vs L1:")
println("  Proj gewinnt:    $proj_beats_l1")
println("  L1 gewinnt:      $l1_beats_proj")

# Gesamtbilanz
wins_fminbox = fminbox_beats_l1 + fminbox_beats_l1proj + fminbox_beats_proj
wins_l1 = l1_beats_fminbox + l1_beats_l1proj + l1_beats_proj
wins_l1proj = l1proj_beats_fminbox + l1proj_beats_l1 + l1proj_beats_proj
wins_proj = proj_beats_fminbox + proj_beats_l1 + proj_beats_l1proj

losses_fminbox = l1_beats_fminbox + l1proj_beats_fminbox + proj_beats_fminbox
losses_l1 = fminbox_beats_l1 + l1proj_beats_l1 + proj_beats_l1
losses_l1proj = fminbox_beats_l1proj + l1_beats_l1proj + proj_beats_l1proj
losses_proj = fminbox_beats_proj + l1_beats_proj + l1proj_beats_proj

println("\n--- GESAMTBILANZ ---")
println()
println("Methode      Siege    Niederlagen    Differenz")
println("-"^50)
println("Fminbox      $(lpad(wins_fminbox, 5))    $(lpad(losses_fminbox, 11))    $(lpad(wins_fminbox - losses_fminbox, 9))")
println("L1           $(lpad(wins_l1, 5))    $(lpad(losses_l1, 11))    $(lpad(wins_l1 - losses_l1, 9))")
println("L1+Proj      $(lpad(wins_l1proj, 5))    $(lpad(losses_l1proj, 11))    $(lpad(wins_l1proj - losses_l1proj, 9))")
println("Proj         $(lpad(wins_proj, 5))    $(lpad(losses_proj, 11))    $(lpad(wins_proj - losses_proj, 9))")

# Effizienz L1 vs L1+Proj bei gleicher Robustheit
l1_faster = count(r -> r.succ_l1 == r.succ_l1proj > 0 && r.avg_l1 < r.avg_l1proj, results)
l1proj_faster = count(r -> r.succ_l1 == r.succ_l1proj > 0 && r.avg_l1proj < r.avg_l1, results)

println("\n--- EFFIZIENZ (L1 vs L1+Proj bei gleicher Robustheit) ---")
println("  L1 schneller:      $l1_faster")
println("  L1+Proj schneller: $l1proj_faster")

# Empfehlung
println("\n")
println("="^70)
println("EMPFEHLUNG")
println("="^70)

diffs = [("Fminbox", wins_fminbox - losses_fminbox),
         ("L1", wins_l1 - losses_l1),
         ("L1+Proj", wins_l1proj - losses_l1proj),
         ("Proj", wins_proj - losses_proj)]
sort!(diffs, by=x->x[2], rev=true)

println()
for (i, (name, diff)) in enumerate(diffs)
    marker = i == 1 ? "→ EMPFOHLEN" : ""
    println("  $i. $name (Differenz: $(diff > 0 ? "+" : "")$diff) $marker")
end

println()
println("Benchmark completed.")