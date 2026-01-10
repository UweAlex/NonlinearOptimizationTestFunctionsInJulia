# examples/benchmark_smart_linesearch.jl
# =============================================================================
# Purpose: 
#   Benchmark comparing L1-Wrapper with different linesearch strategies:
#
#   1. L1 + BackTracking (aktueller Standard)
#   2. L1 + HagerZhang (quadratische Interpolation)
#   3. L1 + SmartBounded (bound-aware: 90% zum Bound wenn nötig, sonst HagerZhang)
#
#   Die Idee: BackTracking ist gut weil es "wegprallt" vom Bound.
#   Aber es verschwendet Evaluierungen. SmartBounded macht das explizit:
#   - Wenn Richtung gegen Bound geht → direkt 90% des Weges, fertig
#   - Sonst → vernünftiger Linesearch (HagerZhang)
# =============================================================================

using NonlinearOptimizationTestFunctions
using Optim
using LineSearches
using Random
using Statistics
using LinearAlgebra

# --- Configuration ---
const NUM_START_POINTS = 20
const MAX_ITERATIONS   = 20_000
const ABS_TOL          = 1e-8
const BOUND_MARGIN     = 0.9  # Wie weit zum Bound (0.9 = 90%)
# ---------------------

# -------------------------------------------------------------------------
# Smart Bounded Linesearch Wrapper
# -------------------------------------------------------------------------
"""
Wrapper um eine Testfunktion der:
1. Die Suchrichtung skaliert wenn sie gegen einen Bound geht
2. So dass α=1 immer feasible ist (mit Margin)

Das "trickst" LBFGS aus: Es denkt α=1 ist ok, aber wir haben
die Richtung schon so skaliert dass wir nicht über den Bound gehen.
"""
struct SmartBoundedWrapper{TF}
    tf::TF                      # Original wrapped testfunction (L1)
    lower::Vector{Float64}
    upper::Vector{Float64}
    margin::Float64
    last_scale::Vector{Float64} # Speichert letzte Skalierung für Debugging
end

function create_smart_bounded(tf_l1, lower, upper; margin=BOUND_MARGIN)
    SmartBoundedWrapper(tf_l1, lower, upper, margin, [1.0])
end

# Die Zielfunktion bleibt gleich
function (w::SmartBoundedWrapper)(x)
    w.tf.f(x)
end

"""
Gradient-Funktion die zusätzlich die Suchrichtung für nächsten Schritt vorbereitet.

Trick: LBFGS berechnet d = -H * g
Wenn wir g skalieren, skaliert auch d.
Aber das ist falsch - wir müssen d skalieren, nicht g.

Besserer Ansatz: Wir können das nicht direkt im Gradient machen.
Stattdessen: Wrapper der die Schrittweite extern begrenzt.
"""

# -------------------------------------------------------------------------
# Alternativer Ansatz: Custom Optimizer mit bounded step
# -------------------------------------------------------------------------
"""
Einfacher Gradient Descent mit Smart Bounded Linesearch.
Nicht so gut wie LBFGS, aber zeigt das Konzept.
"""
function optimize_smart_bounded(f, grad!, x0, lower, upper;
                                maxiter=MAX_ITERATIONS, gtol=1e-8, ftol=ABS_TOL,
                                margin=BOUND_MARGIN)
    x = copy(x0)
    g = similar(x)
    n = length(x)
    
    f_val = f(x)
    grad!(g, x)
    
    f_calls = 1
    g_calls = 1
    
    for iter in 1:maxiter
        gnorm = norm(g)
        if gnorm < gtol
            return (converged=true, minimum=f_val, minimizer=x,
                    f_calls=f_calls, g_calls=g_calls, iterations=iter)
        end
        
        # Suchrichtung (steepest descent)
        d = -g
        
        # Smart bounded step: Berechne max feasible α
        α_max = Inf
        @inbounds for i in eachindex(x)
            if d[i] > 1e-12
                α_max = min(α_max, (upper[i] - x[i]) / d[i])
            elseif d[i] < -1e-12
                α_max = min(α_max, (lower[i] - x[i]) / d[i])
            end
        end
        
        if α_max < 1.0
            # Bound-Fall: Direkt margin * α_max, kein Linesearch
            α = margin * α_max
        else
            # Normalfall: Einfacher Backtracking (könnte HagerZhang sein)
            α = 1.0
            c1 = 1e-4
            f_new = f(x + α * d)
            f_calls += 1
            
            while f_new > f_val + c1 * α * dot(g, d) && α > 1e-12
                α *= 0.5
                f_new = f(x + α * d)
                f_calls += 1
            end
        end
        
        if α < 1e-12
            return (converged=true, minimum=f_val, minimizer=x,
                    f_calls=f_calls, g_calls=g_calls, iterations=iter)
        end
        
        # Update
        x_new = x + α * d
        x_new = clamp.(x_new, lower, upper)  # Sicherheit
        
        f_old = f_val
        f_val = f(x_new)
        f_calls += 1
        x .= x_new
        grad!(g, x)
        g_calls += 1
        
        if abs(f_old - f_val) < ftol * (1 + abs(f_val))
            return (converged=true, minimum=f_val, minimizer=x,
                    f_calls=f_calls, g_calls=g_calls, iterations=iter)
        end
    end
    
    return (converged=false, minimum=f_val, minimizer=x,
            f_calls=f_calls, g_calls=g_calls, iterations=maxiter)
end

# -------------------------------------------------------------------------
# LBFGS mit skalierter Suchrichtung (Hack über Gradient)
# -------------------------------------------------------------------------
"""
Wrapper der den Gradienten so modifiziert dass LBFGS kürzere Schritte macht
wenn wir Richtung Bound gehen.

Idee: Wenn |g| kleiner ist, macht LBFGS kleinere Schritte.
Aber: Das verfälscht die LBFGS-Hessian-Approximation... riskant.
"""

# Besserer Hack: Wrapper der nach jedem Schritt prüft und korrigiert
struct L1SmartWrapper{F,G}
    f::F
    grad!::G
    lower::Vector{Float64}
    upper::Vector{Float64}
    margin::Float64
    x_prev::Vector{Float64}
    scale_history::Vector{Float64}
end

function create_l1_smart_wrapper(tf, lower, upper; margin=BOUND_MARGIN, rho=1e6)
    n = length(lower)
    
    # L1 penalisierte Zielfunktion
    function f_l1(x)
        x_clamped = clamp.(x, lower, upper)
        penalty = rho * sum(i -> max(0.0, lower[i] - x[i]) + max(0.0, x[i] - upper[i]), 1:n)
        return tf.f(x_clamped) + penalty
    end
    
    x_prev = zeros(n)
    scale_history = Float64[]
    
    # Gradient mit Skalierung basierend auf Bound-Nähe
    function grad_l1_smart!(g, x)
        x_clamped = clamp.(x, lower, upper)
        tf.gradient!(g, x_clamped)
        
        # L1 Subgradient
        @inbounds for i in 1:n
            if x[i] < lower[i]
                g[i] -= rho
            elseif x[i] > upper[i]
                g[i] += rho
            end
        end
        
        # Berechne wie weit wir in Richtung -g gehen können
        α_max = Inf
        @inbounds for i in eachindex(x)
            di = -g[i]  # Suchrichtung ist -gradient
            if di > 1e-12
                α_max = min(α_max, (upper[i] - x[i]) / di)
            elseif di < -1e-12
                α_max = min(α_max, (lower[i] - x[i]) / di)
            end
        end
        
        # Wenn wir gegen Bound laufen würden: Skaliere Gradient hoch
        # damit LBFGS kleinere Schritte macht
        if α_max < 1.0 && α_max > 1e-12
            # Wir wollen dass LBFGS ungefähr α = margin * α_max nimmt
            # LBFGS initial step ist oft ~1/|g|
            # Also: größerer |g| → kleinerer Schritt
            scale = 1.0 / (margin * α_max)
            g .*= scale
            push!(scale_history, scale)
        else
            push!(scale_history, 1.0)
        end
        
        return g
    end
    
    return L1SmartWrapper(f_l1, grad_l1_smart!, lower, upper, margin, x_prev, scale_history)
end

# -------------------------------------------------------------------------
# Helper
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
# Benchmark
# -------------------------------------------------------------------------
bounded_tfs = filter_testfunctions(tf -> "bounded" in tf.meta[:properties])

println("Benchmark: L1+BackTracking vs L1+HagerZhang vs L1+Smart")
println("Found $(length(bounded_tfs)) bounded benchmark functions.")
println("Running each with $NUM_START_POINTS random strictly feasible start points")
println("Bound margin for Smart: $BOUND_MARGIN\n")

results = NamedTuple[]
rng = MersenneTwister(42)

for tf_orig in bounded_tfs
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
    tf_smart = create_l1_smart_wrapper(tf, lower, upper)

    # Zähler
    sum_bt_calls = 0.0
    sum_hz_calls = 0.0
    sum_smart_calls = 0.0
    
    bt_success = 0
    hz_success = 0
    smart_success = 0

    lbfgs_bt = LBFGS(linesearch=LineSearches.BackTracking())
    lbfgs_hz = LBFGS(linesearch=LineSearches.HagerZhang())

    for _ in 1:NUM_START_POINTS
        x0 = generate_random_start(lower, upper, rng)

        # --- 1. L1 + BackTracking (Standard) ---
        reset_counts!(tf_l1)
        try
            res = optimize(tf_l1.f, tf_l1.gradient!, x0, lbfgs_bt,
                          Optim.Options(iterations=MAX_ITERATIONS, f_reltol=ABS_TOL))
            calls = get_f_count(tf_l1) + get_grad_count(tf_l1)
            sum_bt_calls += calls
            Optim.converged(res) && (bt_success += 1)
        catch e
            @warn "BackTracking failed for $name: $e"
        end

        # --- 2. L1 + HagerZhang ---
        reset_counts!(tf_l1)
        try
            res = optimize(tf_l1.f, tf_l1.gradient!, x0, lbfgs_hz,
                          Optim.Options(iterations=MAX_ITERATIONS, f_reltol=ABS_TOL))
            calls = get_f_count(tf_l1) + get_grad_count(tf_l1)
            sum_hz_calls += calls
            Optim.converged(res) && (hz_success += 1)
        catch e
            @warn "HagerZhang failed for $name: $e"
        end

        # --- 3. L1 + Smart (Gradient-Skalierung) ---
        reset_counts!(tf)
        empty!(tf_smart.scale_history)
        try
            res = optimize(tf_smart.f, tf_smart.grad!, x0, lbfgs_hz,
                          Optim.Options(iterations=MAX_ITERATIONS, f_reltol=ABS_TOL))
            calls = get_f_count(tf) + get_grad_count(tf)
            sum_smart_calls += calls
            Optim.converged(res) && (smart_success += 1)
        catch e
            @warn "Smart failed for $name: $e"
        end
    end

    avg_bt = bt_success > 0 ? sum_bt_calls / bt_success : Inf
    avg_hz = hz_success > 0 ? sum_hz_calls / hz_success : Inf
    avg_smart = smart_success > 0 ? sum_smart_calls / smart_success : Inf

    succ_bt = bt_success / NUM_START_POINTS
    succ_hz = hz_success / NUM_START_POINTS
    succ_smart = smart_success / NUM_START_POINTS

    push!(results, (
        name = name, n = n,
        avg_bt = avg_bt, avg_hz = avg_hz, avg_smart = avg_smart,
        succ_bt = succ_bt, succ_hz = succ_hz, succ_smart = succ_smart
    ))

    # Status
    rates = [succ_bt, succ_hz, succ_smart]
    names_ls = ["BT", "HZ", "Smart"]
    max_rate = maximum(rates)
    winners = [names_ls[i] for i in eachindex(rates) if rates[i] == max_rate]
    
    status = if length(winners) == 3 || max_rate == 0.0
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

# Paarweise Vergleiche
bt_beats_hz = count(r -> r.succ_bt > r.succ_hz, results)
hz_beats_bt = count(r -> r.succ_hz > r.succ_bt, results)
bt_hz_tie = count(r -> r.succ_bt == r.succ_hz, results)

bt_beats_smart = count(r -> r.succ_bt > r.succ_smart, results)
smart_beats_bt = count(r -> r.succ_smart > r.succ_bt, results)

hz_beats_smart = count(r -> r.succ_hz > r.succ_smart, results)
smart_beats_hz = count(r -> r.succ_smart > r.succ_hz, results)

println("\n--- PAARWEISE VERGLEICHE (Robustheit) ---")
println()
println("BackTracking vs HagerZhang:")
println("  BT gewinnt: $bt_beats_hz")
println("  HZ gewinnt: $hz_beats_bt")
println("  Gleich:     $bt_hz_tie")
println()
println("BackTracking vs Smart:")
println("  BT gewinnt:    $bt_beats_smart")
println("  Smart gewinnt: $smart_beats_bt")
println()
println("HagerZhang vs Smart:")
println("  HZ gewinnt:    $hz_beats_smart")
println("  Smart gewinnt: $smart_beats_hz")

# Gesamtbilanz
wins_bt = bt_beats_hz + bt_beats_smart
wins_hz = hz_beats_bt + hz_beats_smart
wins_smart = smart_beats_bt + smart_beats_hz

losses_bt = hz_beats_bt + smart_beats_bt
losses_hz = bt_beats_hz + smart_beats_hz
losses_smart = bt_beats_smart + hz_beats_smart

println("\n--- GESAMTBILANZ ---")
println()
println("Methode      Siege    Niederlagen    Differenz")
println("-"^50)
println("BackTracking $(lpad(wins_bt, 5))    $(lpad(losses_bt, 11))    $(lpad(wins_bt - losses_bt, 9))")
println("HagerZhang   $(lpad(wins_hz, 5))    $(lpad(losses_hz, 11))    $(lpad(wins_hz - losses_hz, 9))")
println("Smart        $(lpad(wins_smart, 5))    $(lpad(losses_smart, 11))    $(lpad(wins_smart - losses_smart, 9))")

# Effizienz
bt_faster_hz = count(r -> r.succ_bt == r.succ_hz > 0 && r.avg_bt < r.avg_hz, results)
hz_faster_bt = count(r -> r.succ_bt == r.succ_hz > 0 && r.avg_hz < r.avg_bt, results)

bt_faster_smart = count(r -> r.succ_bt == r.succ_smart > 0 && r.avg_bt < r.avg_smart, results)
smart_faster_bt = count(r -> r.succ_bt == r.succ_smart > 0 && r.avg_smart < r.avg_bt, results)

println("\n--- EFFIZIENZ (bei gleicher Robustheit) ---")
println()
println("BackTracking vs HagerZhang:")
println("  BT schneller: $bt_faster_hz")
println("  HZ schneller: $hz_faster_bt")
println()
println("BackTracking vs Smart:")
println("  BT schneller:    $bt_faster_smart")
println("  Smart schneller: $smart_faster_bt")

# Empfehlung
println("\n")
println("="^70)
println("EMPFEHLUNG")
println("="^70)

diffs = [("BackTracking", wins_bt - losses_bt),
         ("HagerZhang", wins_hz - losses_hz),
         ("Smart", wins_smart - losses_smart)]
sort!(diffs, by=x->x[2], rev=true)

println()
for (i, (name, diff)) in enumerate(diffs)
    marker = i == 1 ? "→ EMPFOHLEN" : ""
    println("  $i. $name (Differenz: $(diff > 0 ? "+" : "")$diff) $marker")
end

println()
println("Benchmark completed.")