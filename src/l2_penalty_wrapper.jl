# src/l2_penalty_wrapper.jl
# Domain-safe box constraints via L2 quadratic penalty
# Sehr enge strukturelle und logische Kopie des L1-Wrappers (with_box_constraints)

const L2_BOUND_PENALTY = 1e8   # Bei L2 meist deutlich größer als bei L1 nötig (1e8–1e12 typisch)

"""
    with_l2_box_constraints(tf::TestFunction) -> TestFunction

Domain-safe L2 quadratic penalty wrapper for bounded test functions.
Identische Safety-Logik wie `with_box_constraints` (L1), nur Penalty quadratisch.

- Immer zuerst Projektion auf [lb, ub]
- f und ∇f werden **ausschließlich** im Inneren ausgewertet
- Quadratischer Strafterm + linearer Gradientenzug in der Verletzung
"""
function with_l2_box_constraints(tf::TestFunction)
    "bounded" ∉ tf.meta[:properties] && return tf

    tf_fixed = fixed(tf)
    n = dim(tf_fixed)

    if n < 1
        error("Invalid dimension $n for function $(tf_fixed.name)")
    end

    lb_vec = Float64.(tf_fixed.meta[:lb]())
    ub_vec = Float64.(tf_fixed.meta[:ub]())

    if length(lb_vec) != n || length(ub_vec) != n
        error("Bounds dimension mismatch for $(tf_fixed.name)")
    end

    if any(lb_vec .>= ub_vec)
        error("Invalid bounds for $(tf_fixed.name)")
    end

    x_buffer = Vector{Float64}(undef, n)

    shared_f_count   = tf_fixed.f_count
    shared_grad_count = tf_fixed.grad_count

    # ─────────────────────────────────────────────────────────────────────
    # Objective: clamp → eval inside → add quadratic penalty
    # ─────────────────────────────────────────────────────────────────────
    f_wrap = let x_buffer = x_buffer,
        lb_vec = lb_vec,
        ub_vec = ub_vec,
        tf_inner = tf_fixed,
        f_count_ref = shared_f_count

        function penalty_objective(x::AbstractVector{T}) where T
            any(isnan, x) && return T(NaN)
            any(isinf, x) && return T(Inf)

            violation_sq = zero(T)
            @inbounds @simd for i in eachindex(x)
                xi = x[i]
                if xi < lb_vec[i]
                    viol = lb_vec[i] - xi
                    violation_sq += viol * viol
                    x_buffer[i] = lb_vec[i]
                elseif xi > ub_vec[i]
                    viol = xi - ub_vec[i]
                    violation_sq += viol * viol
                    x_buffer[i] = ub_vec[i]
                else
                    x_buffer[i] = xi
                end
            end

            f_count_ref[] += 1
            f_val = tf_inner.f_original(x_buffer)           # ← immer innerhalb!

            return f_val + (T(L2_BOUND_PENALTY) / T(2)) * violation_sq
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # Gradient: clamp → eval inside → add linear penalty gradient
    # ─────────────────────────────────────────────────────────────────────
    grad_wrap! = let x_buffer = x_buffer,
        lb_vec = lb_vec,
        ub_vec = ub_vec,
        tf_inner = tf_fixed,
        grad_count_ref = shared_grad_count

        function penalty_gradient!(g::AbstractVector{T}, x::AbstractVector{T}) where T
            if any(isnan, x)
                fill!(g, T(NaN))
                return nothing
            end

            if any(isinf, x)
                fill!(g, T(Inf))
                return nothing
            end

            @inbounds @simd for i in eachindex(x)
                x_buffer[i] = clamp(x[i], lb_vec[i], ub_vec[i])
            end

            grad_count_ref[] += 1
            grad_temp = tf_inner.grad_original(x_buffer)     # ← immer innerhalb!
            copyto!(g, grad_temp)

            @inbounds @simd for i in eachindex(g, x)
                xi = x[i]
                if xi < lb_vec[i]
                    g[i] += T(L2_BOUND_PENALTY) * (xi - lb_vec[i])   # negativ → zieht hoch
                elseif xi > ub_vec[i]
                    g[i] += T(L2_BOUND_PENALTY) * (xi - ub_vec[i])   # positiv → zieht runter
                end
                # inner: +0
            end

            return nothing
        end
    end

    grad_wrap = let grad_wrap! = grad_wrap!
        (x) -> begin
            g = similar(x)
            grad_wrap!(g, x)
            return g
        end
    end

    # Metadata – gleiche Logik wie beim L1-Wrapper
    new_meta = deepcopy(tf_fixed.meta)
    new_meta[:name] = string(tf_fixed.meta[:name], "_l2_constrained")

    original_desc = get(new_meta, :description, "")
    new_meta[:description] = isempty(original_desc) ?
        "Domain-safe hard box constraints via L2 quadratic penalty (ρ = $(L2_BOUND_PENALTY))" :
        original_desc * " [Domain-safe L2 quadratic penalty (ρ = $(L2_BOUND_PENALTY))]"

    filter!(p -> p ≠ "bounded", new_meta[:properties])

    new_meta[:lb] = nothing
    new_meta[:ub] = nothing
    new_meta[:constraint_method] = "Domain-safe L2 quadratic penalty"
    new_meta[:penalty_coefficient] = L2_BOUND_PENALTY
    new_meta[:original_bounds] = (lb=lb_vec, ub=ub_vec)

    return TestFunction(
        f_wrap,
        grad_wrap,
        new_meta,
        shared_f_count,
        shared_grad_count
    )
end

export with_l2_box_constraints