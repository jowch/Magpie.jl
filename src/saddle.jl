using LinearAlgebra: Symmetric, eigen, eigvals, norm, I

"""
    classify(H; ε=1e-3) -> Symbol

Morse classification of a critical point from its Hessian `H` (e.g. the `H̄` returned by
[`grad_predict`](@ref)). Returns `:min` (no negative eigenvalues), `:saddle` (an index-1
saddle: exactly one negative eigenvalue), `:max` (all negative), or `:unclassified`
when any eigenvalue is within `ε` of zero (degenerate / ill-determined curvature).

Only index-1 saddles map to `:saddle`; in `d≥3`, higher-index saddles (≥2 negative
eigenvalues but not all) fall to `:unclassified` — adequate for the 2-D use here.
"""
function classify(H; ε::Real = 1.0e-3)
    λ = eigvals(Symmetric(H))
    scale = maximum(abs, λ)
    scale < eps() && return :unclassified                 # ~flat: curvature undetermined
    any(<(ε * scale), abs.(λ)) && return :unclassified    # eigenvalue negligible *relative* to spectrum
    nneg = count(<(0), λ)
    nneg == 0 && return :min
    nneg == length(λ) && return :max
    return nneg == 1 ? :saddle : :unclassified
end

"""
    newton_polish(g::ExactGP, x0; box, iters=20, λ=1e-6, tol=1e-8) -> (x, μ∇, H)

Damped Newton iteration on the posterior-mean field of `g`, seeking a critical point
(`∇μ = 0`) from `x0`. Each step solves `(H̄ + λI) Δ = ∇μ` and clamps the update to `box`.
Returns the located point, its mean gradient, and mean Hessian. This converges to whatever
critical point is nearest the basin of `x0` (min, saddle, or max) — for an index-1 saddle
between two minima, seed it from their midpoint, or use [`saddle_walk`](@ref) when the
connecting path is curved and Newton is captured by a neighbouring extremum.
"""
function newton_polish(g::ExactGP, x0; box::Box, iters::Int = 20, λ::Real = 1.0e-6, tol::Real = 1.0e-8)
    x = collect(float.(x0))
    μ∇, _, H = grad_predict(g, x)
    for _ in 1:iters
        norm(μ∇) < tol && break
        x = clamp.(x .- (Symmetric(H) + λ * I) \ μ∇, box.lb, box.ub)
        μ∇, _, H = grad_predict(g, x)
    end
    res = norm(μ∇)
    return (; x, μ∇, H, residual = res, converged = res < tol && all(box.lb .< x) && all(x .< box.ub))
end

@doc raw"""
    saddle_walk(g::ExactGP, x0; box, iters=60, η=0.05, tol=1e-7) -> (x, μ∇, H)

Min-mode (dimer-like) walk to an **index-1 saddle** of the GP posterior-mean field, starting
from `x0`. At each step the effective force ascends the lowest-curvature mode while descending
all others: with `v` the eigenvector of the smallest eigenvalue of the mean Hessian `H̄`,

```math
F = -
abla\mu + 2\,(
abla\mu \cdot v)\,v, \qquad x \leftarrow \mathrm{clamp}(x + \eta F,\ \mathrm{box}).
```

Iterates until `‖∇μ‖ < tol` or `iters` is reached. Unlike a Newton step, the reflected force
is robust on curved reaction paths where Newton is captured by a neighbouring minimum — it is
the recommended default for [`transition_state`](@ref). Returns the located point, its mean
gradient `∇μ`, and mean Hessian `H̄` (classify with [`classify`](@ref)).
"""
function saddle_walk(g::ExactGP, x0; box::Box, iters::Int = 60, η::Real = 0.05, tol::Real = 1.0e-7)
    x = collect(float.(x0))
    for _ in 1:iters
        μ∇, _, H = grad_predict(g, x)
        norm(μ∇) < tol && break
        v = eigen(Symmetric(H)).vectors[:, 1]          # lowest-curvature mode
        F = -μ∇ + 2 * dot(μ∇, v) * v                    # reflect along v: ascend it, descend rest
        x = clamp.(x .+ η .* F, box.lb, box.ub)
    end
    μ∇, _, H = grad_predict(g, x)
    res = norm(μ∇)
    return (; x, μ∇, H, residual = res, converged = res < tol && all(box.lb .< x) && all(x .< box.ub))
end

# Seed set for transition_state: the two minima plus `nseed` points jittered perpendicular to
# the m1→m2 segment, so the GP mean carries the reaction-path structure between the basins.
function _ts_seed(m1, m2, box; nseed::Int = 5, jit::Real = 0.12, rng = Random.default_rng())
    d = m2 .- m1
    perp = [-d[2], d[1]]; perp = perp ./ max(norm(perp), 1.0e-9)
    pts = [collect(float.(m1)), collect(float.(m2))]
    for i in 1:nseed
        t = i / (nseed + 1); base = m1 .+ t .* d
        push!(pts, clamp.(base .+ (jit * (2 * rand(rng) - 1)) .* perp, box.lb, box.ub))
    end
    return pts
end

"""
    transition_state(f, m1, m2; kernel, noise=1e-3, box, budget=12, nseed=5,
                     predictor=:minmode, η=0.05) -> (; saddle, kind, g, history)

Targeted search for the index-1 transition state (saddle) of `f` lying between two **known**
minima `m1`, `m2`, within a small budget of true `f`-evaluations.

Seeds an [`ExactGP`](@ref) (with the given `kernel`/`noise`) at `m1`, `m2`, and `nseed` points
jittered along the `m1→m2` segment, then iterates: predict the saddle on the GP **mean**
(via [`saddle_walk`](@ref) from the segment midpoint for `predictor=:minmode`, or
[`newton_polish`](@ref) for `predictor=:newton`), evaluate the true `f` there, condition the GP
on that point, and repeat until `budget` total `f`-evaluations have been spent. The lengthscale
is **not** refit (deliberate: at the scarce `n` this loop runs in, MLE drives `ℓ` up and washes
out the saddle — pass a fixed feature-scale `kernel`).

Returns a named tuple: `saddle` (final predicted location), `kind` (its Morse type via
[`classify`](@ref); an index-1 saddle is `:saddle`), `g` (the fitted GP), and `history`
(a vector of `(neval, predicted_saddle, kind)` after each acquisition, for diagnostics).
"""
function transition_state(
        f, m1, m2; kernel, noise::Real = 1.0e-3, box::Box, budget::Int = 12,
        nseed::Int = 5, predictor::Symbol = :minmode, η::Real = 0.05,
        rng = Random.default_rng()
    )
    m1 = collect(float.(m1)); m2 = collect(float.(m2))
    length(m1) == length(box.lb) ||
        throw(ArgumentError("seed dimension $(length(m1)) ≠ box dimension $(length(box.lb))"))
    X = _ts_seed(m1, m2, box; nseed = nseed, rng = rng)
    g = update(ExactGP(kernel; noise = noise), X, [f(p) for p in X])   # one f-eval per seed
    mid = (m1 .+ m2) ./ 2
    predict(gp) = predictor === :newton ? newton_polish(gp, mid; box = box) :
        saddle_walk(gp, mid; box = box, η = η)
    history = Tuple{Int, Vector{Float64}, Symbol}[]
    r = predict(g)
    push!(history, (length(X), r.x, classify(r.H)))
    # Each round spends exactly one new f-eval (incremental Cholesky extension), so total
    # oracle calls = budget — re-evaluating the whole accumulated set would be quadratic.
    while length(X) < budget
        xnew = clamp.(r.x, box.lb, box.ub)
        push!(X, xnew)
        g = update(g, [xnew], [f(xnew)])
        r = predict(g)
        push!(history, (length(X), r.x, classify(r.H)))
    end
    return (saddle = r.x, kind = classify(r.H), converged = r.converged, residual = r.residual, g = g, history = history)
end
