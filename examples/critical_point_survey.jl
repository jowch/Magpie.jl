# # Critical-point survey with a derivative GP
#
# This vignette finds **and classifies all critical points** (minima, maxima, saddles) of a
# function `f : ℝ² → ℝ` from **f-values only** — by treating the gradient `∇f` as a
# Gaussian-process-derived vector field and using an active-learning loop to localize its zeros
# `{x : ∇f(x) = 0}`. Each point is labelled by its **Morse index** (number of negative Hessian
# eigenvalues): `0 → min`, `d → max`, otherwise `saddle`.
#
# Rather than chase a single headline number, we *watch the method work* — the GP posterior mean
# and the acquisition surface evolving as data arrives — and use **learning curves** to see how
# recovery grows with budget, and how that depends on the size of the search domain.

# ## Setup

ENV["GKSwstype"] = "100"  ## GR headless rendering (no display required)

using Magpie, KernelFunctions, LinearAlgebra, Random
using Plots; gr()

# ## The function and its landscape
#
# `f(x) = (x₁²−1)² + (x₂²−1)²` is a separable double-well with **9 critical points** — 4 minima
# `(±1,±1)` (`f=0`), 1 maximum `(0,0)` (`f=2`), and 4 saddles `(±1,0),(0,±1)` (`f=1`) — all
# inside `[-1,1]²`. Here is the surface and where those points sit.

f(x) = (x[1]^2 - 1)^2 + (x[2]^2 - 1)^2

truth      = vcat([[a, b] for a in (-1.0, 1.0) for b in (-1.0, 1.0)],     # minima
                  [[0.0, 0.0]],                                            # maximum
                  [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])      # saddles
true_kinds = vcat(fill(:min, 4), [:max], fill(:saddle, 4))

xl = range(-2, 2; length = 100); yl = range(-2, 2; length = 100)
Zf = [f([xi, yj]) for yj in yl, xi in xl]

plt_contour = contourf(xl, yl, Zf; levels = 12, c = :viridis, colorbar_title = "f(x)",
    aspect_ratio = :equal, xlims = (-2, 2), ylims = (-2, 2),
    xlabel = "x₁", ylabel = "x₂", title = "f(x) = (x₁²−1)² + (x₂²−1)²")
for (kind, mk, col, lab) in ((:min, :circle, :white, "min"), (:max, :rect, :red, "max"),
                             (:saddle, :diamond, :orange, "saddle"))
    idx = findall(==(kind), true_kinds)
    scatter!(plt_contour, [truth[i][1] for i in idx], [truth[i][2] for i in idx];
        m = mk, ms = 8, mc = col, msw = 1.2, label = lab)
end
plt_surface = surface(xl, yl, Zf; c = :viridis, colorbar = false, camera = (40, 35),
    xlabel = "x₁", ylabel = "x₂", zlabel = "f", title = "the four wells (minima)")
plot(plt_contour, plt_surface; layout = (1, 2), size = (900, 400))

# ## The derivative GP
#
# Conditioning an [`ExactGP`](@ref) on `f`-values induces a joint Gaussian over `f` *and* its
# derivatives. [`grad_predict`](@ref) returns, at any query `x`, the posterior **gradient mean**
# `μ∇`, the per-component **gradient variance** `Σ∇`, and the posterior-**mean Hessian** `H̄`.
# It is kernel-generic (autodiff through the posterior mean and kernel), so it works for any
# differentiable kernel — no hand-derived derivative blocks. The critical points are the zeros
# of `μ∇`; their Morse index comes from `H̄`.

# ## The acquisition: `GradStraddle` + `LocalPenalization`
#
# [`GradStraddle`](@ref) is a component-wise *straddle* on the gradient, `Σᵢ[β√(Σ∇)ᵢ − |μ∇ᵢ|]`:
# high where every `∂f/∂xᵢ` is **near zero and still uncertain**, so the loop samples toward the
# zeros of `∇f`. On its own the deterministic argmax tends to **re-query one spot** in the
# informative region — and stacking samples there never resolves that spot's gradient (a gradient
# needs spatially *spread* samples). [`LocalPenalization`](@ref) wraps it with a soft penalty
# around already-observed points (radius `c·ℓ`, a lengthscale-scaled adaptation of González et al.
# 2016): in the region the loop cares about it spreads the queries enough to resolve the field.
# (In the far, near-uniform-uncertainty outskirts it still revisits points — that budget is spent
# on genuine exploration, which is harmless here.)

# ## Search domain
#
# We search the **large** box `[-6,6]²` — the informative `[-1,1]²` centre is only ~1/36 of the
# area. This is deliberately the regime where *where you spend the budget* matters; the learning
# curves at the end contrast it with the natural `[-2,2]²` domain.

box = Box([-6.0, -6.0], [6.0, 6.0])

# ## Extract & classify
#
# After (or during) the run, recover the critical points by **multi-start Newton** on the GP-mean
# gradient field from a regular grid of seeds (Levenberg–Marquardt damping keeps the step
# well-defined through near-singular Hessians), keep the converged & deduplicated iterates, and
# classify each by Morse index.

function newton_polish(g, x0, box; iters = 12, λ = 1e-6)
    x = collect(float.(x0)); μ∇, _, H = grad_predict(g, x)
    for _ in 1:iters
        norm(μ∇) < 1e-7 && break
        x = clamp.(x .- (Symmetric(H) + λ*I) \ μ∇, box.lb, box.ub)
        μ∇, _, H = grad_predict(g, x)
    end
    return (x, μ∇, H)
end

## Morse classification from a (mean) Hessian: index = # negative eigenvalues.
classify(H; ε = 1e-3) = (λ = eigvals(Symmetric(H));
    any(<(ε), abs.(λ))       ? :unclassified :
    count(<(0), λ) == 0      ? :min :
    count(<(0), λ) == length(λ) ? :max : :saddle)

function critical_points(g, box; per_axis = 30, ε = 1e-3, restol = 1e-2)
    polished = [newton_polish(g, x, box) for x in grid_points(box; per_axis = per_axis)]
    conv = filter(p -> norm(p[2]) < restol, polished)
    uniq = unique(p -> round.(p[1]; digits = 1), conv)
    return map(uniq) do (x, _, H)
        (point = x, kind = classify(H; ε = ε))
    end
end

# ## Run the active loop, snapshotting as we go
#
# We drive an [`ActiveLearner`](@ref) by hand so we can capture the GP and the acquisition every
# few steps for the animations below. Seed with 20 random observations, wrap the acquisition with
# [`LocalPenalization`](@ref) over the learner's **live** history (`observe!` mutates `al.Xs` in
# place, so the penalty tracks every new observation), then run 120 acquisition steps, refitting
# `ℓ` and `σ_f²` every 10.

Random.seed!(1)
al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 1.2); noise = 1e-4),
                   GradStraddle(β = 1.96))
for x in [12 .* rand(2) .- 6 for _ in 1:20]; observe!(al, x, f(x)); end
al.acq = LocalPenalization(al.acq, al.Xs; c = 0.5)

snaps = NamedTuple[]
for t in 1:120
    x = acquire(al; over = box)
    if t % 12 == 1 || t == 120                       ## ~11 snapshots for the animations
        push!(snaps, (n = length(al.Xs), g = al.gp, Xs = copy(al.Xs), next = copy(x)))
    end
    observe!(al, x, f(x))
    t % 10 == 0 && fit!(al)
end
gp = posterior_gp(al)

# ## Animation 1 — the GP posterior mean learning the landscape
#
# Zoomed to `[-2,2]²`, the GP mean (with the queries so far) sharpens into the four-well shape as
# observations accrue.

xz = range(-2, 2; length = 60); yz = range(-2, 2; length = 60)
anim_mean = @animate for s in snaps
    Zm = [predmean(s.g, [xi, yj]) for yj in yz, xi in xz]
    inwin = [p for p in s.Xs if all(-2 .≤ p .≤ 2)]
    p = contourf(xz, yz, Zm; levels = 12, c = :viridis, colorbar = false,
        aspect_ratio = :equal, xlims = (-2, 2), ylims = (-2, 2),
        xlabel = "x₁", ylabel = "x₂", title = "GP posterior mean  (n=$(s.n))", size = (440, 420))
    isempty(inwin) || scatter!(p, [q[1] for q in inwin], [q[2] for q in inwin];
        ms = 3.5, mc = :white, msw = 0.5, label = "")
    scatter!(p, [t[1] for t in truth], [t[2] for t in truth];
        m = :star5, ms = 7, mc = :gold, msw = 0.5, label = "")
end
gif(anim_mean; fps = 4)

# ## Animation 2 — the acquisition surface over the full domain
#
# Over all of `[-6,6]²`, the acquisition (the live `GradStraddle` + `LocalPenalization`) shows
# where the loop wants to sample next (the bright argmax marked in red), and the dark "holes" the
# penalty carves around visited points. Watch the bright mass concentrate toward the centre.

xa = range(-6, 6; length = 48); ya = range(-6, 6; length = 48)
anim_acq = @animate for s in snaps
    acqf = LocalPenalization(GradStraddle(β = 1.96), s.Xs; c = 0.5)   ## frozen to this snapshot
    Za = [acqf(s.g, [xi, yj]) for yj in ya, xi in xa]
    p = heatmap(xa, ya, Za; c = :magma, colorbar = false,
        aspect_ratio = :equal, xlims = (-6, 6), ylims = (-6, 6),
        xlabel = "x₁", ylabel = "x₂", title = "acquisition surface  (n=$(s.n))", size = (440, 420))
    plot!(p, [-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]; lc = :white, ls = :dash, lw = 1, label = "")
    scatter!(p, [s.next[1]], [s.next[2]]; m = :star5, ms = 8, mc = :red, msw = 0.5, label = "next")
end
gif(anim_acq; fps = 4)

# ## Recovered & classified critical points
#
# The posterior gradient-norm `‖μ∇‖` is the surface Newton descends; its zeros are the candidate
# critical points. Zoomed to the informative centre, every minimum, the maximum, and every saddle
# is recovered and labelled correctly.

cps = critical_points(gp, box)

xg = range(-2, 2; length = 70); yg = range(-2, 2; length = 70)
Zgn = [norm(grad_predict(gp, [xi, yj])[1]) for yj in yg, xi in xg]
plt_recovered = heatmap(xg, yg, Zgn; c = :magma, colorbar_title = "‖μ∇‖",
    aspect_ratio = :equal, xlims = (-2, 2), ylims = (-2, 2),
    xlabel = "x₁", ylabel = "x₂", title = "Recovered & classified critical points")
for (kind, mk, col, lab) in ((:min, :circle, :cyan, "min"), (:max, :rect, :red, "max"),
                             (:saddle, :diamond, :yellow, "saddle"))
    P = [c.point for c in cps if c.kind == kind && all(-2 .≤ c.point .≤ 2)]
    isempty(P) || scatter!(plt_recovered, [p[1] for p in P], [p[2] for p in P];
        m = mk, ms = 8, mc = col, msw = 1, label = lab)
end
plt_recovered

# !!! note "Spurious points in under-sampled regions"
#     `critical_points` also returns a handful of spurious near-zeros out in the sparsely sampled
#     `[-6,6]²` outskirts (where the GP mean is nearly flat). They are filtered above by keeping
#     only points in `[-2,2]²`; on a real problem, restrict the survey to the region of interest
#     or prune by posterior gradient *variance*.

# ## Learning curves — resolved-of-9, tracked after every observation
#
# Rather than re-run to fixed budgets, we walk **one** incremental loop and, after *each*
# observation, tally how many of the 9 true critical points the GP has **resolved**: a Newton step
# on the GP-mean gradient, started at the true location, stays put (small residual gradient) and
# the mean-Hessian gives the right Morse type. That check is cheap (one polish per true point), so
# the tally updates every step — giving a dense curve from a single pass instead of separate runs.

function n_resolved(g, box)
    count(zip(truth, true_kinds)) do (p, k)
        x, μ∇, H = newton_polish(g, p, box; iters = 8)
        norm(x .- p) < 0.25 && norm(μ∇) < 1e-2 && classify(H) == k
    end
end

## active: random seed phase, then acquisition phase; tally after every observation
function resolve_curve_active(box; ℓ0, seed, nsteps)
    Random.seed!(seed); L = box.ub[1]
    al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), ℓ0); noise = 1e-4), GradStraddle(β = 1.96))
    ns = Int[]; tally = Int[]
    for _ in 1:20
        x = 2L .* rand(2) .- L; observe!(al, x, f(x))
        push!(ns, length(al.Xs)); push!(tally, n_resolved(al.gp, box))
    end
    al.acq = LocalPenalization(al.acq, al.Xs; c = 0.5)
    for t in 1:nsteps
        x = acquire(al; over = box); observe!(al, x, f(x)); t % 10 == 0 && fit!(al)
        push!(ns, length(al.Xs)); push!(tally, n_resolved(al.gp, box))
    end
    (ns, tally)
end

## random: add one uniform point at a time (incremental update); tally after each
function resolve_curve_random(box; ℓ0, seed, n)
    Random.seed!(seed); L = box.ub[1]
    g = ExactGP(with_lengthscale(SqExponentialKernel(), ℓ0); noise = 1e-4)
    ns = Int[]; tally = Int[]
    for t in 1:n
        x = 2L .* rand(2) .- L; g = update(g, [x], [f(x)]); t % 10 == 0 && (g = Magpie.fit(g))
        push!(ns, t); push!(tally, n_resolved(g, box))
    end
    (ns, tally)
end

small = Box([-2.0, -2.0], [2.0, 2.0])
na_s, ta_s = resolve_curve_active(small; ℓ0 = 0.6, seed = 1, nsteps = 120)
nr_s, tr_s = resolve_curve_random(small; ℓ0 = 0.6, seed = 7, n = 140)
na_l, ta_l = resolve_curve_active(box;   ℓ0 = 1.2, seed = 1, nsteps = 120)
nr_l, tr_l = resolve_curve_random(box;   ℓ0 = 1.2, seed = 7, n = 140)

plt_lc = plot(na_l, ta_l; lw = 2, lc = :steelblue, label = "active, [-6,6]²",
    xlabel = "observations", ylabel = "critical points resolved (of 9)",
    title = "Learning curves (tracked every observation)", ylims = (-0.3, 9.3),
    legend = :bottomright, size = (580, 400))
plot!(plt_lc, nr_l, tr_l; lw = 2, ls = :dash, lc = :gray,       label = "random, [-6,6]²")
plot!(plt_lc, na_s, ta_s; lw = 2,            lc = :seagreen,    label = "active, [-2,2]²")
plot!(plt_lc, nr_s, tr_s; lw = 2, ls = :dash, lc = :darkorange, label = "random, [-2,2]²")

# ## Takeaway
#
# The derivative-GP pipeline — [`grad_predict`](@ref), the [`GradStraddle`](@ref) +
# [`LocalPenalization`](@ref) acquisition, multi-start-Newton extraction, and Morse classification
# — recovers and labels all nine critical points from f-values alone, and the animations show *how*:
# the GP mean learns the landscape while the acquisition concentrates the budget on it.
#
# The learning curves make the honest point about *when* active learning helps. On the natural
# `[-2,2]²` domain, random sampling is already a strong baseline — both curves climb to 9 quickly,
# and the active loop has little to add. On the large `[-6,6]²` domain, where most of the space is
# uninformative, random wastes its budget and lags while the active loop still concentrates on the
# centre. Active learning earns its keep when the **budget is scarce relative to the domain**, not
# as a blanket improvement.

using Test                                                                         #src
@test length(filter(c -> any(p -> isapprox(c.point, p; atol=0.25), truth), cps)) ≥ 6   #src
@test count(c -> c.kind == :min, cps) ≥ 3                                           #src
@test ta_l[end] ≥ tr_l[end]    ## active resolves ≥ random on the large domain       #src
@test tr_s[end] ≥ 6            ## random is already strong on the small domain        #src
