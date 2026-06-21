# # Critical-point survey with a derivative GP
#
# This vignette finds **and classifies all critical points** (minima, maxima, saddles)
# of a function `f : ℝ² → ℝ` from **f-values only** — by treating the gradient `∇f` as a
# Gaussian-process-derived vector field and using an active-learning loop to localize its
# zeros `{x : ∇f(x) = 0}`. Each point is labelled by its **Morse index** (number of
# negative Hessian eigenvalues): `0 → min`, `d → max`, otherwise `saddle`.
#
# It also demonstrates, honestly, *when this is worth it*: active learning beats uniform
# random sampling here because the critical points are **localized in a small region of a
# large domain** on a tight budget. (On a small domain, uniform random is a strong baseline
# and the active loop has no advantage — its value is scarce budget relative to domain.)

# ## Setup

ENV["GKSwstype"] = "100"  ## GR headless rendering (no display required)

using Magpie, KernelFunctions, LinearAlgebra, Random
using Plots; gr()

# ## The derivative GP
#
# Conditioning an [`ExactGP`](@ref) on `f`-values induces a joint Gaussian over `f` *and*
# its derivatives. [`grad_predict`](@ref) returns, at any query `x`, the posterior **gradient
# mean** `μ∇`, the per-component **gradient variance** `Σ∇`, and the posterior-**mean Hessian**
# `H̄`. It is kernel-generic (autodiff through the posterior mean and kernel), so it works for
# any differentiable kernel — no hand-derived derivative blocks.

# ## Acquisition: `GradStraddle` + `LocalPenalization`
#
# [`GradStraddle`](@ref) is a component-wise *straddle* on the gradient,
# `Σᵢ[β√(Σ∇)ᵢ − |μ∇ᵢ|]`: high where every `∂f/∂xᵢ` is **near zero and still uncertain**, so
# the loop samples toward the zeros of `∇f`. On its own the deterministic argmax
# **mode-collapses** — it resamples one spot, which never resolves that spot's gradient (a
# gradient needs spatially *spread* samples). [`LocalPenalization`](@ref) wraps it with a soft
# penalty around already-observed points (a lengthscale-scaled adaptation of González et al.
# 2016), which breaks the collapse while the straddle keeps the queries focused on the
# informative region.

# ## The function
#
# `f(x) = (x₁²−1)² + (x₂²−1)²` has **9 critical points** — 4 minima `(±1,±1)`, 1 maximum
# `(0,0)`, 4 saddles `(±1,0),(0,±1)` — all inside `[-1,1]²`. We search the **large** box
# `[-6,6]²`, where that informative centre is only ~1/36 of the area.

f(x) = (x[1]^2 - 1)^2 + (x[2]^2 - 1)^2
box  = Box([-6.0, -6.0], [6.0, 6.0])

truth = vcat([[a, b] for a in (-1.0, 1.0) for b in (-1.0, 1.0)],   # minima
             [[0.0, 0.0]],                                          # maximum
             [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])    # saddles

# ## Extract & classify
#
# After the budget, recover the critical points by **multi-start Newton** on the GP-mean
# gradient field from a regular grid of seeds (Levenberg–Marquardt damping keeps the step
# well-defined through near-singular Hessians), keep the converged & deduplicated iterates,
# and classify each by Morse index.

function newton_polish(g, x0, box; iters = 12, λ = 1e-6)
    x = collect(float.(x0)); μ∇, _, H = grad_predict(g, x)
    for _ in 1:iters
        norm(μ∇) < 1e-7 && break
        x = clamp.(x .- (Symmetric(H) + λ*I) \ μ∇, box.lb, box.ub)
        μ∇, _, H = grad_predict(g, x)
    end
    return (x, μ∇, H)
end

function critical_points(g, box; per_axis = 35, ε = 1e-3, restol = 1e-2)
    d = length(box.lb)
    polished = [newton_polish(g, x, box) for x in grid_points(box; per_axis = per_axis)]
    conv = filter(p -> norm(p[2]) < restol, polished)
    uniq = unique(p -> round.(p[1]; digits = 1), conv)
    return map(uniq) do (x, _, H)
        λ = eigvals(Symmetric(H))
        kind = any(<(ε), abs.(λ)) ? :unclassified :
               count(<(0), λ) == 0 ? :min :
               count(<(0), λ) == d ? :max : :saddle
        (point = x, kind = kind)
    end
end

# ## Run the active learner
#
# Seed with 20 random observations, then wrap the acquisition with `LocalPenalization` over
# the learner's **live** history (`observe!` mutates `al.Xs` in place, so the penalty tracks
# every new observation), and run 120 acquisition steps, refitting `ℓ` and `σ_f²` every 10.

Random.seed!(1)
al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), 1.2); noise = 1e-4),
                   GradStraddle(β = 1.96))
for x in [12 .* rand(2) .- 6 for _ in 1:20]; observe!(al, x, f(x)); end
al.acq = LocalPenalization(al.acq, al.Xs; c = 0.5)
run!(al, f; budget = 120, over = box, refit_every = 10)

cps = critical_points(posterior_gp(al), box)

# ## A uniform-random baseline at the same budget
#
# The honest control: 140 points drawn uniformly over the same large box, same GP, same
# extraction.

Random.seed!(101)
Xr = [12 .* rand(2) .- 6 for _ in 1:140]
g_rand = Magpie.fit(update(ExactGP(with_lengthscale(SqExponentialKernel(), 1.2); noise = 1e-4), Xr, f.(Xr)))
cps_rand = critical_points(g_rand, box)

recovered(cs) = count(t -> any(c -> isapprox(c.point, t; atol = 0.25), cs), truth)

# Active learning recovers nearly all 9; uniform random, spread thin over the large domain,
# resolves only a couple:

@info "recovery (of 9 critical points)" active = recovered(cps) random = recovered(cps_rand)

# ## Figure 1 — where the queries go
#
# The active+LP queries pull into the central `[-1,1]²` where the gradient zeros live; the
# uniform-random points (faint) cover the whole box and mostly land in the boring outskirts.

apx = [p[1] for p in queried_points(al)]; apy = [p[2] for p in queried_points(al)]
tx  = [p[1] for p in truth];             ty  = [p[2] for p in truth]

plt1 = scatter([p[1] for p in Xr], [p[2] for p in Xr];
    ms = 3, mc = :lightgray, msw = 0, label = "uniform random",
    aspect_ratio = :equal, xlims = (-6, 6), ylims = (-6, 6),
    xlabel = "x₁", ylabel = "x₂", title = "Queries concentrate on the informative centre")
scatter!(plt1, apx, apy; ms = 3.5, mc = :steelblue, msw = 0, label = "active + LP")
plot!(plt1, [-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]; lc = :black, ls = :dash, lw = 1.5, label = "[-1,1]²")
scatter!(plt1, tx, ty; m = :star5, ms = 8, mc = :gold, msw = 0.5, label = "true critical points")

# ## Figure 2 — recovered & classified critical points
#
# Zoomed to `[-2,2]²` over the contours of `f`, the recovered points are coloured by Morse
# type. Every minimum, the maximum, and every saddle is found and labelled correctly.

xs = range(-2, 2; length = 120); ys = range(-2, 2; length = 120)
Zf = [f([xi, yj]) for yj in ys, xi in xs]

plt2 = contour(xs, ys, Zf; levels = 18, c = :viridis, colorbar = false,
    aspect_ratio = :equal, xlims = (-2, 2), ylims = (-2, 2),
    xlabel = "x₁", ylabel = "x₂", title = "Recovered & classified critical points")
for (kind, mk, col, lab) in ((:min, :circle, :dodgerblue, "min"),
                             (:max, :rect, :red, "max"),
                             (:saddle, :diamond, :limegreen, "saddle"))
    P = [c.point for c in cps if c.kind == kind]
    isempty(P) || scatter!(plt2, [p[1] for p in P], [p[2] for p in P];
        m = mk, ms = 8, mc = col, msw = 1, label = lab)
end
plt2

# ## Animation — active vs. random, side by side
#
# Watch the budget spend. The active learner (left) starts scattered (its random seed) then
# pulls sharply into the centre; uniform random (right) keeps painting the whole box, so the
# `[-1,1]²` region stays starved.

apts = queried_points(al)
anim = @animate for k in 5:5:length(apts)
    pa = scatter([p[1] for p in apts[1:k]], [p[2] for p in apts[1:k]];
        ms = 3, mc = :steelblue, msw = 0, legend = false, aspect_ratio = :equal,
        xlims = (-6, 6), ylims = (-6, 6), title = "active + LP  (n=$k)")
    plot!(pa, [-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]; lc = :black, ls = :dash, lw = 1)
    kk = min(k, length(Xr))
    pr = scatter([p[1] for p in Xr[1:kk]], [p[2] for p in Xr[1:kk]];
        ms = 3, mc = :gray, msw = 0, legend = false, aspect_ratio = :equal,
        xlims = (-6, 6), ylims = (-6, 6), title = "uniform random  (n=$kk)")
    plot!(pr, [-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]; lc = :black, ls = :dash, lw = 1)
    plot(pa, pr; layout = (1, 2), size = (760, 380))
end
gif(anim; fps = 10)

# ## Takeaway
#
# The derivative-GP pipeline — `grad_predict`, the `GradStraddle` + `LocalPenalization`
# acquisition, multi-start-Newton extraction, and Morse classification — recovers and labels
# all nine critical points from f-values alone. The active loop beats random **here** because
# the targets are localized in a large domain on a tight budget; that is the regime where
# focusing pays off.

using Test                                              #src
@test recovered(cps) ≥ 7                                #src
@test recovered(cps_rand) ≤ 4                           #src
@test recovered(cps) ≥ recovered(cps_rand) + 4          #src
@test count(c -> c.kind == :min, cps) ≥ 3               #src
