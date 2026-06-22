# # Müller–Brown: critical points and the transition state, from f-values only
#
# The [Müller–Brown potential](https://doi.org/10.1007/BF00547608) (Müller & Brown,
# *Theor. Chim. Acta* **53**, 75 (1979)) is the standard 2-D benchmark for **transition-state
# search**: a sum of four Gaussians with three minima (metastable states) joined by two index-1
# saddles (the transition states on the minimum-energy path between basins). Here we treat it as
# an *expensive* field `f : ℝ² → ℝ` and, from **f-values only**, (1) recover and classify *all*
# its critical points with a derivative GP, and (2) find **the** transition state between two known
# minima with a targeted active-learning loop.
#
# Surrogate-based transition-state search is a real method class — Gaussian-process surrogates
# drive nudged-elastic-band ([Koistinen et al. 2017](https://doi.org/10.1063/1.4986787)) and direct
# saddle ([Denzel & Kästner 2018](https://doi.org/10.1063/1.5017103)) optimizers in computational
# chemistry, precisely because each true `f` (an electronic-structure energy) is costly. This
# vignette is a miniature of that workflow on the canonical toy surface.

# ## Setup

ENV["GKSwstype"] = "100"  ## GR headless rendering (no display required)

using Magpie, KernelFunctions, LinearAlgebra, Random
using Statistics: mean, std
using Plots; gr()

# ## The potential and its landscape
#
# The four Gaussians give a deep right basin, a shallow central basin, and a high-left basin,
# separated by two saddles. We work on the conventional window and clip the steep exponential
# walls for display (the true `f` reaches ~10³ at the corners).

const A  = (-200.0, -100.0, -170.0, 15.0)
const aa = (-1.0, -1.0, -6.5, 0.7);  const bb = (0.0, 0.0, 11.0, 0.6)
const cc = (-10.0, -10.0, -6.5, 0.7)
const x0 = (1.0, 0.0, -0.5, -1.0);   const y0 = (0.0, 0.5, 1.5, 1.0)

function mullerbrown(p)
    x, y = p[1], p[2]; s = 0.0
    for k in 1:4
        dx = x - x0[k]; dy = y - y0[k]
        s += A[k] * exp(aa[k]*dx^2 + bb[k]*dx*dy + cc[k]*dy^2)
    end
    s
end

box = Box([-1.5, -0.5], [1.0, 2.0])

# The two minima we will later bracket a transition state between, and the five true critical
# points (literature values) for scoring. `MB_min`→`MC_min` is the right-hand reaction; `S2` is
# the transition state on it.

const MA_min = [-0.558, 1.442]    # high-left minimum
const MB_min = [-0.050, 0.467]    # shallow central minimum
const MC_min = [ 0.623, 0.028]    # deep right minimum
const S1 = [-0.822, 0.624]        # saddle between MA and MB
const S2 = [ 0.212, 0.293]        # saddle between MB and MC  (our target TS)

truth      = [MA_min, MB_min, MC_min, S1, S2]
true_kinds = [:min, :min, :min, :saddle, :saddle]

xl = range(-1.5, 1.0; length = 100); yl = range(-0.5, 2.0; length = 100)
Vclip(p) = min(mullerbrown(p), 150.0)                 ## clip walls for a readable contour
Zf = [Vclip([xi, yj]) for yj in yl, xi in xl]

plt_contour = contourf(xl, yl, Zf; levels = 20, c = :viridis, colorbar_title = "V (clipped)",
    aspect_ratio = :equal, xlims = (-1.5, 1.0), ylims = (-0.5, 2.0),
    xlabel = "x", ylabel = "y", title = "Müller–Brown potential")
for (kind, mk, col, lab) in ((:min, :circle, :white, "minima"), (:saddle, :diamond, :orange, "saddles (TS)"))
    idx = findall(==(kind), true_kinds)
    scatter!(plt_contour, [truth[i][1] for i in idx], [truth[i][2] for i in idx];
        m = mk, ms = 8, mc = col, msw = 1.2, label = lab)
end
plt_surface = surface(xl, yl, Zf; c = :viridis, colorbar = false, camera = (35, 45),
    xlabel = "x", ylabel = "y", zlabel = "V", title = "the basins and barriers")
plot(plt_contour, plt_surface; layout = (1, 2), size = (960, 410))

# The minima are metastable chemical states; the saddles are the transition states that gate the
# rate of hopping between them along the minimum-energy path. Finding and characterising them is
# the central task of reaction-path analysis.

# ## The derivative GP, and taming the dynamic range
#
# Conditioning an [`ExactGP`](@ref) on `f`-values induces a joint Gaussian over `f` *and* its
# derivatives; [`grad_predict`](@ref) returns the posterior gradient mean `μ∇`, per-component
# gradient variance `Σ∇`, and the posterior-mean Hessian `H̄` at any query, by autodiff through the
# posterior mean — kernel-generic, no hand-derived derivative blocks. Critical points are the zeros
# of `μ∇`; their Morse index (number of negative `H̄` eigenvalues) labels them via [`classify`](@ref).
#
# A stationary kernel needs a single feature scale, but the raw potential spans ~10³ over the box
# (deep wells, exp walls). We **clip** at a ceiling and **z-score** so the surface the GP sees is
# ~unit-scale and well-conditioned for a fixed lengthscale — the standard surrogate-conditioning
# move for stiff potentials.

const CEIL = 200.0
let g = grid_points(box; per_axis = 50), v = min.(mullerbrown.(g), CEIL)
    global const _μ = mean(v); global const _σ = std(v)
end
mbt(p) = (min(mullerbrown(p), CEIL) - _μ) / _σ        ## clipped, z-scored — what we condition on

const ℓ0 = 0.3; const NOISE = 1e-3
mbkernel() = with_lengthscale(SqExponentialKernel(), ℓ0)
buildgp(pts) = update(ExactGP(mbkernel(); noise = NOISE), pts, mbt.(pts))

# We fix the lengthscale to the feature scale rather than refitting: at the scarce sample sizes
# here, marginal-likelihood fitting drives `ℓ` up and washes the wells out (see [`Magpie.fit`](@ref)'s
# `ℓ_prior`). `ℓ ≈ 0.3` resolves all five basins.

# ## Extraction & classification from coverage data
#
# Recover the critical points by **multi-start Newton** on the GP-mean gradient field from a grid
# of seeds (Levenberg–Marquardt damping keeps the step defined through near-singular Hessians),
# keep converged & deduplicated iterates, and classify each by Morse index. We reuse the package's
# [`newton_polish`](@ref) and [`classify`](@ref). A near-flat, *under-sampled* region (the steep
# wall in the top-right corner here) also produces spurious gradient-zeros; we prune them by the
# posterior **gradient variance** `Σ∇` — genuine critical points sit where the field is pinned
# down by data (low `Σ∇`), spurious ones where it is not.

function critical_points(g, box; per_axis = 40, ε = 1e-3, restol = 1e-2, maxvar = 0.2)
    pol  = [newton_polish(g, x; box = box, iters = 20) for x in grid_points(box; per_axis = per_axis)]
    conv = filter(p -> norm(p[2]) < restol, pol)
    uniq = unique(p -> round.(p[1]; digits = 1), conv)
    cps  = map(uniq) do (x, _, H)
        (point = x, kind = classify(H; ε = ε), gvar = maximum(grad_predict(g, x)[2]))
    end
    filter(c -> c.gvar ≤ maxvar, cps)             ## drop spurious zeros in under-sampled regions
end

# Condition on a **space-filling sample** (an 11×11 grid; coverage, not active) and extract. All
# three minima and both saddles are recovered with the right Morse type.

cov_pts = grid_points(box; per_axis = 11)
gcov    = buildgp(cov_pts)
cps     = critical_points(gcov, box)

matcherr(p) = minimum(norm(c.point .- p) for c in cps; init = Inf)
@info "extraction from coverage data" n_samples=length(cov_pts) n_cps=length(cps) max_err=maximum(matcherr, truth)

xg = range(-1.5, 1.0; length = 90); yg = range(-0.5, 2.0; length = 90)
Zgn = [norm(grad_predict(gcov, [xi, yj])[1]) for yj in yg, xi in xg]
plt_recovered = heatmap(xg, yg, Zgn; c = :magma, colorbar_title = "‖μ∇‖",
    aspect_ratio = :equal, xlims = (-1.5, 1.0), ylims = (-0.5, 2.0),
    xlabel = "x", ylabel = "y", title = "Recovered & classified critical points")
scatter!(plt_recovered, [p[1] for p in cov_pts], [p[2] for p in cov_pts];
    ms = 2, mc = :white, msw = 0, alpha = 0.4, label = "samples")
for (kind, mk, col, lab) in ((:min, :circle, :cyan, "min"), (:saddle, :diamond, :yellow, "saddle"))
    P = [c.point for c in cps if c.kind == kind]
    isempty(P) || scatter!(plt_recovered, [p[1] for p in P], [p[2] for p in P];
        m = mk, ms = 8, mc = col, msw = 1, label = lab)
end
plt_recovered

# The zeros of `‖μ∇‖` (dark) sit on the true minima and saddles; multi-start Newton lands on each
# and the mean-Hessian types it correctly.

# ## An honest detour: does active learning *enumerate* them faster?
#
# A natural hope is that an active loop — querying where the gradient field is most uncertain
# ([`GradStraddle`](@ref) + [`LocalPenalization`](@ref)) — would recover all five critical points in
# *fewer* evaluations than blind random sampling. On this surface, at scarce budget, **it does not.**
# The measurement below (fixed `ℓ`, recovery counted at tolerance 0.1) shows random matching or
# beating active across `T = 20–40` evaluations.

function active_enum(; seed, T, ninit = 6, c = 0.5)
    Random.seed!(seed)
    Xs = [box.lb .+ (box.ub .- box.lb).*rand(2) for _ in 1:ninit]
    g  = buildgp(Xs); acq0 = GradStraddle(β = 1.96); acq = LocalPenalization(acq0, Xs; c = c)
    for _ in 1:(T - ninit)
        x = acquire(g, acq; over = box); g = update(g, [x], [mbt(x)]); push!(Xs, x)
        acq = LocalPenalization(acq0, Xs; c = c)
    end
    g
end
random_enum(; seed, T) = (Random.seed!(seed); buildgp([box.lb .+ (box.ub .- box.lb).*rand(2) for _ in 1:T]))
recov(g; atol = 0.1) = count(zip(truth, true_kinds)) do (p, k)
    any(c -> c.kind == k && norm(c.point .- p) < atol, critical_points(g, box; per_axis = 30))
end

enum_T = [20, 30, 40]
act_enum = [mean(recov(active_enum(; seed = s, T = T)) for s in 1:4) for T in enum_T]
rnd_enum = [mean(recov(random_enum(; seed = s, T = T)) for s in 1:4) for T in enum_T]
@info "scarce enumeration (mean recovered of 5, 4 seeds)" T=enum_T active=act_enum random=rnd_enum

# This is **extraction-limited**, not acquisition-limited. The Newton extractor needs spatial
# *coverage* to represent all five basins; concentrating the budget toward the gradient-zeros
# starves the basins the loop has not visited yet, so it recovers *fewer* points than a sample that
# spreads out. Blind enumeration is the wrong job for active learning here — which sets up the job
# that is right.

# ## The active-learning win: targeted transition-state search
#
# The real chemistry question is not "enumerate everything" but "given two known minima, find **the**
# transition state between them." That is where exploiting structure pays off. [`transition_state`](@ref)
# seeds a GP at the two minima plus a few points along the connecting segment, then iterates
# *predict-saddle (on the GP mean, by a min-mode walk) → evaluate true `f` → update* until the budget
# is spent — concentrating every evaluation on the one saddle of interest.

Random.seed!(1)
res = transition_state(mbt, MB_min, MC_min; kernel = mbkernel(), noise = NOISE,
                       box = box, budget = 12, nseed = 5, predictor = :minmode)
@info "targeted TS search" predicted=round.(res.saddle; digits=4) truth_S2=S2 err=round(norm(res.saddle .- S2); digits=4) kind=res.kind

# ### Convergence: targeted vs random, averaged over seeds
#
# Localization error to `S2` as evaluations accrue, **averaged over 8 seeds** (mean ± 1σ). The
# random baseline gets its most charitable shot: it extracts *every* saddle of its GP mean (raw
# multi-start Newton, no variance prune) and is scored on the one nearest `S2`; if it recovers no
# saddle that seed, the error is capped at a sentinel of `1.0` ("not found"). Targeted sits at
# ~0.03 throughout — flat and well below the 0.1 threshold — while random averages ~0.4: with no
# notion of *which* saddle it is after, it usually localizes the wrong one (or none).

const NTS = 8                          ## seeds to average the convergence curves over
const SENT = 1.0                       ## "saddle not found" sentinel (caps random error)

## every saddle of a GP mean, by raw multi-start Newton (no gradient-variance prune — the random
## baseline is credited with any saddle it stumbles on).
raw_saddles(g) = let pol = [newton_polish(g, x; box = box, iters = 20) for x in grid_points(box; per_axis = 30)]
    conv = filter(p -> norm(p[2]) < 1e-2, pol)
    uniq = unique(p -> round.(p[1]; digits = 1), conv)
    [x for (x, _, H) in uniq if classify(H) == :saddle]
end

## targeted: ‖predicted − S2‖ at each evaluation count, as a Dict n → err (history covers n = 7…25)
function targeted_hist(seed)
    Random.seed!(seed)
    res = transition_state(mbt, MB_min, MC_min; kernel = mbkernel(), noise = NOISE,
                           box = box, budget = 25, nseed = 5, predictor = :minmode)
    Dict(n => norm(xs .- S2) for (n, xs, _) in res.history)
end
## random: nearest-saddle error to S2 from T uniform samples, capped at the sentinel
function random_err(seed, T)
    Random.seed!(seed)
    X   = [box.lb .+ (box.ub .- box.lb).*rand(2) for _ in 1:T]
    sad = raw_saddles(buildgp(X))
    isempty(sad) ? SENT : min(SENT, minimum(norm(s .- S2) for s in sad))
end

th     = [targeted_hist(s) for s in 1:NTS]
ns     = sort(collect(keys(th[1])))
t_mean = [mean(d[n] for d in th) for n in ns]
t_std  = [std(d[n] for d in th)  for n in ns]
budgets = collect(7:2:25)
r_mat  = [random_err(s, T) for s in 1:NTS, T in budgets]      ## NTS × #budgets
r_mean = vec(mean(r_mat; dims = 1)); r_std = vec(std(r_mat; dims = 1))

@info "TS convergence (mean over $NTS seeds)" targeted_end=round(t_mean[end]; digits=3) random_mean=round(mean(r_mean); digits=3) random_notfound=count(==(SENT), r_mat)

plt_conv = plot(ns, t_mean; ribbon = t_std, fillalpha = 0.15, lw = 2, marker = :circle, mc = :steelblue,
    lc = :steelblue, label = "targeted (transition_state)",
    xlabel = "true f-evaluations", ylabel = "‖predicted − S2‖",
    title = "Transition-state localization (mean ± 1σ, $NTS seeds)",
    yscale = :log10, ylims = (0.015, 1.5), legend = :left, size = (580, 400))
plot!(plt_conv, budgets, r_mean; ribbon = r_std, fillalpha = 0.12, lw = 2, ls = :dash, marker = :diamond,
    mc = :darkorange, lc = :darkorange, label = "random + nearest-saddle (cap $SENT)")
hline!(plt_conv, [0.1]; lc = :gray, ls = :dot, lw = 1, label = "0.1 threshold")

# ### Watching the saddle walk converge
#
# Re-run the targeted loop and snapshot the predicted saddle after each evaluation, over the GP-mean
# contour. The seed minima are fixed; new true evaluations (white) accrue near the path, and the
# predicted saddle (red star) homes onto the true `S2` (gold).

function ts_snapshots(; seed, budget)
    Random.seed!(seed)
    d = MC_min .- MB_min; perp = [-d[2], d[1]]; perp ./= max(norm(perp), 1e-9)
    X = [copy(MB_min), copy(MC_min)]
    for i in 1:5
        t = i/6; base = MB_min .+ t.*d
        push!(X, clamp.(base .+ (0.12*(2rand() - 1)).*perp, box.lb, box.ub))
    end
    snaps = NamedTuple[]
    g = buildgp(X)
    mid = (MB_min .+ MC_min) ./ 2
    while true
        xs, _, _ = saddle_walk(g, mid; box = box)
        push!(snaps, (n = length(X), g = g, Xs = copy(X), pred = xs))
        length(X) ≥ budget && break
        push!(X, clamp.(xs, box.lb, box.ub)); g = buildgp(X)
    end
    snaps
end

snaps = ts_snapshots(; seed = 1, budget = 12)
xc = range(-1.5, 1.0; length = 70); yc = range(-0.5, 2.0; length = 70)
anim_ts = @animate for s in snaps
    Zm = [predmean(s.g, [xi, yj]) for yj in yc, xi in xc]
    p = contourf(xc, yc, Zm; levels = 18, c = :viridis, colorbar = false, aspect_ratio = :equal,
        xlims = (-1.5, 1.0), ylims = (-0.5, 2.0), xlabel = "x", ylabel = "y",
        title = "targeted TS search  (n=$(s.n))", size = (460, 430))
    scatter!(p, [q[1] for q in s.Xs], [q[2] for q in s.Xs]; ms = 4, mc = :white, msw = 0.4, label = "evals")
    scatter!(p, [MB_min[1], MC_min[1]], [MB_min[2], MC_min[2]]; m = :utriangle, ms = 7, mc = :cyan, msw = 0.6, label = "known minima")
    scatter!(p, [S2[1]], [S2[2]]; m = :star5, ms = 9, mc = :gold, msw = 0.5, label = "true S2")
    scatter!(p, [s.pred[1]], [s.pred[2]]; m = :star5, ms = 9, mc = :red, msw = 0.5, label = "predicted")
end
gif(anim_ts; fps = 2)

# ## Takeaway
#
# The derivative-GP machinery — [`grad_predict`](@ref), multi-start-Newton extraction, and
# [`classify`](@ref) Morse typing — is the **general** tool: from f-values on a coverage sample it
# recovers and labels every minimum and saddle of the Müller–Brown surface. Active learning, by
# contrast, does **not** speed up blind *enumeration* of all the critical points: that task is
# limited by the spatial coverage the extractor needs, so a budget concentrated by an acquisition
# starves the unvisited basins and random sampling matches or wins.
#
# Where active learning earns its keep is the **targeted** task that actually has structure to
# exploit: localizing *the* transition state between two known minima. [`transition_state`](@ref)
# spends every evaluation on that one saddle and reaches it in ~10 true `f`-evaluations, where the
# random baseline — having no notion of which saddle it is after — does not reliably find it at the
# same budget. The honest reading is mechanistic, not promotional: match the tool to the task —
# coverage-driven extraction for the global picture, active learning for the targeted search.

using Test                                                                              #src
@test count(zip(truth, true_kinds)) do (p, k)                                           #src
    any(c -> c.kind == k && norm(c.point .- p) < 0.1, cps)                              #src
end ≥ 4                                          ## extraction recovers ≥4/5 correctly typed #src
@test res.kind == :saddle                        ## the targeted prediction is an index-1 saddle #src
@test norm(res.saddle .- S2) < 0.1               ## …localized to the true S2                #src
@test t_mean[end] < 0.1 < mean(r_mean)          ## targeted ≪ 0.1 ≪ random (mean over seeds)  #src
