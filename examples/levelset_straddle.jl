# # Level-set recovery with Straddle
#
# This vignette walks through a complete active-learning experiment:
# recovering the **unit-circle level set** `f(x) = ‖x‖ − 1 = 0`
# in two dimensions using the Straddle acquisition function.

# ## Setup

ENV["GKSwstype"] = "100"  ## GR headless rendering (no display required)

using Magpie, KernelFunctions, LinearAlgebra, Random
using Plots; gr()

# ## What is level-set estimation?
#
# Many scientific tasks ask not *where is the minimum of f?* but
# *where does f change sign?* — the boundary of a feasible region,
# a decision surface, or a phase boundary.  Bayesian optimisation
# frameworks such as Surrogates.jl and BayesianOptimization.jl target
# minima; Magpie's Straddle acquisition targets a **contour**.
#
# The Straddle criterion (Bryan et al. 2005) is:
#
# ```
# a(x) = β σ(x) − |μ(x) − h|
# ```
#
# where `h` is the target level, `μ(x)` and `σ(x)` are the GP posterior
# mean and standard deviation, and `β` is a balancing coefficient.
# High `σ` pulls the query toward unexplored space; the penalty `|μ − h|`
# pulls it toward the boundary.  Together they concentrate observations
# exactly where the classification decision is most uncertain —
# **near the level set, not at any extremum**.

# ## The function and GP model
#
# We use the unit-circle signed-distance function as a clean, visual test case.

Random.seed!(42)

f(x) = norm(x) - 1.0   ## signed distance to the unit circle

# Build an `ActiveLearner` with an `ExactGP` (squared-exponential kernel,
# lengthscale 0.5, small observation noise) and the Straddle acquisition
# targeting `h = 0`.

al = ActiveLearner(
    ExactGP(with_lengthscale(SqExponentialKernel(), 0.5); noise=1e-4),
    Straddle(h=0.0),
)

# ## Cold start
#
# Seed the GP with 10 random observations spread over the full domain
# `[−2, 2]²` so the model has a rough global picture before the
# acquisition-guided phase begins.

box = Box([-2.0, -2.0], [2.0, 2.0])

for x in [4 .* rand(2) .- 2 for _ in 1:10]
    observe!(al, x, f(x))
end

# ## Active-learning run
#
# `run!` selects 40 further queries, refitting the GP hyperparameters every
# 10 steps.  The first call to `fit!` after the cold start sets the kernel
# parameters; subsequent refits keep them current as new data arrive.

run!(al, f; budget=40, over=box, refit_every=10)

# ## Results

g = posterior_gp(al)

# ### Figure 1 — posterior mean with true contour and queries
#
# Build a 50×50 evaluation grid and compute the GP posterior mean at each
# point.  Overlay the true level set (`contour!` at `levels=[0.0]`) and mark
# every point the active learner queried.

xs = range(-2, 2; length=50)
ys = range(-2, 2; length=50)

Z_mean = [predmean(g, [xi, yj]) for yj in ys, xi in xs]
Z_true = [f([xi, yj]) for yj in ys, xi in xs]   ## the true signed-distance field

pts = queried_points(al)
px  = [p[1] for p in pts]
py  = [p[2] for p in pts]

plt1 = heatmap(xs, ys, Z_mean;
    title  = "GP posterior mean",
    xlabel = "x₁", ylabel = "x₂",
    c      = :RdBu, clim = (-2, 2),
    aspect_ratio = :equal, xlims = (-2, 2), ylims = (-2, 2),
)
## overlay the TRUE level set (zero contour of f), not the posterior's own contour
contour!(plt1, xs, ys, Z_true; levels=[0.0], lw=2, lc=:black, label="true h=0")
scatter!(plt1, px, py;
    ms     = 4,
    mc     = :white,
    msw    = 1,
    label  = "queries",
)

# ### Figure 2 — Straddle acquisition surface
#
# The acquisition `a(g, p)` reveals which regions the learner found most
# informative.  High values (warm colours) lie near the level set where the
# GP is still uncertain.

a = al.acq   ## the Straddle functor, already holds h=0.0

Z_acq = [a(g, [xi, yj]) for yj in ys, xi in xs]

plt2 = heatmap(xs, ys, Z_acq;
    title  = "Straddle acquisition",
    xlabel = "x₁", ylabel = "x₂",
    c      = :viridis,
    aspect_ratio = :equal, xlims = (-2, 2), ylims = (-2, 2),
)

# ## Hidden assertions (run when this file is executed directly; stripped from the rendered page)

using Test  #src
grid = grid_points(box; per_axis=50)                                             #src
recovery = sum(sign(predmean(g, p)) == sign(f(p)) for p in grid) / length(grid) #src
@test recovery > 0.95                                                            #src
@test sum(abs(f(p)) < 0.2 for p in queried_points(al)) /                        #src
      length(queried_points(al)) > 0.5                                          #src
