# # GP-UDE: Van der Pol (stiff solver)
#
# A relaxation oscillator (μ=1.5) — mildly stiff. We generate training data with
# `AutoTsit5(Rosenbrock23())` and demonstrate that the same solver can be passed to
# `train!` via the `solver=` kwarg (forwarded to every in-loss ODE solve).
#
# For μ=1.5 both Tsit5 and Rosenbrock produce essentially identical trajectories, so we
# train with `Tsit5` (which pairs cleanly with `GaussAdjoint+MooncakeVJP`) and verify
# the stiff-solver assertion separately to show the kwarg round-trips.  For strongly
# stiff systems (μ≫1) you would pass `solver=AutoTsit5(Rosenbrock23())` to `train!`
# together with `sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))`.
#
# Short horizon `(0, 3)` for single-shooting recovery.

ENV["GKSwstype"] = "100"  ## GR headless
using Magpie, OrdinaryDiffEq, SciMLSensitivity, KernelFunctions, LinearAlgebra, Random
using Plots; gr()
Random.seed!(7)

# ## True system — Van der Pol oscillator (μ=1.5)

vdp!(du, u, p, t) = (du[1] = u[2]; du[2] = 1.5*(1 - u[1]^2)*u[2] - u[1]; nothing)

u0 = [2.0, 0.0]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length=20))
# Stiff solver for data generation — a best-practice for relaxation oscillators.
target = Array(solve(ODEProblem(vdp!, u0, tspan), AutoTsit5(Rosenbrock23()); saveat=ts))

# ## Build the GP-UDE field

Z = kmeans_anchors(target, 12; rng=MersenneTwister(7))
field = ExactGPField(SqExponentialKernel(), Z; d=2)

# Default `solver=Tsit5()` pairs cleanly with `GaussAdjoint+MooncakeVJP`.
# For strongly stiff systems pass `solver=AutoTsit5(Rosenbrock23())` together with
# `sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))` to `train!`.
field, vopt = train!(field, (ts, target); tspan, maxiters=150, λ=1/(20*2))

# ## Posterior and trajectory

gps = posterior_gps(field, vopt)
μs, Σs = propagate(gps, u0, tspan; method=PULL(), ts=ts)
μmat = reduce(hcat, μs)

p1 = plot(ts, target[1,:], label="x (true)", lw=2, c=:blue)
plot!(p1, ts, target[2,:], label="ẋ (true)", lw=2, c=:red)
plot!(p1, ts, μmat[1,:], label="x (GP)", ls=:dash, lw=2, c=:blue)
plot!(p1, ts, μmat[2,:], label="ẋ (GP)", ls=:dash, lw=2, c=:red)
xlabel!(p1, "t"); ylabel!(p1, "state"); title!(p1, "Van der Pol: GP-UDE trajectory")
savefig(p1, "vdp_trajectory.png")

# ## Anti-rot assertion (#src lines run on direct execution only)
# Asserts trajectory RMSE via make_loss with the stiff solver — verifying the `solver=` kwarg
# round-trips into `make_loss` correctly (μ=1.5 is mildly stiff; both solvers agree closely).

using Test  #src
ext = Base.get_extension(Magpie, :MagpieSciMLExt)                                              #src
dloss = ext.make_loss(field, Magpie.FieldLayout(field.n, field.d), u0, tspan, ts, target;       #src
                      solver=AutoTsit5(Rosenbrock23()), λ=0.0, λσ=0.0)                          #src
sol_rmse = sqrt(dloss(vopt) / (20*2))                                                           #src
@info "VdP sol_rmse = $sol_rmse"                                                               #src
@test sol_rmse < 0.15                                                                           #src
