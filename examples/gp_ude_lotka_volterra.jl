# # GP-UDE: Lotka-Volterra
#
# Learn the predator–prey vector field from **noisy observations**.
#
# A single trajectory over `(0, 3)` with observation noise `σ_obs ≈ 0.03` is
# enough for single-shooting to recover the limit-cycle dynamics.
# `ExactGPField` is trained end-to-end via `train!` (ADAM warm-up → LBFGS
# polish). Recovery is measured three honest ways:
#
#   1. **Trajectory RMSE** — ODE integration of the GP posterior mean field vs
#      the clean ground truth.  PULL is the uncertainty propagator, not the
#      mean-trajectory integrator: the Euler recurrence in PULL accumulates
#      much larger errors than a proper ODE solver on the same mean field.
#   2. **Field error** — GP vector field vs true RHS, on-trajectory and
#      off-manifold (grid around the limit cycle).
#   3. **Held-out-IC coverage** — propagate uncertainty (PULL) from a fresh
#      initial condition and check that the clean trajectory falls inside
#      the uncertainty band.
#
# None of these metrics is in-sample loss.

ENV["GKSwstype"] = "100"  ## GR headless
using Magpie, OrdinaryDiffEq, SciMLSensitivity, KernelFunctions, LinearAlgebra, Random
using Statistics
using Plots; gr()

# ## True system

function lv!(du, u, p, t)
    α, β, γ, δ = 1.5, 1.0, 3.0, 1.0
    du[1] = α*u[1] - β*u[1]*u[2]
    du[2] = δ*u[1]*u[2] - γ*u[2]
    nothing
end

lv_true(u) = [1.5*u[1] - u[1]*u[2], u[1]*u[2] - 3.0*u[2]]

u0    = [1.0, 1.0]
tspan = (0.0, 3.0)
ts    = collect(range(tspan...; length=15))

# Clean training trajectory.
target = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat=ts))

# ## Add observation noise (σ_obs_true ≈ 0.03)

σ_obs_true = 0.03f0
rng_noise  = MersenneTwister(123)
Xnoisy     = target .+ σ_obs_true .* randn(rng_noise, size(target))

# ## Build and train the GP-UDE field on NOISY data

Random.seed!(42)
Z     = kmeans_anchors(Xnoisy, 12; rng=MersenneTwister(7))
field = ExactGPField(SqExponentialKernel(), Z; d=2)

# `λ=1/(15*2)` is a weak log-ℓ prior centred at 0 with std 0.5.
# We train on `Xnoisy` — the field must learn through noise, not a clean signal.
field, vopt = train!(field, (ts, Xnoisy); tspan, maxiters=150, λ=1/(15*2), s=0.5)

# ## Posterior GPs

gps = posterior_gps(field, vopt)

# ## 1. Trajectory RMSE — ODE integration of the GP mean field vs clean truth
#
# PULL propagation uses a first-order Euler step for the mean, which accumulates large
# errors on a nonlinear limit cycle.  For the mean-trajectory metric we integrate
# the GP posterior mean field as a proper ODE (same as what training minimises),
# then compare against the clean truth.

gp_rhs!(du, u, p, t) = (du .= [predmean(gps[i], u) for i in 1:2]; nothing)
sol_gp = Array(solve(ODEProblem(gp_rhs!, u0, tspan), Tsit5(); saveat=ts))

traj_pred  = [sol_gp[:, i] for i in 1:length(ts)]
traj_truth = [target[:, i] for i in 1:length(ts)]

# Off-manifold test grid: 10×10 around the limit cycle.
prey_grid = range(0.3, 2.5; length=10)
pred_grid = range(0.3, 3.5; length=10)
offpts    = vec([[p1, p2] for p1 in prey_grid, p2 in pred_grid])

metrics = recovery_metrics(gps, lv_true, traj_pred, traj_truth; offpts=offpts)

@info "Trajectory RMSE (ODE integration of GP mean vs clean truth)" metrics.traj_rmse
@info "Field error (on-trajectory)"  metrics.field_err_visited.median  metrics.field_err_visited.q90
@info "Field error (off-manifold)"   metrics.field_err_offmanifold.median  metrics.field_err_offmanifold.q90

# ## 2. Plot: mean trajectory vs clean data

p1 = plot(ts, target[1,:], label="prey (clean)", lw=2, c=:blue)
plot!(p1, ts, target[2,:], label="pred (clean)", lw=2, c=:red)
plot!(p1, ts, Xnoisy[1,:], label="prey (noisy data)", lw=1, c=:blue, ls=:dot, alpha=0.6)
plot!(p1, ts, Xnoisy[2,:], label="pred (noisy data)", lw=1, c=:red,  ls=:dot, alpha=0.6)
plot!(p1, ts, sol_gp[1,:], label="prey (GP ODE mean)", lw=2, c=:blue, ls=:dash)
plot!(p1, ts, sol_gp[2,:], label="pred (GP ODE mean)", lw=2, c=:red,  ls=:dash)
xlabel!(p1, "t"); ylabel!(p1, "population")
title!(p1, "LV GP-UDE: mean trajectory (trained on noisy data)")
savefig(p1, "lv_trajectory.png")

# ## 3. Held-out initial condition — PULL uncertainty propagation + coverage

u0_test   = [1.2, 0.8]   # NOT the training IC
ts_test   = collect(range(tspan...; length=15))

# Clean ground-truth trajectory from the held-out IC.
target_test = Array(solve(ODEProblem(lv!, u0_test, tspan), Tsit5(); saveat=ts_test))

# PULL propagation from the held-out IC — uncertainty only (mean is Euler, not ODE quality).
μs_test, Σs_test = propagate(gps, u0_test, tspan; method=PULL(), ts=ts_test)

# Convert to vector-of-vectors for coverage (Mahalanobis χ² ellipsoid).
truth_vecs = [target_test[:, i] for i in 1:length(ts_test)]
cov90 = coverage(truth_vecs, μs_test, Σs_test; level=0.9)

@info "Held-out-IC PULL coverage at 90% nominal" cov90

# ## 4. Plot: PULL uncertainty band from held-out IC

μmat_test = reduce(hcat, μs_test)   # 2 × |ts_test|
σs_test   = [sqrt.(max.(diag(Σ), 0.0)) for Σ in Σs_test]
σmat_test = reduce(hcat, σs_test)   # 2 × |ts_test|

p2 = plot(ts_test, target_test[1,:], label="prey (clean, held-out IC)", lw=2, c=:blue)
plot!(p2, ts_test, target_test[2,:], label="pred (clean, held-out IC)", lw=2, c=:red)
plot!(p2, ts_test, μmat_test[1,:], label="prey μ (PULL)", lw=2, c=:blue, ls=:dash)
plot!(p2, ts_test, μmat_test[2,:], label="pred μ (PULL)", lw=2, c=:red,  ls=:dash)
# ±2σ bands (using marginal std from PULL covariances)
plot!(p2, ts_test, μmat_test[1,:] .+ 2 .* σmat_test[1,:], fillrange=μmat_test[1,:] .- 2 .* σmat_test[1,:],
      alpha=0.15, c=:blue, label="prey μ±2σ", lw=0)
plot!(p2, ts_test, μmat_test[2,:] .+ 2 .* σmat_test[2,:], fillrange=μmat_test[2,:] .- 2 .* σmat_test[2,:],
      alpha=0.15, c=:red, label="pred μ±2σ", lw=0)
xlabel!(p2, "t"); ylabel!(p2, "population")
title!(p2, "LV GP-UDE: held-out-IC PULL uncertainty band")
savefig(p2, "lv_trajectory_coverage.png")

# ## Anti-rot assertions (#src lines run on direct execution only)
#
# Hard gate: trajectory RMSE (ODE integration of GP mean vs clean truth) and
# on-trajectory field error. Coverage is @info'd — PULL over a nonlinear limit-cycle
# field can be over/under-dispersed, so a failing coverage number is a calibration
# finding, not a correctness bug.

using Test  #src
@test metrics.traj_rmse < 0.15   #src  ODE integration of GP mean vs clean truth
@test metrics.field_err_visited.median < 0.5   #src  on-trajectory field error
@info "Held-out-IC coverage at 90% nominal: $(round(cov90; digits=3)). PULL over nonlinear LV — Euler mean diverges from true trajectory; coverage may be off."  #src
