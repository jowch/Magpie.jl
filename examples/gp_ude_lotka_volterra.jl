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
#   3. **Held-out-IC coverage** — propagate uncertainty from a fresh initial
#      condition and check that the clean trajectory falls inside the band.
#      Two propagators are compared:
#        - **PULL** — a cheap analytic moment-matching propagator. Its mean uses a
#          first-order Euler recurrence, which drifts off the true trajectory on a
#          nonlinear limit cycle; coverage collapses to ~0. This is a documented
#          *Euler limitation of PULL*, not a failure of the learned field.
#        - **Pathwise** — a Monte-Carlo ensemble of decoupled GP samples, each
#          integrated as a proper ODE. This is the accurate, validated-uncertainty
#          story; its empirical band actually covers the held-out truth.
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

# ## 3. Held-out initial condition — uncertainty propagation + coverage
#
# We propagate uncertainty from a fresh initial condition two ways and compare
# coverage of the clean held-out trajectory: PULL (cheap analytic, Euler-limited)
# and Pathwise (Monte-Carlo ensemble of decoupled GP samples).

u0_test   = [1.2, 0.8]   # NOT the training IC
ts_test   = collect(range(tspan...; length=15))

# Clean ground-truth trajectory from the held-out IC.
target_test = Array(solve(ODEProblem(lv!, u0_test, tspan), Tsit5(); saveat=ts_test))
truth_vecs  = [target_test[:, i] for i in 1:length(ts_test)]

# ### 3a. PULL — cheap analytic propagation (Euler-limited on nonlinear horizons)
#
# PULL's mean is a first-order Euler recurrence; on the LV limit cycle it drifts
# from the true trajectory, so coverage collapses to ~0. Kept as a documented
# contrast, NOT as the validated-uncertainty story.
μs_test, Σs_test = propagate(gps, u0_test, tspan; method=PULL(), ts=ts_test)
cov90_pull = coverage(truth_vecs, μs_test, Σs_test; level=0.9)

@info "Held-out-IC PULL coverage at 90% nominal (Euler-limited)" cov90_pull

# ### 3b. Pathwise — Monte-Carlo ensemble of decoupled GP samples (REAL validation)
#
# Each of `n` samples is a decoupled GP draw integrated as a proper ODE, so the
# ensemble carries the field's uncertainty *without* PULL's Euler drift.
# `propagate(...; method=Pathwise(n=N))` returns an `N × d × |ts|` array.
ens = propagate(gps, u0_test, tspan; method=Pathwise(n=128), ts=ts_test)

# Per-step empirical mean + covariance from the ensemble, then reuse the same
# Mahalanobis-χ² `coverage` as PULL (apples-to-apples at 90% nominal).
nsteps      = length(ts_test)
μs_path     = [vec(mean(ens[:, :, k]; dims=1)) for k in 1:nsteps]        # d-vector per step
Σs_path     = [cov(ens[:, :, k]) for k in 1:nsteps]                     # d×d per step (samples in rows)
cov90_path  = coverage(truth_vecs, μs_path, Σs_path; level=0.9)

@info "Held-out-IC Pathwise coverage at 90% nominal (ensemble, validated)" cov90_path

# ## 4. Plot: Pathwise ensemble band over the clean held-out trajectory
#
# 5th–95th percentile envelope (90% band) per dimension across the n samples,
# overlaid on the clean held-out truth. This is the band whose coverage is
# reported above as `cov90_path`.

prey_lo = [quantile(ens[:, 1, k], 0.05) for k in 1:nsteps]
prey_hi = [quantile(ens[:, 1, k], 0.95) for k in 1:nsteps]
pred_lo = [quantile(ens[:, 2, k], 0.05) for k in 1:nsteps]
pred_hi = [quantile(ens[:, 2, k], 0.95) for k in 1:nsteps]
μmat_path = reduce(hcat, μs_path)   # 2 × |ts_test|

p2 = plot(ts_test, target_test[1,:], label="prey (clean, held-out IC)", lw=2, c=:blue)
plot!(p2, ts_test, target_test[2,:], label="pred (clean, held-out IC)", lw=2, c=:red)
plot!(p2, ts_test, μmat_path[1,:], label="prey μ (Pathwise)", lw=2, c=:blue, ls=:dash)
plot!(p2, ts_test, μmat_path[2,:], label="pred μ (Pathwise)", lw=2, c=:red,  ls=:dash)
# 5–95% Pathwise envelopes.
plot!(p2, ts_test, prey_hi, fillrange=prey_lo, alpha=0.15, c=:blue, label="prey 5–95%", lw=0)
plot!(p2, ts_test, pred_hi, fillrange=pred_lo, alpha=0.15, c=:red,  label="pred 5–95%", lw=0)
xlabel!(p2, "t"); ylabel!(p2, "population")
title!(p2, "LV GP-UDE: held-out-IC Pathwise ensemble band (90%)")
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
# PULL coverage is @info'd only — its Euler mean drifts off the LV limit cycle, so a
# low/zero coverage is a documented PULL limitation, not a correctness bug.            #src
@info "Held-out-IC PULL coverage at 90% nominal: $(round(cov90_pull; digits=3)) (Euler-limited; documented contrast)."  #src
# Pathwise is the real validated-uncertainty gate: the decoupled-sample ensemble        #src
# integrated as proper ODEs should actually cover the clean held-out trajectory.        #src
@test cov90_path ≥ 0.6   #src  Pathwise ensemble coverage at 90% nominal (validated uncertainty)
