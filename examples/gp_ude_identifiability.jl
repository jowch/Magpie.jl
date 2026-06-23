# # GP-UDE: identifiability (logℓ–logσ ridge + off-data divergence)
#
# **Non-identifiability is a feature, not a bug — but it must be surfaced.**
#
# A GP-UDE field has a classic flat direction in the (logℓ, logσ) plane: many
# combinations of lengthscale and output scale, together with the weight vector `w`,
# produce nearly identical training-trajectory fits. This example demonstrates:
#
#   1. **Ridge slice**: sweep (logℓ, logσ) holding `w` fixed at the optimum — reveals the
#      exact scale-invariance and the narrow well in logℓ. The combined landscape is flat
#      (ratio ≫ 1 across vs along the valley).
#   2. **Multi-seed divergence**: training from different initial (logℓ₀, logσ₀) seeds
#      converges to solutions with similar trajectory RMSE but diverging field predictions
#      off the training support — the identifiability problem in concrete numbers.
#
# Decision D2 ("surface, don't hide"): show the problem honestly.

ENV["GKSwstype"] = "100"  ## GR headless
using Magpie, OrdinaryDiffEq, SciMLSensitivity, KernelFunctions, LinearAlgebra, Random
using Statistics
using Plots; gr()

# ## True system (Lotka-Volterra)

function lv!(du, u, p, t)
    α, β, γ, δ = 1.5, 1.0, 3.0, 1.0
    du[1] = α * u[1] - β * u[1] * u[2]
    du[2] = δ * u[1] * u[2] - γ * u[2]
    return nothing
end

lv_true(u) = [1.5 * u[1] - u[1] * u[2], u[1] * u[2] - 3.0 * u[2]]

u0 = [1.0, 1.0]
tspan = (0.0, 3.0)
ts = collect(range(tspan...; length = 15))

target = Array(solve(ODEProblem(lv!, u0, tspan), Tsit5(); saveat = ts))

σ_obs_true = 0.03f0
rng_noise = MersenneTwister(123)
Xnoisy = target .+ σ_obs_true .* randn(rng_noise, size(target))

# Shared anchor grid (same Z for all seeds so w-blocks correspond to same locations).
Random.seed!(42)
Z = kmeans_anchors(Xnoisy, 12; rng = MersenneTwister(7))

# Extension (get once — field_loss lives in MagpieSciMLExt).
ext = Base.get_extension(Magpie, :MagpieSciMLExt)

# ## 1. Ridge slice: sweep (logℓ, logσ) with w fixed at the optimum
#
# Train once (weak λ=1e-4 logℓ prior so it lands freely on the ridge) to get a
# reference optimum, then sweep (logℓ, logσ) while holding `w` fixed.
# This reveals the exact scale-symmetry: predmean = kuZ'α is INVARIANT to logσ when
# w is fixed (logσ enters both kuZ and K_ZZ symmetrically; the scale cancels in α).
# Result: a flat valley in logσ at every logℓ, and a sharp well in logℓ.

Random.seed!(1)
field_ref = ExactGPField(SqExponentialKernel(), Z; d = 2, logℓ0 = 0.0)
train!(field_ref, (ts, Xnoisy); tspan, maxiters = 150, λ = 1.0e-4, s = 0.5)
vopt = field_ref.v0

@info "Reference optimum" logℓ = round(vopt[1]; digits = 3) logσ = round(vopt[2]; digits = 3)

# Pure data-fit loss (no regularizer) for the ridge slice.
loss_raw = ext.field_loss(
    field_ref, Magpie.SingleShooting(), [(ts, Xnoisy)];
    tspan, u0 = u0, λ = 0.0, λσ = 0.0
)
loss_at_opt = loss_raw(vopt)
@info "Data-fit loss at reference optimum" loss_at_opt

logℓ_ref, logσ_ref = vopt[1], vopt[2]
xs = range(logℓ_ref - 1.2, logℓ_ref + 1.2; length = 11)  # logℓ axis
ys = range(logσ_ref - 1.5, logσ_ref + 1.5; length = 11)  # logσ axis

@info "Computing 11×11 ridge slice (121 forward solves, w fixed at reference optimum)..."
M_slice = ridge_slice(loss_raw, vopt; idx = (1, 2), grid = (xs, ys))

# Ridge flatness: full-grid range vs valley-internal range.
# "Valley" = cells with data-fit loss < 3× the minimum value in the slice.
loss_vals_finite = filter(v -> isfinite(v) && v < 1.0e5, vec(M_slice))
loss_min = minimum(loss_vals_finite)
loss_range_full = maximum(loss_vals_finite) - loss_min
valley_vals = filter(v -> v < loss_min * 3.0, loss_vals_finite)
loss_range_valley = isempty(valley_vals) ? 0.0 : maximum(valley_vals) - minimum(valley_vals)
flatness_ratio = loss_range_full / max(loss_range_valley, 1.0e-10)

@info "Ridge flatness (w fixed at optimum)" valley_range = round(loss_range_valley; digits = 3) full_range = round(loss_range_full; digits = 3) ratio = round(flatness_ratio; digits = 0)

# Heatmap (clamp diverged solves to a finite ceiling).
M_plot = clamp.(M_slice, 0.0, loss_at_opt * 8)
p_ridge = heatmap(
    collect(xs), collect(ys), M_plot';
    xlabel = "logℓ", ylabel = "logσ",
    title = "Data-fit landscape: (logℓ, logσ) with w fixed at optimum",
    color = :viridis, clims = (minimum(M_plot), loss_at_opt * 5)
)
scatter!(p_ridge, [logℓ_ref], [logσ_ref]; marker = :star5, ms = 10, mc = :red, label = "ref optimum")
savefig(p_ridge, "identifiability_ridge.png")

# ## 2. Multi-seed training: converge to different ridge solutions
#
# Three seeds with different logℓ₀ initializations converge to different (logℓ, logσ)
# combinations.  With a very weak logℓ prior (λ=1e-4), the optimizer can land anywhere
# on the flat logσ direction and in a modest well in logℓ.
# We report trajectory RMSE (vs clean truth) as the "data fit" metric —
# this is fairer than NLL because the observation-noise σ_obs is also trained and
# can drive NLL very negative while RMSE stays bounded.

SEED_INITS = [(1, 0.0), (10, 1.5), (17, 0.3)]   # (seed, logℓ₀)

_train_seed(seed, logℓ0) = begin
    Random.seed!(seed)
    f = ExactGPField(SqExponentialKernel(), Z; d = 2, logℓ0 = logℓ0)
    train!(f, (ts, Xnoisy); tspan, maxiters = 150, λ = 1.0e-4, s = 0.5)
    # Trajectory RMSE: integrate GP mean field vs clean truth (same metric as the LV example).
    gps = posterior_gps(f)
    gp_rhs!(du, u, p, t) = (du .= [predmean(gps[i], u) for i in 1:2]; nothing)
    sol_gp = Array(solve(ODEProblem(gp_rhs!, u0, tspan), Tsit5(); saveat = ts))
    traj_rmse = sqrt(sum(abs2, sol_gp .- target) / length(target))
    (v = f.v0, gps = gps, traj_rmse = traj_rmse)
end

@info "Training 3 seeds (weak regulariser)..."
seed_results = [_train_seed(seed, logℓ0) for (seed, logℓ0) in SEED_INITS]
vopts_ms = [r.v     for r in seed_results]
gps_ms = [r.gps   for r in seed_results]
traj_rmses = [r.traj_rmse for r in seed_results]

@info "Multi-seed trajectory RMSE (all should be similar — fitted the data):"
for ((seed, logℓ0), v, rmse) in zip(SEED_INITS, vopts_ms, traj_rmses)
    @info "  Seed $seed (logℓ₀=$logℓ0)" logℓ = round(v[1]; digits = 3) logσ = round(v[2]; digits = 3) traj_rmse = round(rmse; digits = 4)
end

# ## 3. Off-data divergence
#
# Measure field error at off-manifold test points for each seed solution.
# These points are OUTSIDE the training trajectory support.
# Even though all seeds fit the training trajectory (similar RMSE), their
# field predictions diverge off the training support.

prey_grid = range(0.3, 2.5; length = 10)
pred_grid = range(0.3, 3.5; length = 10)
offpts = vec([[p1, p2] for p1 in prey_grid, p2 in pred_grid])

ferrs_ms = map(enumerate(zip(SEED_INITS, gps_ms))) do (i, ((seed, logℓ0), gps))
    fe = field_error(gps, lv_true, offpts)
    @info "  Off-data field error" seed logℓ = round(vopts_ms[i][1]; digits = 3) median = round(fe.median; digits = 4) q90 = round(fe.q90; digits = 4)
    fe
end

all_medians_ms = [fe.median for fe in ferrs_ms]
ferr_spread_ms = maximum(all_medians_ms) - minimum(all_medians_ms)
traj_rmse_spread = maximum(traj_rmses) - minimum(traj_rmses)

@info "Summary" traj_rmse_spread = round(traj_rmse_spread; digits = 4) ferr_spread = round(ferr_spread_ms; digits = 4) medians = round.(all_medians_ms; digits = 4)

# ## 4. Visualise the divergence
#
# Two side-by-side bars: trajectory RMSE (should be similar) and off-data field
# error (should diverge) for each seed solution.

seed_labels = [
    "seed $(s)\nlogℓ=$(round(v[1]; digits = 2))"
        for ((s, li), v) in zip(SEED_INITS, vopts_ms)
]

p_rmse = bar(
    seed_labels, traj_rmses;
    ylabel = "Trajectory RMSE",
    title = "On-data fit (similar — all fit the trajectory)",
    legend = false, color = :steelblue
)

p_ferr = bar(
    seed_labels, all_medians_ms;
    ylabel = "Field error (off-data median)",
    title = "Off-data divergence (similar on-data, diverging off-data)",
    legend = false, color = :orange
)

p_both = plot(p_rmse, p_ferr; layout = (1, 2), size = (800, 350))
savefig(p_both, "identifiability_divergence.png")

# ## Anti-rot assertions (#src lines run on direct execution only)
#
# 1. Ridge flatness: the raw data-fit loss (w fixed) spans much more variation
#    across the full (logℓ, logσ) grid than within the low-loss valley.
#    A ratio > 5 is conservative; actual is typically > 1e6 (exact scale symmetry).
# 2. Trajectory RMSE spread across seeds: all solutions fit the training trajectory
#    reasonably well. A spread < 0.15 confirms all seeds found valid fits.
# 3. Off-data field error spread: different ridge solutions diverge off the training
#    manifold. A spread > 0.1 confirms the identifiability problem.

using Test  #src
# Ridge flatness: full-grid range >> valley-internal range (exact scale symmetry).  #src
@test flatness_ratio > 5.0   #src
# Multi-seed trajectory RMSE spread: all seeds found comparable training fits.      #src
@test traj_rmse_spread < 0.15   #src
# Off-data field errors diverge across ridge solutions.                            #src
@test ferr_spread_ms > 0.1   #src
