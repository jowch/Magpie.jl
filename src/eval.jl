# Evaluation harness for GP-UDE recovery quality.
#
# Pure module — no OrdinaryDiffEq / SciMLSensitivity / Mooncake.
# Operates on already-computed posteriors and propagated outputs.
# StatsFuns.chisqinvcdf is imported at the Magpie module level.

# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

"""
    coverage(truth, μs, Σs; level=0.9) -> Float64

Fraction of steps where the truth is inside the `level` credible region.

**Convention:** Mahalanobis (χ²) ellipsoid. For step `n`, the credible ellipsoid
is `{x : (x−μ)ᵀ Σ⁻¹ (x−μ) ≤ χ²_{d,level}}`, where `d = length(μs[n])` and
`χ²_{d,level}` is the `level` quantile of the chi-squared distribution with `d`
degrees of freedom.

Rationale for χ²/Mahalanobis over per-dimension marginal boxes:
  - Gives exactly `level` nominal coverage for a multivariate Gaussian regardless
    of dimension `d`, whereas a per-dim box gives only `level^d` for independent
    dims (severely low: `0.9^3 ≈ 0.73` for d=3).
  - The set is the natural credible region for a Gaussian: isocontours of the
    predictive density.
  - Requires one Cholesky per step, not d separate quantile evaluations.

`truth`, `μs` — `AbstractVector` of length-`d` vectors (one per time-step).
`Σs`          — `AbstractVector` of `d×d` matrices (full posterior covariances).
"""
function coverage(
        truth::AbstractVector,
        μs::AbstractVector,
        Σs::AbstractVector;
        level::Real = 0.9,
    )
    n = length(truth)
    @assert length(μs) == n && length(Σs) == n "truth, μs, Σs must have the same length"
    d = length(first(truth))
    χ²_thr = chisqinvcdf(d, level)     # χ²_{d, level} threshold

    hits = 0
    counted = 0
    for i in 1:n
        Σ = Σs[i]
        isposdef(Σ) || continue         # skip degenerate steps (e.g. PULL's t=0 Σ=0 point mass)
        counted += 1
        r = truth[i] .- μs[i]          # residual vector
        C = _chol(Σ)                    # Cholesky of Σ (uses existing chokepoint)
        # maha² = rᵀ Σ⁻¹ r = ‖L \ r‖²  (where Σ = LLᵀ)
        maha2 = sum(abs2, C.L \ r)
        maha2 ≤ χ²_thr && (hits += 1)
    end
    counted == 0 && return NaN          # all steps degenerate — undefined coverage
    return hits / counted
end

# ---------------------------------------------------------------------------
# field_error
# ---------------------------------------------------------------------------

"""
    field_error(gps, truefield, pts) -> NamedTuple{(:median, :q90)}

Median and 90th-percentile of `‖[predmean(g,z) for g in gps] − truefield(z)‖₂`
over `pts` (a vector of states).

`gps`       — `Vector{<:AbstractGPModel}`, one GP per output dimension.
              `predmean(gps[j], z)` gives the posterior mean for dimension `j` at `z`.
`truefield` — callable `z -> AbstractVector` returning the true RHS at state `z`.
`pts`       — `AbstractVector` of states (each state is whatever `predmean` accepts).

Returns `(median=…, q90=…)` as a `NamedTuple`.
"""
function field_error(gps, truefield, pts::AbstractVector)
    errs = map(pts) do z
        pred = [predmean(g, z) for g in gps]
        true_ = truefield(z)
        norm(pred .- true_)
    end
    return (median = median(errs), q90 = quantile(errs, 0.9))
end

# ---------------------------------------------------------------------------
# recovery_metrics
# ---------------------------------------------------------------------------

"""
    recovery_metrics(gps, truefield, traj_pred, traj_truth; offpts=nothing) -> NamedTuple

Combine trajectory RMSE and field errors into one NamedTuple.  Pure — no solver.

# Arguments
- `gps`        — `Vector{<:AbstractGPModel}` posterior GP models (one per dim).
- `truefield`  — callable `z -> AbstractVector` of true RHS.
- `traj_pred`  — pre-integrated predicted trajectory (vector of state vectors).
- `traj_truth` — ground-truth trajectory aligned with `traj_pred`.
- `offpts`     — optional off-manifold test points; if `nothing`, only
                 on-trajectory field error is computed and `field_err_offmanifold`
                 is `(median=NaN, q90=NaN)`.

# Returns
`NamedTuple` with fields:
- `traj_rmse`             — root-mean-square error between `traj_pred` and `traj_truth`.
- `field_err_visited`     — `(median, q90)` field error at `traj_truth` states.
- `field_err_offmanifold` — `(median, q90)` field error at `offpts` (or `(NaN, NaN)`).

Caller must pre-integrate the trajectory (no solver is built here).
"""
function recovery_metrics(gps, truefield, traj_pred, traj_truth; offpts = nothing)
    # Trajectory RMSE — pure, caller supplies both trajectories
    traj_rmse = sqrt(mean(sum(abs2, p .- t) for (p, t) in zip(traj_pred, traj_truth)))

    # Field error at visited (on-trajectory) points
    fe_vis = field_error(gps, truefield, traj_truth)

    # Field error at off-manifold points (optional)
    fe_off = if offpts !== nothing
        field_error(gps, truefield, offpts)
    else
        (median = NaN, q90 = NaN)
    end

    return (
        traj_rmse = traj_rmse,
        field_err_visited = fe_vis,
        field_err_offmanifold = fe_off,
    )
end

# ---------------------------------------------------------------------------
# ridge_slice
# ---------------------------------------------------------------------------

"""
    ridge_slice(loss, v; idx=(1,2), grid) -> Matrix

Evaluate `loss` over a 2-D grid of two parameter indices around `v`.

Pure replay of the supplied `loss` closure — `ridge_slice` does NOT build one.
Useful for visualising identifiability (e.g. the (logℓ, logσ) plane) or verifying
that a minimum is well-localised.

# Arguments
- `loss` — callable `w::AbstractVector -> Real`.
- `v`    — centre parameter vector (used for all dimensions not in `idx`).
- `idx`  — 2-tuple `(i, j)` of parameter indices to sweep (default `(1,2)`).
- `grid` — `(xs, ys)` pair of 1-D grids for dimensions `idx[1]` and `idx[2]`
           respectively (e.g. `(range(-2,2;length=40), range(-3,3;length=40))`).

# Returns
`Matrix` of size `(length(xs), length(ys))` where entry `[a, b]` is
`loss(v_with_idx[1]=xs[a], v_with_idx[2]=ys[b])`.
"""
function ridge_slice(loss, v::AbstractVector; idx::Tuple{Int, Int} = (1, 2), grid)
    xs, ys = grid
    nx, ny = length(xs), length(ys)
    out = Matrix{Float64}(undef, nx, ny)
    w = copy(v)        # copy so v is never mutated
    i1, i2 = idx
    for (a, x) in enumerate(xs)
        w[i1] = x
        for (b, y) in enumerate(ys)
            w[i2] = y
            out[a, b] = loss(w)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# pathwise_moments
# ---------------------------------------------------------------------------

"""
    pathwise_moments(ens) -> (μs, Σs)

Convert a Pathwise ensemble `ens` of shape `N × d × T` (samples × dimension ×
time-step) into per-step posterior moments consumable by [`coverage`](@ref):
`μs[k]` is the length-`d` sample mean and `Σs[k]` the `d×d` sample covariance at
time-step `k`. Samples are the rows of each `ens[:, :, k]` slice.
"""
function pathwise_moments(ens::AbstractArray{<:Real, 3})
    N, d, T = size(ens)
    N ≥ 2 || throw(ArgumentError("pathwise_moments needs ≥2 samples, got N=$N"))
    μs = [vec(mean(@view(ens[:, :, k]); dims = 1)) for k in 1:T]
    Σs = [Matrix(cov(@view(ens[:, :, k]))) for k in 1:T]   # cov over rows → d×d
    return μs, Σs
end
