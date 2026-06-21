# Capability B core — pure (no SciML import). See ext/MagpieSciMLExt.jl for solve-touching code.

"""
    GPField

Protocol root for the Capability B GP-UDE fields. Every concrete field implements:

  - `unpack(field, v) -> NamedTuple` — flat trained vector → named params (pure layout).
  - `regularizer(field, v; kw...) -> Real` — priors (+ KL for SVGP); pure, no solver.
  - `posterior(field, v) -> Vector{<:AbstractGPModel}` — solver-free reconstruction (ext).
  - `field_rhs(field, v) -> Function` — the in-loss α/Cholesky + `du += f(u)` closure (ext).

The shooting strategy (`SingleShooting`/`MultipleShooting`) is orthogonal to field type;
the loss is `field_loss(field, shooting, data) = shooting_data_term(...) + regularizer(...)`.
"""
abstract type GPField end

"""
    CompositeField(known, gp)

A UDE field that decomposes as `f(u,t) = known(u,t) + gp(u)`, where `known` is a fixed
(u,t)->du function supplying the known-physics part and `gp` is a trainable `GPField`
learning the residual.

Protocol delegates entirely to `gp`: `unpack`, `regularizer`, and `posterior` forward to `cf.gp`.
The `field_rhs` in the extension bakes `known` into the closure so the returned `rhs!` does:
    du .= cf.known(u, t); du .+= gpfield(cf.gp, u, pf)
This is the proven `du .= known; du .+= gp` ordering, and α stays threaded in `pf` (R1).
"""
struct CompositeField{Kf,Gf<:GPField} <: GPField
    known::Kf   # (u,t) -> du_known  (fixed, not trained)
    gp::Gf      # the residual GP field that is trained
end

# Protocol delegation — CompositeField trains only the inner gp.
unpack(cf::CompositeField, v)           = unpack(cf.gp, v)
regularizer(cf::CompositeField, v; kw...) = regularizer(cf.gp, v; kw...)

# v0 accessor: CompositeField is transparent — its trained-vector layout IS the inner gp's.
# (train! and _init_vec read field.v0 directly, so forward the property.)
# NOTE on write-back: `field.v0 .= sol.u` (in-place broadcast) works because `.=` calls
# getproperty to fetch the inner Vector, then mutates it in-place. A plain assignment
# `field.v0 = sol.u` (no dot) would throw a MethodError (no setproperty! defined; YAGNI).
Base.getproperty(cf::CompositeField, s::Symbol) =
    s === :known ? getfield(cf, :known) :
    s === :gp    ? getfield(cf, :gp)   :
    getproperty(getfield(cf, :gp), s)   # forward n, d, v0, lognoise, etc. to inner gp

"""Trainable GP-UDE field: one shared kernel + fixed anchors `Z`, `d` independent outputs."""
struct ExactGPField{Tp,TZ} <: GPField
    prior::Tp        # AbstractGPs.GP (mean + kernel); kernel hypers are overridden per-eval from pf
    Z::TZ            # Vector{Vector{Float64}} of anchors
    n::Int           # number of anchors
    d::Int           # output dimension
    lognoise::Float64     # FIXED jitter on anchor values (the field is never observed; identifiability mech. 2)
    v0::Vector{Float64}   # initial flat TRAINED params [logℓ, logσ, logσ_obs, vec(w)] — lognoise is NOT trained here
end

"""Sparse variational GP as a full spine `AbstractGPModel` (per interface-design). Stores enough to
serve the whole AbstractGP contract via the whitened moments: `α = L_ZZ'\\μ` (predmean), plus the
inducing Cholesky `L_ZZ` and the variational factor `L_S` (var/cov — needed by SVGP PULL uncertainty).
mean/var/cov methods are defined in Task 7 via `svgp_moments`."""
struct SparseGP{Tp,TZ,Tα,TL,TS} <: AbstractGPModel
    prior::Tp; Z::TZ; α::Tα; L_ZZ::TL; L_S::TS
end

# SINGLE SOURCE OF TRUTH for the hyper-prefix width. The trained vector ALWAYS begins
# `[logℓ, logσ, logσ_obs, <field-specific>...]`. logσ_obs (index 3) is the observation-noise
# log-std used ONLY by the data term (Gaussian NLL); it is NOT threaded into the solve param `pf`.
# Every field-specific block offset is `NHYP + ...`, so inserting/removing a hyper slot touches
# this one constant — no raw `v[3...]` index arithmetic survives downstream.
const NHYP = 3

"Flat-Vector layout helper for Stage-1/2 trained params `[logℓ, logσ, logσ_obs, vec(w)]` (lognoise is fixed on the field)."
struct FieldLayout; n::Int; d::Int; end

# Shooting + uncertainty selector types (named by users; methods live in the ext).
struct SingleShooting end
struct MultipleShooting; nsegments::Int; λ::Float64; λ0::Float64; end
MultipleShooting(; nsegments, λ=100.0, λ0=1e4) = MultipleShooting(nsegments, λ, λ0)
struct PULL end
struct Pathwise; n::Int; end
Pathwise(; n=128) = Pathwise(n)

# Stubs implemented in MagpieSciMLExt (require OrdinaryDiffEq + SciMLSensitivity).
# Declared in CORE so the ext can EXTEND them and `using Magpie: train!` resolves
# (an extension cannot inject new standalone names into the parent namespace).
function train! end
function propagate end
function posterior_gps end
function posterior_sparsegps end
# `posterior` is the new canonical solver-free reconstruction generic (unifies
# posterior_gps/posterior_sparsegps). Task 1.1 defines the generic + "not loaded" fallback
# and EXPORTS it; the ext bodies + the alias unification land in Task 1.2 (so the existing
# ext-defined `posterior_gps`/`posterior_sparsegps` methods are left intact this phase).
function posterior end
train!(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
propagate(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
posterior_gps(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
posterior_sparsegps(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
posterior(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")

"""Multi-output SVGP field: ONE shared set of `M` inducing points `Z` (in state space), per-output
variational `(μ, L_S)`, `dout` independent outputs. Trained vector (verified shared-Z layout):
`[logℓ, logσ, logσ_obs, vec(Z)(D·M), vec(μ)(M·dout), vec(L_S)(dout·nLS(M))]`; jitter is fixed on the field."""
struct SVGPField{Tp,TZ} <: GPField
    prior::Tp; Z0::TZ; M::Int; dout::Int; D::Int; jitter::Float64; v0::Vector{Float64}
end

# ---------------------------------------------------------------------------
# Pure field core — no SciML import.
# ---------------------------------------------------------------------------

"""
    ExactGPField(kernel, Z; d, mean, logℓ0, logσ0, lognoise)

Convenience constructor. Builds the zero-mean prior and default flat params
`v0 = [logℓ0, logσ0, logσ_obs0, zeros(n*d)]` (lognoise is NOT a trained slot; logσ_obs is the
trained observation-noise log-std used by the Gaussian NLL data term, init `log(0.1)`).
"""
# lognoise=log(1e-2): the fixed anchor-value jitter. 1e-2 (not 1e-4) keeps K_ZZ well-conditioned so the
# Mooncake Cholesky-solve BACKWARD (potrs) doesn't hit SingularException when the optimizer explores
# long lengthscales / near-duplicate anchors — `_chol(check=false)` only guards the forward. Verified:
# 1e-2 removes the exception AND recovers LV better (sol_rmse 0.04 vs the fragile 1e-4).
function ExactGPField(kernel::Kernel, Z; d::Int=1, mean=AbstractGPs.ZeroMean(),
                      logℓ0=0.0, logσ0=0.0, logσ_obs0=log(0.1), lognoise=log(1e-2))
    n = length(Z)
    v0 = vcat(logℓ0, logσ0, logσ_obs0, zeros(n*d))   # lognoise NOT trained; logσ_obs IS trained (data-term only)
    ExactGPField(AbstractGPs.GP(mean, kernel), collect(Z), n, d, Float64(lognoise), v0)
end

"""Output-scaled squared-exponential kernel: `exp(2logσ) * SE(exp(logℓ))`."""
_kernel(logℓ, logσ) = exp(2logσ) * with_lengthscale(SqExponentialKernel(), exp(logℓ))

# Peel a ScaledKernel (exp(2logσ)*with_lengthscale(...)) to recover ℓ from the inner kernel.
# The field kernel returned by _kernel is a ScaledKernel; _lengthscale in fit.jl handles the
# inner TransformedKernel (with_lengthscale(...)); this peel forwards to that method.
_lengthscale(k::KernelFunctions.ScaledKernel) = _lengthscale(k.kernel)

# Layout accessors — trained vector is [logℓ, logσ, logσ_obs, vec(w)]; lognoise lives on the field.
nw(L::FieldLayout) = L.n * L.d
# hyp exposes ALL three hyper slots; logσ_obs is consumed only by the data term (not the solve pf).
hyp(L::FieldLayout, v) = (logℓ=v[1], logσ=v[2], logσ_obs=v[3])
# w-block lives AFTER the hyper prefix: indices NHYP+1 .. NHYP+nw.
wmat(L::FieldLayout, v) = reshape(v[NHYP+1 : NHYP+nw(L)], L.n, L.d)

# --- GPField protocol: ExactGPField (layout [logℓ, logσ, vec(w)]; lognoise fixed on field) ---

"""
    unpack(field::ExactGPField, v) -> (logℓ, logσ, w)

Flat trained vector → named params. `w` is an `n×d` matrix of anchor weights.
"""
function unpack(field::ExactGPField, v)
    L = FieldLayout(field.n, field.d)
    h = hyp(L, v)
    (logℓ=h.logℓ, logσ=h.logσ, logσ_obs=h.logσ_obs, w=wmat(L, v))
end

"""
    regularizer(field::ExactGPField, v; λ, logℓ_ref, s, λσ, sσ) -> Real

Hyperparameter priors: a logℓ Gaussian (breaks the ℓ–σ ridge) plus a weak logσ Gaussian.
Pure — no solver. Evaluated once per loss call.
"""
function regularizer(field::ExactGPField, v; λ=1.0, logℓ_ref=0.0, s=0.5, λσ=1.0, sσ=1.0, _kw...)
    p = unpack(field, v)
    λ*(p.logℓ - logℓ_ref)^2/(2s^2) + λσ*p.logσ^2/(2sσ^2)
end

"""
    solve_alpha(field, logℓ, logσ, lognoise, w) -> Matrix{n×d}

`α = (K_ZZ + σ_n² I)⁻¹ w` via a single shared Cholesky through `_chol`.
Call INSIDE the loss so `∂α/∂θ` stays alive through autodiff.
"""
function solve_alpha(field::ExactGPField, logℓ, logσ, lognoise, w)
    k = _kernel(logℓ, logσ)
    # RELATIVE jitter exp(lognoise)·σ² (σ²=exp(2logσ) is the kernel diagonal). Scaling with σ² bounds
    # cond(K_ZZ) ≈ 1 + n/exp(lognoise) regardless of how far logσ/logℓ drift — an ABSOLUTE jitter goes
    # negligible once logσ grows, and the Mooncake Cholesky-solve BACKWARD then throws SingularException.
    K = kernelmatrix(k, field.Z) + exp(lognoise + 2*logσ) * I
    return _chol(K) \ w
end

"""
    gpfield(field, u, pf) -> Vector{d}

Forward field callable. `pf = [logℓ, logσ, vec(α)...]`; `Z` is fixed (closed over `field`).
Returns `kuZ' * α` as a length-`d` vector.
"""
function gpfield(field::ExactGPField, u, pf)
    k   = _kernel(pf[1], pf[2])
    kuZ = [k(u, z) for z in field.Z]
    α   = reshape(@view(pf[3:end]), field.n, field.d)
    return vec(kuZ' * α)
end

# ---------------------------------------------------------------------------
# SVGPField convenience constructor + layout helpers. Task 7b.
# ---------------------------------------------------------------------------

"""
    SVGPField(kernel, Z0; dout, mean, logℓ0, logσ0, jitter) -> SVGPField

Convenience constructor for a multi-output SVGP field with `dout` independent outputs
sharing one set of `M` inducing points `Z0` (in state space).

Initial flat params:  `v0 = [logℓ0, logσ0, logσ_obs0, vec(Z)(D·M), vec(μ)(M·dout), vec(L_S)(dout·nLS(M))]`.
`μ0 = 0` (prior mean); `L_S` diag raw=0 ⇒ exp=1 ⇒ S=I, KL=0. `logσ_obs0=log(0.1)` is the trained
observation-noise log-std used by the Gaussian NLL data term (NOT the kzz conditioning jitter).
`jitter` is a RELATIVE factor (σ²-scaled, not absolute) — default 1e-4 matches `L_ZZ_factor`.
"""
function SVGPField(kernel::Kernel, Z0::AbstractVector; dout::Int=1, mean=AbstractGPs.ZeroMean(),
                   logℓ0=0.0, logσ0=0.0, logσ_obs0=log(0.1), jitter=1e-4)
    M = length(Z0); D = length(first(Z0))
    μ0  = zeros(M*dout)
    # diag raw=0 ⇒ exp=1 ⇒ S=I, KL=0; off-diag raw=0 as well
    Ls0 = reduce(vcat, [vcat(zeros(M), zeros(nLS(M)-M)) for _ in 1:dout])
    v0  = vcat(logℓ0, logσ0, logσ_obs0, reduce(vcat, Z0), μ0, Ls0)
    SVGPField(AbstractGPs.GP(mean, kernel), collect(Z0), M, dout, D, Float64(jitter), v0)
end

# Layout helpers — flat vector is [logℓ, logσ, logσ_obs, vec(Z)(D·M), vec(μ)(M·dout), vec(L_S)(dout·nLS(M))].
# jitter is fixed on the field (not a trained slot); logσ_obs (index 3, the NHYP prefix) is the
# data-term observation noise (not threaded into the SVGP solve pf). Block sizes:
#   Z:  D·M     (starts at NHYP+1)
#   μ:  M·dout  (starts after Z)
#   L_S: dout·nLS(M) (starts after μ)
# All offsets route through NHYP — no raw `v[3...]` index arithmetic survives. The svgp_* accessors
# below are themselves the SVGP-layout source of truth (each block offset is computed from NHYP).

"Extract inducing locations as a D×M matrix from flat param vector `v`."
svgp_Z(f::SVGPField, v) = reshape(v[NHYP+1 : NHYP+f.D*f.M], f.D, f.M)

"Extract variational mean as an M×dout matrix from flat param vector `v`."
svgp_μ(f::SVGPField, v) = reshape(v[NHYP+f.D*f.M+1 : NHYP+f.D*f.M+f.M*f.dout], f.M, f.dout)

"Extract raw L_S flat vector for output `i` from flat param vector `v`."
function svgp_Lsblk(f::SVGPField, v, i)
    o = NHYP + f.D*f.M + f.M*f.dout
    v[o+(i-1)*nLS(f.M)+1 : o+i*nLS(f.M)]
end

# --- GPField protocol: SVGPField (layout [logℓ, logσ, vec(Z), vec(μ), vec(L_S)]; jitter fixed) ---

"""
    unpack(field::SVGPField, v) -> (logℓ, logσ, logσ_obs, Z, μ, Ls)

Flat trained vector → named params. `Z` is `D×M`, `μ` is `M×dout`,
`Ls` is a `Vector` of `dout` `LowerTriangular` variational Cholesky factors.
"""
function unpack(field::SVGPField, v)
    (logℓ=v[1], logσ=v[2], logσ_obs=v[3], Z=svgp_Z(field, v), μ=svgp_μ(field, v),
     Ls=[unpack_LS(svgp_Lsblk(field, v, i), field.M) for i in 1:field.dout])
end

"""
    regularizer(field::SVGPField, v; λ, logℓ_ref, s) -> Real

Whitened collapsed KL across all outputs (summed once, not per trajectory) plus the
logℓ Gaussian prior. Pure — no solver.
"""
function regularizer(field::SVGPField, v; λ=1.0, logℓ_ref=0.0, s=0.5, _kw...)
    p = unpack(field, v)
    kl = sum(svgp_kl(p.μ[:,i], p.Ls[i]) for i in 1:field.dout)
    kl + λ*(p.logℓ - logℓ_ref)^2/(2s^2)
end

# ---------------------------------------------------------------------------
# Pure SVGP math — no SciML. Task 7.
# ---------------------------------------------------------------------------

"""
    nLS(M) -> Int

Number of free parameters in a lower-triangular M×M matrix (the variational
factor `L_S` stored in flat form).
"""
nLS(M) = (M*(M+1)) ÷ 2

"""
    unpack_LS(raw, M) -> LowerTriangular

Rebuild the lower-triangular variational Cholesky factor `L_S` from a flat raw
vector of length `nLS(M)`. Diagonal entries are `exp(raw[diag])` for positivity;
off-diagonal entries are taken as-is. Column-major lower layout.
"""
function unpack_LS(raw::AbstractVector, M::Int)
    L = zeros(eltype(raw), M, M)
    idx = 1
    for j in 1:M, i in j:M
        L[i, j] = (i == j) ? exp(raw[idx]) : raw[idx]
        idx += 1
    end
    return LowerTriangular(L)
end

"""
    svgp_kl(μ, S_L) -> Real

Whitened collapsed KL divergence KL[q(v) ‖ p(v)] where `q(v) = N(μ, S_L S_L')` and
`p(v) = N(0, I)`. Prior on whitened variable `v = L_ZZ' \\ u` is standard normal, so:

    KL = 0.5 * (‖S_L‖²_F + ‖μ‖² - M - 2 Σ log diag(S_L))
"""
svgp_kl(μ, S_L) = (M = length(μ); 0.5*(sum(abs2, S_L) + dot(μ, μ) - M - 2*sum(log, diag(S_L))))

"""
    L_ZZ_factor(prior, Z; jitter=1e-4) -> LowerTriangular

Cholesky factor `L_ZZ` of the inducing-point kernel matrix `K(Z, Z)`.

Uses a **relative** jitter `jitter * σ²` (where `σ² ≈ mean diagonal of K_ZZ`) to keep
`K_ZZ` positive-definite regardless of how far the signal variance drifts during
optimization. An absolute jitter goes negligible once `σ²` grows, which causes
`SingularException` in Mooncake's Cholesky backward — the same pattern that hit
Stage 1's ExactGP training (fixed there by `exp(lognoise + 2*logσ)`).

Default `jitter=1e-4` is a relative factor (not absolute).
"""
function L_ZZ_factor(prior, Z; jitter=1e-4)
    K = AbstractGPs.cov(prior, Z)
    s2 = sum(i -> K[i,i], 1:size(K,1)) / size(K,1)   # ≈ σ² (mean diagonal)
    return _chol(K + (jitter * s2) * I).L
end

"""
    svgp_moments(prior, Z, L_ZZ, α, L_S, u) -> (μ_star, σ²)

Whitened predictive mean and variance at a single point `u`.

    A     = L_ZZ \\ k(Z, u)               # whitened cross-covariance
    μ_star = m(u) + k(Z, u)' α           # posterior mean (α = L_ZZ' \\ μ_v)
    σ²    = k(u,u) - A'A + ‖L_S' A‖²   # posterior variance with variational correction
"""
function svgp_moments(prior, Z, L_ZZ, α, L_S, u)
    kZu  = vec(AbstractGPs.cov(prior, Z, [u]))
    A    = L_ZZ \ kZu
    μ_star = only(AbstractGPs.mean(prior, [u])) + dot(kZu, α)
    σ2   = only(AbstractGPs.var(prior, [u])) - dot(A, A) + sum(abs2, L_S' * A)
    return μ_star, σ2
end

# ---------------------------------------------------------------------------
# SparseGP — full AbstractGPModel via svgp_moments. Task 7.
# ---------------------------------------------------------------------------

"""
    SparseGP(prior, Z, μ, L_S; jitter=1e-4) -> SparseGP

Convenience constructor from variational parameters `μ` (whitened mean) and `L_S`
(lower-triangular variational Cholesky). Computes and caches `L_ZZ` and `α = L_ZZ' \\ μ`.
"""
function SparseGP(prior, Z, μ::AbstractVector, L_S; jitter=1e-4)
    L_ZZ = L_ZZ_factor(prior, Z; jitter)
    SparseGP(prior, Z, L_ZZ' \ μ, L_ZZ, L_S)
end

"""Posterior mean at a single input `u` (scalar)."""
predmean(g::SparseGP, u) = svgp_moments(g.prior, g.Z, g.L_ZZ, g.α, g.L_S, u)[1]

"""Posterior mean vector at `xs`."""
Statistics.mean(g::SparseGP, xs::AbstractVector) = [predmean(g, x) for x in xs]

"""Posterior marginal variance vector at `xs`."""
Statistics.var(g::SparseGP, xs::AbstractVector) =
    [svgp_moments(g.prior, g.Z, g.L_ZZ, g.α, g.L_S, x)[2] for x in xs]

"""
    cov(g::SparseGP, xs, ys) -> Matrix

Posterior cross-covariance between input sets `xs` and `ys`, including the
variational correction from `L_S`.

    Ax = L_ZZ \\ K(Z, xs),   Ay = L_ZZ \\ K(Z, ys)
    Cov = K(xs, ys) - Ax' Ay + (L_S' Ax)' (L_S' Ay)
"""
function Statistics.cov(g::SparseGP, xs::AbstractVector, ys::AbstractVector)
    Ax = g.L_ZZ \ AbstractGPs.cov(g.prior, g.Z, xs)   # M × |xs|
    Ay = g.L_ZZ \ AbstractGPs.cov(g.prior, g.Z, ys)   # M × |ys|
    AbstractGPs.cov(g.prior, xs, ys) .- Ax'Ay .+ (g.L_S'Ax)' * (g.L_S'Ay)
end

"""Posterior covariance matrix within `xs` (symmetric)."""
Statistics.cov(g::SparseGP, xs::AbstractVector) = Matrix(Symmetric(Statistics.cov(g, xs, xs)))

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Decoupled (Matheron) GP sampler — pure core, no SciML, no autodiff. Task 8.
# ---------------------------------------------------------------------------

"""
    DecoupledGPSample

A single pathwise sample from a GP posterior via the Matheron (decoupled) trick:

    f(x) = prior_rff(x)  +  Σⱼ k(x, zⱼ) vⱼ

where `prior_rff(x) = wᵀ φ(x)` is a random Fourier feature prior draw and
`v = K(Z,Z)⁻¹(u − Φw)` is the inducing-point correction (given one draw `u`
of the function values at `Z`).

Callable: `(s::DecoupledGPSample)(x::AbstractVector) -> Float64`.
"""
struct DecoupledGPSample{TZ,Tk}
    w::Vector{Float64}; ω::Matrix{Float64}; b::Vector{Float64}; D::Int
    v::Vector{Float64}; Z::TZ; kernel::Tk
end

"""
    _rff_features(x, ω, b, D) -> Vector{Float64}

Random Fourier features: `√(2/D) .* cos.(ω' * x .+ b)`.
"""
_rff_features(x, ω, b, D) = sqrt(2/D) .* cos.(ω' * x .+ b)

"""
    build_decoupled_sample(kernel, Z, u; ℓ, σ, D, jitter, rng) -> DecoupledGPSample

Build one pathwise sample from the GP posterior at inducing points `Z` with observed
(drawn) function values `u`.

- `ℓ`: SE lengthscale (spectral density is `N(0, I/ℓ²)`)
- `σ`: output scale (RFF prior amplitude)
- `D`: number of random Fourier features (default 512)
- `jitter`: diagonal jitter on `K(Z,Z)` for numerical stability (default 1e-6)
- `rng`: random number generator
"""
function build_decoupled_sample(kernel, Z, u; ℓ::Real, σ::Real=1.0, D::Int=512, jitter=1e-6,
                                rng=Random.default_rng())
    din = length(first(Z))
    ω = randn(rng, din, D) ./ ℓ                  # SE spectral density N(0, I/ℓ²)
    b = rand(rng, D) .* (2π)
    w = (σ .* randn(rng, D))                      # output-scale enters the RFF prior amplitude
    Φw = [dot(w, _rff_features(z, ω, b, D)) for z in Z]
    K  = kernelmatrix(kernel, Z) + jitter*I
    v  = _chol(K) \ (u .- Φw)
    return DecoupledGPSample(w, ω, b, D, v, collect(Z), kernel)
end

"""
    (s::DecoupledGPSample)(x::AbstractVector) -> Float64

Evaluate the decoupled GP sample at input `x`:
    prior RFF value + inducing-point update correction.
"""
function (s::DecoupledGPSample)(x::AbstractVector)
    prior_x  = dot(s.w, _rff_features(x, s.ω, s.b, s.D))
    update_x = dot([s.kernel(x, z) for z in s.Z], s.v)
    return prior_x + update_x
end

# ---------------------------------------------------------------------------

"""
    kmeans_anchors(X, k; iters, rng) -> Vector{Vector{Float64}}

Tiny pure-Julia Lloyd's k-means on COLUMNS of `X` (each column = one d-dim state).
Returns `k` cluster centres as `Vector{Vector{Float64}}`.
"""
function kmeans_anchors(X::AbstractMatrix, k::Int; iters::Int=50, rng=Random.default_rng())
    d, N = size(X); @assert k <= N
    C = [Vector{Float64}(X[:, j]) for j in randperm(rng, N)[1:k]]
    assign = zeros(Int, N)
    for _ in 1:iters
        for i in 1:N
            best, bd = 1, Inf
            for c in 1:k
                dist = 0.0
                @inbounds for r in 1:d; dist += (X[r, i] - C[c][r])^2; end
                dist < bd && (bd = dist; best = c)
            end
            assign[i] = best
        end
        for c in 1:k
            m = findall(==(c), assign); isempty(m) && continue
            C[c] = vec(sum(@view(X[:, m]); dims=2)) ./ length(m)
        end
    end
    return C
end
