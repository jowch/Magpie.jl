module MagpieSciMLExt

using Magpie
using Magpie: _chol, predmean, ExactGP, ExactGPField, SparseGP, SVGPField, FieldLayout, gpfield, solve_alpha
using OrdinaryDiffEq
using SciMLSensitivity
import SciMLBase
using LinearAlgebra
using KernelFunctions, AbstractGPs
import DifferentiationInterface as DI
import Mooncake
import Optimization
import OptimizationOptimJL: LBFGS
import OptimizationOptimisers: Adam

# MooncakeVJP is UNEXPORTED — bind once (SciMLSensitivityMooncakeExt auto-fires; Mooncake is a core hard dep).
const MOONCAKEVJP = SciMLSensitivity.MooncakeVJP()
const DEFAULT_SENSEALG = GaussAdjoint(autojacvec = MOONCAKEVJP)

# ===========================================================================
# Unified GP-UDE loss skeleton. Shooting is orthogonal to field type:
#
#   field_loss(field, shooting, data) =
#       shooting_data_term(field, shooting, field_rhs(field, v), data) + regularizer(field, v)
#
# `field_rhs(field, v)` builds the in-loss α/Cholesky + the `du += f(u)` closure (per field;
# R1: α recomputed in-loss, threaded into `pf`, NEVER closure-captured). `shooting_data_term`
# is the ONLY place segmentation lives — field-agnostic — and returns ONLY the data fit (the
# regularizer/KL is added ONCE per loss call by `field_loss`, OUTSIDE the trajectory loop).
#
# `data` is always a `Vector{<:Tuple}`: single-shooting passes a 1-element `[(ts, X)]`.
# ===========================================================================

"""
    field_loss(field, shooting, data; kw...) -> (v -> Real)

Build the scalar GP-UDE loss `v -> shooting_data_term(...) + regularizer(field, v)`. The data
term and the regularizer split cleanly: the regularizer (priors + SVGP KL) is evaluated once
per `v`, outside any trajectory/segment loop.
"""
# logσ_obs (the Gaussian-NLL observation-noise log-std) lives at index NHYP of `v` and is consumed
# ONLY by the data term — it is NOT in the solve `pf`. field_loss extracts it from `v` and threads it
# into shooting_data_term; callers may also pass logσ_obs explicitly (it wins via the trailing kw...).
field_loss(field, shooting, data; kw...) =
    v -> shooting_data_term(field, shooting, field_rhs(field, v), data;
                            logσ_obs=v[Magpie.NHYP], kw...) +
         Magpie.regularizer(field, v; kw...)

# --- field_rhs: per-field in-loss α/Cholesky + the `du += f(u)` closure (R1) ---

"""
    field_rhs(cf::CompositeField, v) -> (pf, rhs!)

Build the CompositeField RHS: delegates α/Cholesky to the inner GP field's `field_rhs` and
wraps the returned closure so it does `du .= cf.known(u,t)` FIRST, then adds the GP residual.
α is threaded in `pf` via the inner field — never closure-captured (R1).
The `known` closure is called every RHS evaluation but carries no trained params.
"""
function field_rhs(cf::Magpie.CompositeField, v)
    pf, _ = field_rhs(cf.gp, v)   # inner GP: solves α, builds pf (inner closure unused — CompositeField rebuilds via gpfield)
    known = cf.known                         # do NOT closure-capture α or pf — only the fixed known fn
    function rhs!(du, u, _pf, t; known_physics=(u,t)->zero(u))
        du .= known(u, t)                    # known physics first (baked in; ignores kwarg known_physics)
        du .+= gpfield(cf.gp, u, _pf)       # GP residual (α threaded in _pf, R1)
        return nothing
    end
    return (pf, rhs!)
end

"""
    field_rhs(field::ExactGPField, v) -> rhs!

Build the ExactGPField RHS at trained params `v`: solves `α = (K_ZZ + σ_n² I)⁻¹ w` in-loss
(via `solve_alpha`, so `∂α/∂θ` stays alive), threads `α` into `pf = [logℓ, logσ, vec(α)]`, and
returns `rhs!(du, u, pf, t)` adding `kuZ'·α`. α is NEVER closure-captured.
"""
function field_rhs(field::ExactGPField, v)
    L  = FieldLayout(field.n, field.d)
    h  = Magpie.hyp(L, v)
    α  = solve_alpha(field, h.logℓ, h.logσ, field.lognoise, Magpie.wmat(L, v))  # lognoise FIXED, in-loss
    pf = vcat(h.logℓ, h.logσ, vec(α))                                          # α threaded into pf (R1)
    rhs!(du, u, _pf, t; known_physics) = (du .= known_physics(u, t); du .+= gpfield(field, u, _pf); nothing)
    return (pf, rhs!)
end

"""
    field_rhs(field::SVGPField, v) -> (pf, rhs!)

Build the SVGPField RHS at trained params `v`: forms the ONE shared-Z Cholesky `L_ZZ` in-loss
(relative jitter `field.jitter·σ²`), solves `α = L_ZZ'\\μ`, threads `Z` (trainable) and `α` into
`pf = [logℓ, logσ, vec(Z), vec(α)]`, and returns `rhs!` adding `du[i] += Σⱼ k(u,Zⱼ)·αⱼ`.
"""
function field_rhs(field::SVGPField, v)
    M, dout, D = field.M, field.dout, field.D
    logℓ, logσ = v[1], v[2]
    k    = Magpie._kernel(logℓ, logσ)
    Z    = Magpie.svgp_Z(field, v)        # D×M
    μ    = Magpie.svgp_μ(field, v)        # M×dout
    Zvec = [Z[:,j] for j in 1:M]
    # RELATIVE in-loss jitter: field.jitter · σ² — must match posterior/L_ZZ_factor
    jit  = field.jitter * exp(2*logσ)
    L_ZZ = _chol(kernelmatrix(k, Zvec) + jit*I).L   # ONE shared Cholesky (shared Z)
    α    = L_ZZ' \ μ                                  # M×dout
    pf   = vcat(logℓ, logσ, vec(Z), vec(α))          # Z trainable + α threaded into pf (R1)
    function rhs!(du, u, _pf, t; known_physics)
        du .= known_physics(u, t)
        kk = Magpie._kernel(_pf[1], _pf[2])
        Zr = reshape(_pf[3:2+D*M], D, M)
        αr = reshape(_pf[3+D*M:2+D*M+M*dout], M, dout)
        for i in 1:dout
            du[i] += sum(kk(u, @view Zr[:,j]) * αr[j,i] for j in 1:M)
        end
        return nothing
    end
    return (pf, rhs!)
end

# --- shooting_data_term: the ONLY place segmentation lives; field-agnostic. ---
# Returns ONLY the data fit (no regularizer — field_loss adds it once, outside the loop).
# `data` is always a Vector{<:Tuple} of (ts, X); single-shooting = a 1-element vector.

"""
    shooting_data_term(field, ::SingleShooting, (pf, rhs!), data; logσ_obs, u0, tspan, ...) -> Real

Single-shooting data fit (Gaussian NLL): for each `(ts, X)` in `data`, integrate the RHS over
`tspan` saving at `ts` (R2: `Array(sol)`, never `sol[:,i]`) and accumulate
`Σ (Aᵢ−Xᵢ)²/(2σ_obs²)`; add the normalizer `(Nd/2)·log(2π σ_obs²)` once over all scalar
observations. `σ_obs² = exp(2·logσ_obs)`. logσ_obs is the trained observation-noise log-std
(separate from the K_ZZ conditioning jitter). For single trajectories `u0` defaults to col 1.
"""
function shooting_data_term(field, ::Magpie.SingleShooting, (pf, rhs!), data;
                            logσ_obs, u0=nothing, tspan=nothing, known_physics=(u,t)->zero(u),
                            solver=Tsit5(), sensealg=DEFAULT_SENSEALG, _kw...)
    f!(du, u, p, t) = rhs!(du, u, p, t; known_physics)
    T = eltype(pf)
    σ2 = exp(2*logσ_obs)
    sse = zero(T)
    Nd  = 0
    for (ts, X) in data
        ic  = u0 === nothing ? collect(X[:, 1]) : u0
        tsp = tspan === nothing ? (first(ts), last(ts)) : tspan
        sol = solve(ODEProblem(f!, ic, tsp, pf), solver; saveat=ts, sensealg)
        A = Array(sol)                                  # R2: Array(sol), never sol[:,i]
        size(A) == size(X) || return convert(T, 1e6)    # divergence guard → finite sentinel
        sse += sum(abs2, A .- X)
        Nd  += length(X)
    end
    # Gaussian NLL: SSE/(2σ²) + (Nd/2)·log(2π σ²). σ_obs is scale-not-ratio (identifiable).
    return sse/(2σ2) + (Nd/2)*log(2π*σ2)
end

"""
    shooting_data_term(field, ms::MultipleShooting, (pf, rhs!), data; logσ_obs, tspan, s0, ...) -> Real

Multiple-shooting data fit over a single trajectory `data == [(ts, X)]`: splits into `ms.nsegments`
segments with free per-segment initial nodes `s0` (d×S). The endpoint-vs-data misfit is the Gaussian
NLL (`Σ(endp−X)²/(2σ_obs²) + (Sd/2)·log(2π σ_obs²)`, σ_obs²=exp(2·logσ_obs)) — identical to single
shooting; the continuity penalty `ms.λ` and node-0 anchor penalty `ms.λ0` are soft constraints (own
weights), not data likelihood, so they stay un-scaled. `s0` is supplied by the caller.
"""
function shooting_data_term(field, ms::Magpie.MultipleShooting, (pf, rhs!), data;
                            logσ_obs, s0, known_physics=(u,t)->zero(u),
                            solver=Tsit5(), sensealg=DEFAULT_SENSEALG, _kw...)
    (ts, X) = only(data)
    S = ms.nsegments
    seg_idx = round.(Int, range(1, length(ts); length=S+1))
    seg_t   = [ts[i] for i in seg_idx]
    f!(du, u, p, t) = rhs!(du, u, p, t; known_physics)
    σ2 = exp(2*logσ_obs)
    dataerr = cont = zero(eltype(pf))
    Nd = 0
    for i in 1:S
        sol  = solve(ODEProblem(f!, s0[:, i], (seg_t[i], seg_t[i+1]), pf), solver;
                     saveat=[seg_t[i+1]], sensealg)
        endp = Array(sol)[:, end]                        # R2: Array(sol) before indexing
        dataerr += sum(abs2, endp .- X[:, seg_idx[i+1]])
        Nd += length(endp)
        i < S && (cont += sum(abs2, endp .- s0[:, i+1]))
    end
    # Gaussian NLL on the data-misfit term (matches SingleShooting); penalties keep their own weights.
    return dataerr/(2σ2) + (Nd/2)*log(2π*σ2) + ms.λ*cont + ms.λ0*sum(abs2, s0[:, 1] .- X[:, 1])
end

# ---------------------------------------------------------------------------
# Signature-preserving WRAPPERS onto the skeleton (retire in Phase 6; 8 call sites depend
# on them). They forward to field_loss, preserving exact argument order + reg conventions.
# ---------------------------------------------------------------------------

# Stage-1 single-shooting loss. Old reg = λ·logℓ + λσ·logσ (ExactGPField regularizer default).
make_loss(field::ExactGPField, L::FieldLayout, u0, tspan, ts, X; kw...) =
    field_loss(field, Magpie.SingleShooting(), [(collect(ts), X)]; u0=u0, tspan=tspan, kw...)

# build_loss: Stage-1 SingleShooting + Stage-2 MultipleShooting dispatch.
# Arg order preserved: (field, L, u_data, t_data, tspan, shooting; kw...).
function build_loss(field, L, u_data, t_data, tspan, ::Magpie.SingleShooting; kw...)
    u0 = u_data isa AbstractMatrix ? collect(u_data[:, 1]) : [u_data[1]]
    field_loss(field, Magpie.SingleShooting(), [(collect(t_data), u_data)]; u0=u0, tspan=tspan, kw...)
end

# Stage-2 multiple shooting. The trained vector is [logℓ, logσ, logσ_obs, vec(w), vec(s0)] (s0 is d×S);
# the loss extracts s0 from `v` then forwards to shooting_data_term. Old MS reg was logℓ-only
# (no λσ term) — preserved by passing λσ=0.0 to the field regularizer. The s0 block starts AFTER the
# hyper prefix + the w-block: offset NHYP + nwL (routed through Magpie.NHYP — no hardcoded 3).
function build_loss(field::ExactGPField, L::FieldLayout, u_data, t_data, tspan,
                    ms::Magpie.MultipleShooting; kw...)
    S   = ms.nsegments
    nwL = L.n * L.d
    off = Magpie.NHYP + nwL                                     # s0 starts after [logℓ,logσ,logσ_obs,vec(w)]
    regkw = merge((λσ=0.0,), values(kw))                        # old MS reg: logℓ-only (λσ=0 unless overridden)
    return function loss(v)
        s0 = reshape(v[off+1 : off + field.d*S], field.d, S)
        shooting_data_term(field, ms, field_rhs(field, v), [(collect(t_data), u_data)];
                           s0=s0, logσ_obs=v[Magpie.NHYP], kw...) +
            Magpie.regularizer(field, v; regkw...)
    end
end

# ---------------------------------------------------------------------------
# _init_vec: build initial optimisation vector from the field + shooting strategy
# ---------------------------------------------------------------------------

# SingleShooting: optimise field params only — just copy v0
_init_vec(field, u_data, t_data, ::Magpie.SingleShooting) = copy(field.v0)

# MultipleShooting: append per-segment initial nodes (seeded from data) to the field params
function _init_vec(field, u_data, t_data, ms::Magpie.MultipleShooting)
    idx = round.(Int, range(1, length(t_data); length=ms.nsegments+1))
    s0  = hcat([collect(u_data[:, idx[i]]) for i in 1:ms.nsegments]...)   # d × S nodes from data
    return vcat(copy(field.v0), vec(s0))
end

# ---------------------------------------------------------------------------
# train!: ONE optimizer driver for every GPField — Optimization.jl ADAM→LBFGS, outer Mooncake.
# Field-specific loss/init via the small internal `_train_loss`/`_train_init` helpers; the
# ADAM→LBFGS recipe + param storage are shared. `data` normalized to Vector{<:Tuple}
# (a bare `(ts, X)` tuple is wrapped to a 1-element vector).
# ---------------------------------------------------------------------------

# Normalize the user's `data` arg to a Vector{<:Tuple} of trajectories.
_as_trajectories(data::AbstractVector{<:Tuple}) = data
_as_trajectories(data::Tuple) = [data]

# Per-field loss builder (drives the unified skeleton; preserves each field's trained-vector layout).
function _train_loss(field::ExactGPField, trajs, shooting, tspan; kw...)
    (t_data, u_data) = only(trajs)                 # Exact training is single-trajectory
    L = FieldLayout(field.n, field.d)
    build_loss(field, L, u_data, t_data, tspan, shooting; kw...)
end
_train_loss(field::SVGPField, trajs, ::Magpie.SingleShooting, tspan; kw...) =
    svgp_elbo_loss(field, trajs; tspan, kw...)

# CompositeField: forward to field_loss (field_rhs for CompositeField is defined above).
# The trained-vector layout IS the inner gp's (CompositeField delegates all layout to cf.gp).
function _train_loss(cf::Magpie.CompositeField, trajs, shooting, tspan; kw...)
    field_loss(cf, shooting, trajs; tspan=tspan, kw...)
end

# Per-field initial optimisation vector (Exact MS appends per-segment s0 nodes; SVGP = field.v0).
_train_init(field::ExactGPField, trajs, shooting) =
    (tu = only(trajs); _init_vec(field, tu[2], tu[1], shooting))
_train_init(field::SVGPField, trajs, ::Magpie.SingleShooting) = copy(field.v0)
# CompositeField: delegates to the inner gp's v0 (layout is identical).
_train_init(cf::Magpie.CompositeField, trajs, shooting) =
    (tu = only(trajs); _init_vec(cf.gp, tu[2], tu[1], shooting))

# Default recipe is ADAM warm-up → LBFGS polish (verified: pure LBFGS-from-zero blows the weights up;
# ADAM's bounded steps find the basin first). Set `adam_iters=0` for pure-LBFGS (diagnostics only).
function Magpie.train!(field::GPField, data;
                       shooting=Magpie.SingleShooting(),
                       tspan=nothing,
                       ad=DI.AutoMooncake(; config=nothing),
                       adam_lr=0.05, adam_iters=1000, optimizer=LBFGS(), maxiters=200, kw...)
    trajs = _as_trajectories(data)
    tsp   = tspan === nothing ? (first(trajs[1][1]), last(trajs[1][1])) : tspan
    loss  = _train_loss(field, trajs, shooting, tsp; kw...)
    v_init = _train_init(field, trajs, shooting)
    optf = Optimization.OptimizationFunction((v, _p) -> loss(v), ad)
    v = v_init
    if adam_iters > 0
        s1 = Optimization.solve(Optimization.OptimizationProblem(optf, v), Adam(adam_lr); maxiters=adam_iters)
        v = s1.u
    end
    sol = Optimization.solve(Optimization.OptimizationProblem(optf, v), optimizer; maxiters)
    field.v0 .= sol.u[1:length(field.v0)]          # store FIELD prefix; drop s0 for MultipleShooting
    return field, sol.u
end

# ---------------------------------------------------------------------------
# posterior: canonical solver-free reconstruction. posterior_gps/posterior_sparsegps
# are thin aliases forwarding here (8 call sites + exports unchanged).
# ---------------------------------------------------------------------------

# NOTE: must be `Magpie.posterior` (qualified) to EXTEND the core stub; an unqualified
# `function posterior` here would create MagpieSciMLExt.posterior and shadow it, leaving the
# public `Magpie.posterior` with only the "not loaded" stub. Same for the aliases below.
Magpie.posterior(field::GPField) = Magpie.posterior(field, field.v0)

# CompositeField: posterior IS the residual GP (the trained part). `known` is fixed, not returned.
Magpie.posterior(cf::Magpie.CompositeField, v) = Magpie.posterior(cf.gp, v)

# ExactGPField → d single-output ExactGPs. Relative jitter must match solve_alpha so α == trained α.
function Magpie.posterior(field::ExactGPField, v)
    L = FieldLayout(field.n, field.d); h = Magpie.hyp(L, v)
    k = Magpie._kernel(h.logℓ, h.logσ)
    jit = exp(field.lognoise + 2*h.logσ)          # RELATIVE jitter — must match solve_alpha so α == the trained field's
    K = kernelmatrix(k, field.Z) + jit * I
    C = _chol(K)
    α = C \ Magpie.wmat(L, v)                      # (n×d) weights — reuse the Cholesky factor
    prior = AbstractGPs.GP(field.prior.mean, k)
    return [ExactGP(prior, field.Z, zeros(field.n), C, α[:, i], jit) for i in 1:field.d]
end

# Thin aliases forwarding to `posterior` (8 call sites + exports unchanged; retire in Phase 6).
Magpie.posterior_gps(field::ExactGPField) = Magpie.posterior(field)
Magpie.posterior_gps(field::ExactGPField, v) = Magpie.posterior(field, v)

# ---------------------------------------------------------------------------
# SVGPField: multi-output multi-trajectory ELBO loss (shared Z). Task 7b.
# ---------------------------------------------------------------------------

"""
    svgp_elbo_loss(field::SVGPField, trajectories; tspan, ...) -> Function

Returns a scalar loss `v -> -ELBO` over multiple trajectories sharing one SVGP field.

`trajectories` is a `Vector` of `(t_data, u_data)` pairs (each `u_data` is `dout × T`).
The data term sums over all trajectories (multi-trajectory verified).

Relative in-loss jitter: `jit = field.jitter * exp(2*logσ)` — scales with σ² so the
Cholesky stays well-conditioned when logσ drifts during training (absolute jitter goes
negligible and crashes Mooncake's backward with `SingularException`). Matches
`L_ZZ_factor`'s relative-jitter convention so reconstructed α equals the trained field's.

Z is in the param vector (trainable); never closure-captured (R1).
"""
# Thin wrapper onto the skeleton: SVGP training is single-shooting over `trajectories`.
# Preserves the exact ELBO (data + KL + logℓ prior) via field_loss → SVGP field_rhs +
# field-agnostic shooting_data_term + the SVGPField regularizer (KL + logℓ prior, once).
svgp_elbo_loss(field::SVGPField, trajectories; tspan, kw...) =
    field_loss(field, Magpie.SingleShooting(), trajectories; tspan=tspan, kw...)

# ---------------------------------------------------------------------------
# posterior(::SVGPField): reconstruct dout SparseGPs from trained params. Task 7b.
# Relative jitter field.jitter·σ² matches the in-loss Cholesky so reconstructed α == trained α.
# ---------------------------------------------------------------------------

function Magpie.posterior(field::SVGPField, v)
    logσ = v[2]
    k  = Magpie._kernel(v[1], logσ)
    Z  = Magpie.svgp_Z(field, v)
    Zvec = [Z[:,j] for j in 1:field.M]
    # RELATIVE jitter — must match field_rhs(::SVGPField) (field.jitter · σ²)
    jit  = field.jitter * exp(2*logσ)
    L_ZZ = _chol(kernelmatrix(k, Zvec) + jit*I).L
    μ    = Magpie.svgp_μ(field, v)                   # M×dout
    prior = AbstractGPs.GP(field.prior.mean, k)
    return [SparseGP(prior, Zvec, L_ZZ' \ μ[:,i], L_ZZ,
                     Magpie.unpack_LS(Magpie.svgp_Lsblk(field, v, i), field.M))
            for i in 1:field.dout]
end

# Thin aliases forwarding to `posterior` (call sites + exports unchanged; retire in Phase 6).
Magpie.posterior_sparsegps(field::SVGPField) = Magpie.posterior(field)
Magpie.posterior_sparsegps(field::SVGPField, v) = Magpie.posterior(field, v)

# ---------------------------------------------------------------------------
# PULL uncertainty propagation (Stage 4). No ODE solver — discrete moment-matching recurrence.
# Operates on already-built ExactGPs (from Magpie.update); no training involved.
# ---------------------------------------------------------------------------

import ForwardDiff

field_mean(gps, u) = [predmean(g, u) for g in gps]
pull_jacobian(gps, u) = ForwardDiff.jacobian(uu -> field_mean(gps, uu), u)
field_var(gps, u) = Diagonal([only(AbstractGPs.var(g, [u])) for g in gps])

"""
    pull_propagate(gps, u0, ts; buffer=20) -> (μs, Σs)

Propagate a Gaussian uncertainty (μ, Σ) forward through the GP field via a corrected
moment-matching recurrence (no ODE solver). `gps` is a `Vector{ExactGP}` (one per output
dimension). Returns `μs` and `Σs` — vectors of mean vectors and covariance matrices at
each time step in `ts`.

Recurrence (PULL, arXiv:2211.11103 eq 36b, with cross-cov Dₙ):
    Aₙ = I + h·Jₙ,  Jₙ = ForwardDiff Jacobian of field_mean at μₙ
    Vₙ = diag GP marginal variance at μₙ
    Dₙ = buffer-truncated cross-cov: h · Σᵢ (∏ Aₖ) · cov_f(μᵢ, μₙ)   (carries one h)
    Σₙ₊₁ = Sym(AₙΣₙAₙᵀ) + h²·Vₙ + h(AₙDₙ + DₙᵀAₙᵀ)

Note: the field-variance term is h²·Vₙ — Euler `x→x+h·f` gives `Var(h·f)=h²·Var(f)` (NOT a
white-noise rate h·Vₙ). Exact linear-field oracle is eq 21b Σ(t)=(β/a²)(1−e^{at})² (coherent),
not the white-noise (β/−2a)(1−e^{2at}); the cross-cov Dₙ realizes the coherence ("past matters").
"""
function pull_propagate(gps, u0, ts; buffer::Int=20)
    d = length(u0); μ = collect(float.(u0)); Σ = zeros(d, d)
    μs = [copy(μ)]; Σs = [copy(Σ)]; histμ = [copy(μ)]; histA = Matrix{Float64}[]
    for n in 1:(length(ts)-1)
        h = ts[n+1] - ts[n]
        A = I + h .* pull_jacobian(gps, μ)
        V = field_var(gps, μ)                                # marginal variance V_n (buffer-free term)
        Dn = _pull_Dn(gps, histμ, histA, μ, h, d; buffer)
        # PULL eq 36b: Σ_{n+1} = A Σ A' + h²·V_n + h·(A D_n + D_nᵀ A')  (D_n already carries one h, eq 37).
        # The field-VARIANCE term is h² (Euler: Var(h·f)=h²·Var(f)), NOT h (that would be a white-noise rate).
        Σ = Matrix(Symmetric(A*Σ*A' + h^2 .* Matrix(V) + h .* (A*Dn + Dn'*A')))
        for k in 1:d; Σ[k,k] < 0 && (@warn "PULL: negative variance clamped" step=n; Σ[k,k]=eps()); end
        μ = μ + h .* field_mean(gps, μ)
        push!(histμ, copy(μ)); push!(histA, A); push!(μs, copy(μ)); push!(Σs, copy(Σ))
    end
    return μs, Σs
end

"""
    _pull_Dn(gps, histμ, histA, μ, h, d; buffer) -> Matrix

Compute the cross-covariance Dₙ = h · Σᵢ (∏_{k=i+1}^{n-1} Aₖ) · cov_f(μᵢ, μₙ),
summing over PAST states only (i = 0..n-1), not the current state.

Correct recurrence (eq-37): nearest-past term (i=n-1) has empty product (I); each
earlier term accumulates one more A to the LEFT: prodA ← prodA * histA[i] where
histA[i] is the Jacobian factor at past state i (A_{i-1→i} in 0-indexed notation).
Sum starts at npast = length(histμ)-1 (excluding the current state self-term) and
runs back to lo. The key index fixes vs the original buggy loop:
  - `npast` (not `length(histμ)`) as the upper bound — excludes the self-term.
  - `histA[i]` (not `histA[i-1]`) — correct Jacobian at past state i.
"""
function _pull_Dn(gps, histμ, histA, μ, h, d; buffer::Int=20)
    Dn = zeros(d, d)
    buffer == 0 && return Dn
    npast = length(histμ) - 1                      # exclude the current state (self term)
    npast == 0 && return Dn                        # D_0 = 0 (no past states yet)
    lo = max(1, npast - buffer + 1)
    prodA = Matrix{Float64}(I, d, d)
    for i in npast:-1:lo                            # past states ν_i, nearest first
        covf = Diagonal([only(AbstractGPs.cov(gps[k], [histμ[i]], [μ])) for k in 1:d])
        Dn += prodA * covf                          # nearest-past term has product I
        i > lo && (prodA = prodA * histA[i])        # A_k = histA[i] (Jacobian at state i)
    end
    Dn .*= h
    return Dn
end

"""
    _pull_Dn_sequence(gps, u0, ts; buffer=20) -> Vector{Matrix}

Internal: propagate the mean path and return the per-step Dₙ sequence (one matrix per
time step, length = length(ts)-1). Used by tests to assert the cross-cov telescope
against a brute-force reference in a non-constant-Jacobian regime.
"""
function _pull_Dn_sequence(gps, u0, ts; buffer::Int=20)
    d = length(u0); μ = collect(float.(u0))
    histμ = [copy(μ)]; histA = Matrix{Float64}[]; Dns = Matrix{Float64}[]
    for n in 1:(length(ts)-1)
        h = ts[n+1] - ts[n]
        A = I + h .* pull_jacobian(gps, μ)
        Dn = _pull_Dn(gps, histμ, histA, μ, h, d; buffer)
        push!(Dns, copy(Dn))
        μ = μ + h .* field_mean(gps, μ)
        push!(histμ, copy(μ)); push!(histA, A)
    end
    return Dns
end


# ---------------------------------------------------------------------------
# Stage-4: public `propagate` dispatch — PULL and Pathwise ensemble.
# `Magpie.propagate` (QUALIFIED) extends the core stub declared in src/gpude.jl.
# ---------------------------------------------------------------------------

using Random: MersenneTwister

# ---- ExactGP vector field -------------------------------------------------

"""
    propagate(gps, u0, tspan; method=PULL(), ts, buffer) -> (μs, Σs) | ensemble

Propagate uncertainty through a GP vector field (one `ExactGP` per output dimension).

- `method=PULL()` — analytic moment-matching via `pull_propagate`.
- `method=Pathwise(n=N)` — Monte-Carlo ensemble of `N` decoupled GP samples integrated as plain ODEs.

Returns `(μs, Σs)` for PULL, or an `N × d × length(ts)` array for Pathwise.
"""
function Magpie.propagate(gps::AbstractVector{<:ExactGP}, u0, tspan;
                          method=Magpie.PULL(),
                          ts=collect(range(tspan...; length=21)),
                          buffer=20)
    method isa Magpie.PULL && return pull_propagate(gps, u0, ts; buffer)
    return _pathwise(gps, u0, tspan, ts, method)
end

"""
    propagate(field::ExactGPField, u0, tspan; kw...) -> (μs, Σs) | ensemble

Reconstruct the posterior GPs from the trained field and dispatch to `propagate(gps, ...)`.
"""
Magpie.propagate(field::ExactGPField, u0, tspan; kw...) =
    Magpie.propagate(Magpie.posterior_gps(field), u0, tspan; kw...)

# Internal Pathwise integrator for ExactGP fields.
function _pathwise(gps, u0, tspan, ts, m::Magpie.Pathwise)
    d = length(u0); S = m.n
    out = zeros(S, d, length(ts))
    for sidx in 1:S
        samplers = [begin
            k = g.prior.kernel                                    # ScaledKernel (carries σ²)
            ℓ = Magpie._lengthscale(k)                            # peel ScaledKernel → inner TransformedKernel
            σ = sqrt(k(g.x[1], g.x[1]))                           # k(x,x) = σ² for stationary SE
            # Draw a consistent inducing-value sample from the posterior at the anchors.
            uvals = Magpie.mean(g, g.x) .+
                    Magpie._chol(Magpie.cov(g, g.x) + 1e-8 * I).L *
                    randn(MersenneTwister(sidx * 131 + i), length(g.x))
            Magpie.build_decoupled_sample(k, g.x, uvals;
                                          ℓ=ℓ, σ=σ,                       # decorrelate the RFF-phase RNG from the
                                          rng=MersenneTwister(sidx * 131 + i + 500_000))  # inducing-draw RNG above
        end for (i, g) in enumerate(gps)]
        rhs!(du, u, p, t) = (for i in 1:d; du[i] = samplers[i](u); end; nothing)
        sol = solve(ODEProblem(rhs!, collect(float.(u0)), tspan), Tsit5(); saveat=ts)
        out[sidx, :, :] = Array(sol)
    end
    return out
end

# ---- SparseGP vector field -------------------------------------------------

"""
    propagate(sgps, u0, tspan; method=PULL(), ts, buffer) -> (μs, Σs) | ensemble

Propagate uncertainty through a sparse GP vector field (one `SparseGP` per output dimension).
PULL uses `pull_propagate` (unchanged — `SparseGP` implements `predmean`/`var`/`cov`).
Pathwise draws from the whitened variational posterior and integrates as plain ODEs.
"""
function Magpie.propagate(sgps::AbstractVector{<:SparseGP}, u0, tspan;
                          method=Magpie.PULL(),
                          ts=collect(range(tspan...; length=21)),
                          buffer=20)
    method isa Magpie.PULL && return pull_propagate(sgps, u0, ts; buffer)
    return _pathwise_svgp(sgps, u0, tspan, ts, method)
end

"""
    propagate(field::SVGPField, u0, tspan; kw...) -> (μs, Σs) | ensemble

Reconstruct the sparse posterior GPs from the trained field and dispatch.
"""
Magpie.propagate(field::SVGPField, u0, tspan; kw...) =
    Magpie.propagate(Magpie.posterior_sparsegps(field), u0, tspan; kw...)

# Internal Pathwise integrator for SparseGP fields.
# Draws each sample by: (1) drawing whitened v_s ~ N(μ_i, S), (2) lifting to u_s = L_ZZ v_s,
# (3) building a decoupled sampler from inducing locations Z and values u_s.
function _pathwise_svgp(sgps, u0, tspan, ts, m::Magpie.Pathwise)
    d = length(u0); out = zeros(m.n, d, length(ts))
    for sidx in 1:m.n
        samplers = [begin
            k = g.prior.kernel
            ℓ = Magpie._lengthscale(k)
            σ = sqrt(k(g.Z[1], g.Z[1]))
            # Recover variational mean μ_i from the stored α = L_ZZ' \ μ_i.
            μ_i = g.L_ZZ' * g.α
            # Draw whitened v_s ~ N(μ_i, S) where S = L_S L_S'.
            v_s = μ_i .+ g.L_S * randn(MersenneTwister(sidx * 977 + i + 500_000), length(μ_i))
            # Lift to inducing-value sample u_s = L_ZZ v_s.
            u_s = g.L_ZZ * v_s
            Magpie.build_decoupled_sample(k, g.Z, u_s;
                                          ℓ=ℓ, σ=σ,
                                          rng=MersenneTwister(sidx * 977 + i))
        end for (i, g) in enumerate(sgps)]
        rhs!(du, u, p, t) = (for i in 1:d; du[i] = samplers[i](u); end; nothing)
        out[sidx, :, :] = Array(solve(ODEProblem(rhs!, collect(float.(u0)), tspan), Tsit5(); saveat=ts))
    end
    return out
end

# ---- CompositeField propagation ----------------------------------------------
# CompositeField = known_physics (fixed) + residual GP field.
# Pathwise: each ensemble sample integrates `du = known(u,t) + sampler_i(u)` — the full
# composite field with GP uncertainty. PULL for CompositeField would require propagating the
# combined Jacobian (∂known/∂u + ∂GP_mean/∂u), which is correct in principle but adds
# complexity without matching what the FHN example needs. Pathwise is the right tool for
# CompositeField coverage; PULL raises a clear error rather than silently producing wrong results.

"""
    propagate(cf::CompositeField, u0, tspan; method=Pathwise(n=128), ts, ...) -> ensemble

Propagate uncertainty through a CompositeField = `known_physics + residual_GP`.

Only `method=Pathwise(n=N)` is supported. Each ensemble sample integrates an ODE whose
RHS is `du = cf.known(u,t) + sampler_i(u)`, where `sampler_i` is a decoupled GP sample
drawn from the residual-GP posterior. Returns an `N × d × length(ts)` array.

`method=PULL()` raises an error — the PULL moment recurrence requires the combined
Jacobian of `known + GP_mean`, which is correct but not implemented. Use Pathwise, which
gives the full composite-field uncertainty without the linear-field assumption.
"""
function Magpie.propagate(cf::Magpie.CompositeField, u0, tspan;
                          method=Magpie.Pathwise(128),
                          ts=collect(range(tspan...; length=21)),
                          buffer=20)
    if method isa Magpie.PULL
        error("PULL is not implemented for CompositeField. Use Pathwise for composite-field " *
              "uncertainty propagation (it correctly integrates known_physics + GP sample).")
    end
    gps = Magpie.posterior(cf)   # residual ExactGPs
    known = cf.known
    _pathwise_composite(gps, known, u0, tspan, ts, method)
end

"""
    Magpie.propagate(cf::CompositeField, u0, tspan; method, ts, ...) -> ensemble

Variant that reconstructs from `v` directly (for one-shot use without first calling `posterior`).
"""
Magpie.propagate(cf::Magpie.CompositeField, u0, tspan, v; kw...) =
    Magpie.propagate(cf, u0, tspan; kw...)   # v0 is already stored in cf.gp.v0

# Internal Pathwise integrator for CompositeField.
# Each sample integrates `du = known(u,t) + sampler_i(u)` — full composite field.
# Mirrors `_pathwise` but adds the known_physics term to each RHS evaluation.
function _pathwise_composite(gps, known, u0, tspan, ts, m::Magpie.Pathwise)
    d = length(u0); S = m.n
    out = zeros(S, d, length(ts))
    for sidx in 1:S
        samplers = [begin
            g = gps[i]
            k = g.prior.kernel
            ℓ = Magpie._lengthscale(k)
            σ = sqrt(k(g.x[1], g.x[1]))
            uvals = Magpie.mean(g, g.x) .+
                    Magpie._chol(Magpie.cov(g, g.x) + 1e-8 * I).L *
                    randn(MersenneTwister(sidx * 131 + i), length(g.x))
            Magpie.build_decoupled_sample(k, g.x, uvals;
                                          ℓ=ℓ, σ=σ,
                                          rng=MersenneTwister(sidx * 131 + i + 500_000))
        end for i in 1:length(gps)]
        function rhs!(du, u, p, t)
            kphys = known(u, t)
            for i in 1:d
                du[i] = kphys[i] + samplers[i](u)
            end
            nothing
        end
        sol = solve(ODEProblem(rhs!, collect(float.(u0)), tspan), Tsit5(); saveat=ts)
        out[sidx, :, :] = Array(sol)
    end
    return out
end

end # module
