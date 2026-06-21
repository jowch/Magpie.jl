# Critical-Point Survey via Derivative-GP + Vector-Zero Acquisitions

**Date:** 2026-06-21
**Status:** Design approved, pre-implementation
**Type:** Exploratory example / Capability-A extension (ahead of the post-Capability-B example pass)

## Goal

Demonstrate finding and **classifying all critical points** of a function `f: ℝ^d → ℝ`
(minima, maxima, saddles) by treating the gradient `∇f` as a Gaussian-process-derived
vector field and using an active-learning acquisition to localize its zeros
`{x : ∇f(x) = 0}`. Each found point is labeled by **Morse index** (number of negative
Hessian eigenvalues): `0 → min`, `d → max`, otherwise `saddle`.

This is the topology-of-the-critical-set capability that plain Bayesian optimization
does not provide.

## Prior-art grounding (honesty about what is/isn't novel)

The core mechanism is **not novel** — and that is fine for an example, it means we
follow a citable method rather than invent one:

- **Inatsu, Sugita, Toyoura, Takeuchi (2020), "Active Learning for Enumerating Local
  Minima Based on Gaussian Process Derivatives," *Neural Computation* 32(10).** Models
  `f` with a GP, derives the `∇f` process analytically (f-values observed, gradients
  *not* observed), builds confidence intervals on each `∇f` component and on the
  Hessian `λ_min`, and actively samples to enumerate `{∇f=0 ∧ λ_min>0}`. Explicitly
  notes saddles/maxima follow by changing the Hessian condition. 2–3D experiments.
  Same author as the `RandStraddle` (2024) already in this package — clean lineage.
- **Scalar level-set lineage:** Bryan & Schneider (2005, Straddle) → Gotovos/Krause
  (2013, LSE with confidence-bound classification) → Inatsu (2024, randomized
  straddle). This package already implements `Straddle`, `RandStraddle`.
- **Chemistry (Jónsson/Koistinen 2017–2026, GP-NEB / GP-dimer):** mature derivative-GP
  saddle search, but finds *one* saddle per run via path topology; never enumerates or
  classifies.

**What this example does that the literature has not combined:** classify **all** Morse
indices in a single sweep (Inatsu enumerates one type at a time), and — Phase 2 — fold
in gradient observations. A recombination near the frontier, not a toy.

### Same as Inatsu / different / extension

- **Same:** f-only observation + analytic `∇f` process; per-component treatment of the
  gradient; Hessian-eigenvalue classification test; active loop to enumerate; low-D.
- **Different (a deliberate, disclosed substitution):** Inatsu's acquisition is a
  confidence-interval *classifier* (Gotovos lineage) with PAC-style guarantees; we
  *also* offer a Bryan-style component-Straddle heuristic, and compare the two. We use
  the posterior-**mean** Hessian for classification (not Inatsu's `λ_min` CI) in the
  first pass.
- **Extension:** full Morse-index labeling in one sweep.

## Why component-wise (not a norm)

`E‖∇f‖² = ‖μ_∇‖² + tr(Σ_∇)` is a fine post-hoc *confidence score* but a poor
*acquisition*: both terms are positive, so minimizing it flees uncertainty and
maximizing it chases large gradients — neither gives explore/exploit tension. And
straddling the scalar field `g=‖∇f‖²` at level 0 is degenerate: `g≥0` so `0` is its
minimum, touched tangentially, never crossed transversally.

Component-wise works because near a nondegenerate critical point `x*`,
`∇f(x) ≈ H(x−x*)`, so **each** `∂f/∂xᵢ` crosses zero *transversally* through a
`(d−1)`-surface; the critical point is the intersection of those `d` surfaces. This
decomposes the hard codim-`d` isolated-zero problem into `d` well-posed codim-1
level-set problems the existing `Straddle` machinery already solves.

## Architecture

Module is `Magpie`. `ExactGP` fields (`prior, x, δ, C, α, noise`) are package-internal
and used directly. Prior kernel is `with_lengthscale(SqExponentialKernel(), ℓ)` (RBF,
unit variance), `ℓ = _lengthscale(g.prior.kernel)`.

### 1. Derivative-predict helper (`src/`, new) — the only new numeric piece

Given an `ExactGP` (RBF) and query `x ∈ ℝ^d`, return:

- `μ_∇(x) ∈ ℝ^d` — posterior-mean gradient = `(∂ₓ k(x, X)) · α`
- `diag Σ_∇(x) ∈ ℝ^d` — marginal posterior variance of each `∂f/∂xᵢ`
  = `∂²ᵢᵢ k(x,x) − (∂ᵢ k(x,X)) C⁻¹ (∂ᵢ k(x,X))ᵀ` (reuses cached `C`)
- `H̄(x) ∈ ℝ^{d×d}` — posterior-mean Hessian = `(∂²ₓ k(x, X)) · α`

RBF derivative blocks (unit variance, lengthscale `ℓ`), with `r = x − x'`:
- `∂k/∂xᵢ = −(rᵢ/ℓ²) k`
- `∂²k/∂xᵢ∂xⱼ = k · [ rᵢrⱼ/ℓ⁴ − δᵢⱼ/ℓ² ]` (this is the Hessian block w.r.t. the
  *query*; for `Σ_∇` we need the `∂²/∂xᵢ∂x'ⱼ` cross-block
  `k·[δᵢⱼ/ℓ² − rᵢrⱼ/ℓ⁴]`, evaluated at `x=x'` giving `δᵢⱼ/ℓ²` for the prior term).

**ponytail:** RBF only; hardcode the derivative algebra. Generalize to other kernels
only when a second kernel actually needs it.

### 2. Two gradient acquisitions (`src/acquisitions.jl`, new) over the helper

Both are `AcquisitionFunction` callables `(g, x)`; differ only in scoring:

- `GradStraddle(; β)` — component Straddle:
  `score(x) = minᵢ [ β·√(Σ_∇)ᵢᵢ − |μ_∇,ᵢ(x)| ]`. The `min` enforces the AND
  (high only where *every* component is near-zero AND uncertain). Reuses the package's
  existing Straddle idea on the gradient components.
- `GradLSE(; β)` — Inatsu CI-classifier, **fidelity (b) "pragmatic"**: classify a point
  as a critical-candidate iff `0 ∈ [μ_∇,ᵢ ± β√(Σ_∇)ᵢᵢ]` for all `i`; acquire the most
  *ambiguous* unclassified point (largest summed CI half-width among candidates near the
  boundary). CIs on gradient components only (exact-Gaussian, cheap). **Deferred (a):**
  add a CI on `λ_min` (needs the Hessian posterior *distribution*, sampled) for full
  Inatsu fidelity.

### 3. Active loop (reused)

`ActiveLearner`: seed → `acquire` (via the chosen gradient acquisition over a `Box`)
→ `observe!` f → `fit!` → repeat to budget. No loop changes.

### 4. Extract + classify (exemplar-side helper)

After the budget: on a fine grid, keep points where all `d` component CIs contain 0
(candidates), cluster, polish each with a few Newton steps on the GP-mean gradient
(`x ← x − H̄⁻¹ μ_∇`), then classify by Morse index from `eigvals(H̄)`. Report
`E‖∇f‖² = ‖μ_∇‖² + tr Σ_∇` as the per-point confidence score.

### 5. Validation = the one runnable check (`test/`, exemplar)

**Himmelblau** `f=(x²+y−11)²+(x+y²−7)²` on `[−5,5]²`: **9 critical points with known
analytic locations** — 4 minima (value 0), 1 maximum `(−0.2708,−0.9230)`, 4 saddles.
Observe f only, run the loop, assert all 9 recovered near analytic locations with
correct Morse index. Run the same assertion for **both** `GradStraddle` and `GradLSE`
(the Phase-1 A/B comparison).

## Comparison study (two axes, phased one at a time)

- **Phase 1 (f-only):** acquisition flavor — `GradStraddle` vs `GradLSE`. Same data, GP,
  extractor; only the acquisition swaps.
- **Phase 2:** observation mode — f-only vs f+∇f. Requires derivative *observations* in
  the Gram matrix (a `(1+d)n × (1+d)n` augmented system, noticeably ill-conditioned —
  needs a gradient-likelihood noise floor; GPyTorch ships a warning on its
  `RBFKernelGrad`). Bigger change; **deferred** until Phase 1 lands.

## Placement

- `src/`: derivative-predict helper + `GradStraddle` + `GradLSE` (real reusable
  Capability-A additions — the existing acquisition set is not closed).
- `test/`: Himmelblau exemplar with the A/B comparison and Morse-index assertions.
- Package `src/` for Phase 2 (derivative-observation GP) untouched until Phase 1 is done.

## Open items to verify during implementation

- Whether the RBF `∂k` algebra composes cleanly with how `g.x` stores inputs (vector of
  points) and `AbstractGPs.cov(g.prior, xs, g.x)`; may need to map over training points.
- Conditioning of `Σ_∇` near data-dense regions (a noise floor / `check=false` Cholesky
  is already house style via `_chol`).
- Newton polish robustness when `H̄` is near-singular (clamp / fall back to gradient
  descent on `‖μ_∇‖`).

## Deferred (explicitly not in this pass)

- `GradLSE` fidelity (a): `λ_min` confidence interval (sampled Hessian posterior).
- Phase 2: f+∇f derivative observations.
- Kernels other than RBF.
- Promoting the extract/classify helper from the exemplar into `src/` (do it only if a
  second consumer appears).
