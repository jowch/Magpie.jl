# Active-Learning Foundation Hardening — Design Spec

**Date:** 2026-06-22
**Status:** Phase 1 detailed; Phases 2–3 scoped
**Author:** Jonathan Chen (with Claude)
**Supersedes/extends:** the critical-point survey work ([2026-06-21-critical-point-survey-design.md](2026-06-21-critical-point-survey-design.md)) — that branch's findings are the leaf-level instances of the foundation gaps addressed here.

## Motivation

A foundation audit of the active-learning capability (Capability A) surfaced 32 confirmed
findings — 20 ergonomics, 12 correctness — that collapse into a single theme: **the GP math
and the spine contract are sound, but the package is still a research script, not a library.**
The vision is a *professional-grade package that is actually useful*: correct and ergonomic
functionality for actually doing active learning, not a toy.

The critical-point work is the evidence and the motivation, but it is not the architecture
driver. Per the project thesis (`CLAUDE.md`): *the bridges/foundation are the product;
applications are examples, not architecture drivers.* This spec hardens the foundation; a
later phase reshapes the API and ships critical-points as a worked example on top.

## Positioning & layering (the architectural decision this spec encodes)

**Critical-points is the flagship *worked example*, not the headline capability.** The headline
is the foundation: a differentiable, uncertainty-aware GP + an active-learning loop +
composable, gradient-aware acquisitions. The ecosystem gap Magpie fills is the *acquisitions*
(Straddle, BALD-for-classification, gradient-straddle), not critical-point finding.

This fixes where each piece lives. **Phase 2** performs the move; this spec records the target
boundary so Phase 1 changes are made with it in mind:

| Piece | Altitude | Rationale |
|---|---|---|
| `grad_predict` (μ∇, gradient variance, mean-Hessian) | **Core, kernel-generic** | "Posterior gradient/Hessian of a GP" is a general capability. |
| `GradStraddle` / `RandGradStraddle` | **Core acquisition** | "Actively seek ∇μ=0" is a general acquisition, sits beside `Straddle`. |
| stationary-point polish (`newton_polish`, generalized) | **Core utility** | "Polish to a stationary point of the GP mean" is reusable. |
| `classify` (Morse index), `critical_points` survey | **Application module** (`Magpie.CriticalPoints`) | Morse theory is domain-specific; callable but namespaced. |
| `transition_state` / min-mode dimer walk | **Application module** | Most domain-specific (TS/saddle search). |

**Phase decomposition** (each phase: its own spec → plan → implementation):

- **Phase 1 — Correctness & table-stakes (this spec).** In-place correctness, reproducibility,
  and type-stability fixes across the whole Capability A surface. No public-API reshaping, no
  module restructuring. Shippable on its own; makes the foundation trustworthy.
- **Phase 1.5 — Multi-output spine (this spec).** Generalize the active-learning `ExactGP` from
  single-output (`δ,α` vectors) to `n×d` weights, converging on the gp-ude `ExactGPField`
  representation, so both capabilities share one multi-output-capable spine. Multi-output
  *acquisitions* are deferred (later phase); this phase makes the *model and contract*
  multi-output. Done early because the spine contract is what everything else builds on.
- **Phase 2 — Public API & ergonomics.** The core-vs-application layering above; promote
  `grad_predict`/`GradStraddle` to first-class core; ship a namespaced `CriticalPoints`
  application module (kills the copy-paste); first-class `Grid` maximizer; loop trace/history
  accessors; documented acquisition extension contract; getting-started docs + curated exports.
- **Phase 3 — Classification parity.** `predict_prob`, latent-vs-observation semantics,
  label-encoding tolerance, and a real (or honestly-documented) `fit` story for `LaplaceGP`.

---

## Phase 1 scope

Six workstreams. Each is a contained, correctness-or-table-stakes change that does **not**
move code between files or rename public entry points (those are Phase 2). Where a fix touches
code destined for the application layer (e.g. `classify`), the fix is made in place and will
travel with the code when Phase 2 moves it.

### 1.1 Input validation at every public boundary  *(HIGH, correctness)*

Today `update` / `observe!` / `Box` / `acquire` validate nothing; bad input surfaces as a
cryptic `LinearAlgebra`/broadcast error deep in the stack.

- Add one internal `_validate_obs(X, y)`: non-empty, `length(X) == length(y)`, all-finite,
  and (when prior points exist) consistent input dimension. One actionable message naming the
  offending count/index. Call it at the top of both `update` methods (`ExactGP`, `LaplaceGP`)
  and `observe!`.
- `Box`: add a validating inner constructor — `length(lb) == length(ub)` and `all(lb .≤ ub)`,
  else `ArgumentError` naming the offending dimension(s).
- Dimension guard: `acquire`/`transition_state` check the domain dimension against the GP/seed
  dimension before optimizing.
- High-D grid backstop: `grid_points` throws (or `@warn`s) when `per_axis^d` exceeds a cap
  (default `1e6`), suggesting `SobolPolish`/`Points`.

`_validate_obs` must accept **vector-valued** `y` (a multi-output observation is a length-`d`
vector), not assume a scalar — see Phase 1.5.

**Decision D2 (labels):** for `LaplaceGP`, accept `AbstractVector{<:Real}` and normalize —
`{0,1}`→`Bool`, `{-1,+1}`→`Bool` — erroring with a clear message on any other value set, rather
than relying on the `Bool` type wall. *Recommended: coerce the two standard encodings, error
otherwise.* (Alternative: stay `Bool`-only but give a helpful error. Recommendation is coerce,
because silent `Bool`-only rejection of `[0,1]` integer labels is a classic first-use stumble.)

### 1.2 Reproducibility — thread an RNG through the loop  *(MEDIUM, correctness)*

Stochasticity is currently ambient: `resample` draws from the global RNG, `_ts_seed` calls
unseeded `rand()`, the loop has no `rng` — yet randomized acquisitions already *hold* an `rng`.
One seed cannot reproduce a survey.

- `ActiveLearner` gains an `rng::AbstractRNG` field (default `Random.default_rng()`).
- Make `resample(a, rng)` the primary method; `run!`/`acquire` pass `al.rng` to it.
- Thread `rng` through `transition_state`/`_ts_seed` (keyword, default `Random.default_rng()`).
- **Single source of stochasticity:** one seeded `rng` on the learner reproduces the whole run.

### 1.3 Type-stable storage — mutable, parameterized on the data  *(MEDIUM, correctness — resolved)*

`ActiveLearner` and the unconditioned `ExactGP` store `Vector{Any}`. Replace with parameterized
concrete *data* storage.

**Mutable struct: confirmed.** The in-place `observe!`/`fit!`/`run!` loop reassigns the `gp` and
`acq` fields each round (`update`/`fit` return new immutable GPs; `acq` is rewrapped with
`LocalPenalization`), so a mutable struct is required. A functional/immutable design is possible
but fights the bang-API idiom for a fundamentally stateful loop; we keep mutable.

**Decision D1 — resolved: parameterize on the data only, `{TX,TY}`, both free.**

    mutable struct ActiveLearner{TX,TY}
        gp::AbstractGPModel        # abstract on purpose — see below
        acq::AcquisitionFunction   # abstract on purpose — see below
        Xs::Vector{TX}; Ys::Vector{TY}
        acq_vals::Vector{Float64}; rng::AbstractRNG
    end

**Refinement to the audit's suggestion (`{G,A,TX,TY}`):** the `gp` and `acq` fields must stay
*abstractly typed*, NOT parameterized. The concrete GP type changes during a run — the first
`update` takes `ExactGP{…,Nothing,…}` (unconditioned) to `ExactGP{…,Cholesky,…}` (conditioned) —
and `acq` changes type when wrapped by `LocalPenalization`. A concrete `gp::G` parameter would
make `al.gp = update(al.gp, …)` a type error on that first transition. So we parameterize only on
the *data* (`Xs`/`Ys`), which is the type-stability win we actually want; the abstract `gp`/`acq`
fields are recovered to concrete type at the function boundary (`acquire(al.gp, acq; …)` already
function-barriers the hot path), so there is no meaningful instability cost.

`TY` stays **free** to span the three real cases: `Float64` (regression), `Bool` (binary
classification), and `Vector{Float64}` (multi-output, see Phase 1.5). `TX` stays free for the
input representation (e.g. `Vector{Float64}` or scalar 1-D).

**Empty-state typing:** a mutable struct's parameters are fixed at construction, so `TX`/`TY`
must be known before the first `observe!`. Provide a convenience constructor with inferred
defaults (`TX = Vector{Float64}`; `TY` from the GP kind — `Float64`/`Vector{Float64}` for
`ExactGP` by its output dim `d`, `Bool` for `LaplaceGP`), plus a typed/seed-data escape hatch
(`ActiveLearner{TX,TY}(gp, acq)` or `ActiveLearner(gp, acq, X0, Y0)` inferring from seed data)
for full flexibility. `ExactGP` initial state likewise stores a typed empty `x` (not `Any[]`).

### 1.4 Convergence gating on the stationary-point walkers  *(HIGH, correctness)*

`newton_polish` / `saddle_walk` / `transition_state` clamp to the box and return whatever point
they land on — including boundary-clamped, non-converged iterates — with no signal, and
`classify` then assigns a Morse label to a non-stationary point.

**Decision D3 (return shape):** each walker returns its convergence status alongside the point.
*Recommended: a small named tuple* `(; x, μ∇, H, converged::Bool, residual::Float64)` (and the
same `converged`/`residual` surfaced in the `transition_state` result). `converged` is
`norm(μ∇) < tol` **and** the point is interior (not pinned to a box face with non-zero gradient).
Downstream (the Phase-2 `critical_points`) drops non-converged/boundary iterates. Optionally
`@warn` when the budget is exhausted without convergence.

*This is a breaking change to the walkers' return values* — acceptable now (pre-1.0, the only
consumers are in-repo) and far cheaper than after Phase 2 promotes them.

### 1.5 Scale-invariant `classify` threshold  *(MEDIUM, correctness)*

`classify` uses an absolute eigenvalue floor `ε=1e-3`; the mean-Hessian scales as `σ_f²/ℓ²`, so
valid extrema on low- or high-magnitude landscapes get mislabeled `:unclassified`. Today the
tests only pass because they pre-normalize the surface.

- Make the threshold relative: `abs(λ_i) < ε * maximum(abs, λ)` with a small absolute floor to
  handle the all-flat case. Document that `classify` is then scale-invariant (users no longer
  z-score solely to satisfy it). Default `ε` becomes a *relative* tolerance (e.g. `1e-3`).

### 1.6 Kernel-generic `fit`  *(MEDIUM, correctness)*

`fit` hard-asserts `SqExponentialKernel` and reconstructs the kernel as RBF, while
`grad_predict`/`_prior_grad_var_const` advertise and test Matérn-3/2 and Matérn-5/2 — so the
moment the loop refits a Matérn GP (the common `refit_every>0` path) it throws.

- Make `fit` reconstruct from `_basekernel(g.prior.kernel)` over the supported families
  (`SqExponentialKernel`, `Matern32Kernel`, `Matern52Kernel` — the set `grad_predict` supports),
  reusing the existing `_basekernel`/`_outputscale`/`_lengthscale` peelers.
- Convert the silent late `@assert` into a `fit`-entry check that names the supported kernels
  and points to the derivative path, for families outside the set.

**Decision D4 (supported set):** SqExp + Matérn-3/2 + Matérn-5/2 (matches `grad_predict`). Other
families error clearly. *Recommended* — keeps `fit` and the derivative GP in lockstep.

---

---

## Phase 1.5 — Multi-output spine

Generalize the active-learning `ExactGP` from single-output to `d` independent outputs sharing
one kernel, converging on the gp-ude `ExactGPField` representation so both capabilities share one
multi-output-capable spine. This is a *modeling* change (not pure correctness), kept distinct
from Phase 1 but done in the same foundation pass because the spine contract underlies everything.

### Representation (aligned with gp-ude `ExactGPField`)

- The prior carries an **output dimension `d`** (the `ExactGP` gains `d::Int`; `d=1` is the
  current scalar case and stays the zero-ceremony default).
- The cached residuals/weights widen from vectors to **`n×d` matrices**: `δ = Y - m(X)` is `n×d`,
  `α = C \ δ` is `n×d`. **`C` (the `n×n` shared-kernel Cholesky) is unchanged** — the single most
  important consequence: incremental `update_chol`, the `_chol` chokepoint, and all AD properties
  carry over untouched; only the right-hand side gains columns.
- `predmean(g, u)` returns a **length-`d` vector** (`vec(k(u,X) * α)`), matching gp-ude's
  `gpfield` callable. `mean(g, xs)` returns an `(length(xs))×d` (or vector-of-vectors) result.
- **Predictive variance is shared across outputs.** Because the outputs are independent with a
  shared kernel, `Var[f_i(x)] = k(x,x) − k(x,X) C⁻¹ k(X,x)` is identical for every output `i` —
  so `var`/`mean_and_var` compute the variance **once** (a scalar per point), not per output. The
  per-output structure lives entirely in the mean. This both simplifies the spine and tells the
  eventual multi-output acquisitions exactly what varies (mean per output, shared uncertainty).

### What changes, what doesn't

- **Changes:** `ExactGP` struct (`d` field, matrix `δ`/`α`), `update`/`_update_incremental`
  (matrix RHS), `mean`/`predmean`/`mean_and_var` (vector/matrix returns), `_validate_obs`
  (accept length-`d` `y`), the `ActiveLearner` `TY = Vector{Float64}` path.
- **Unchanged:** `C`/`_chol`/`update_chol` (shared kernel), `var`/`cov` *quadratic-form* math
  (variance is shared; only its broadcast over outputs is new), the AD story.

### Deliberately deferred to a later phase

- **Multi-output acquisitions.** `Straddle`/`GradStraddle` are scalar-field acquisitions; a
  multi-output level set or gradient-zero needs a defined reduction (which output? a scalarization
  over the per-output means against the shared variance?). Out of scope here — Phase 1.5 makes the
  *model* multi-output and leaves a single-output (or `d=1`) acquisition path as the supported one,
  with a clear error if a multi-output GP is handed to a scalar-only acquisition.
- **Convergence with gp-ude's `SparseGP <: AbstractGPModel`.** gp-ude already exposes a
  `SparseGP` spine model; full unification of the two GP families is a cross-capability project,
  not this pass. Phase 1.5 only ensures the `ExactGP` representation *matches* gp-ude's so that
  convergence is later a merge, not a rewrite.

## Out of scope for Phase 1 (explicitly deferred)

- Promoting `grad_predict`/`GradStraddle` to core; the `CriticalPoints` application module;
  killing the `critical_points` copy-paste → **Phase 2** (needs the module move).
- First-class `Grid` maximizer, `run!` threading the maximizer, history/trace accessors,
  acquisition extension-contract docs, getting-started docs, curated exports → **Phase 2**.
- `predict_prob`, latent-semantics docs, `LaplaceGP` `fit` story → **Phase 3**.

(Note: the superseded "active beats random" exemplar is *in* Phase 1 — workstream 1.7 below —
not deferred; it is a cheap correctness/honesty fix.)

### 1.7 Reconcile the superseded exemplar with the docs  *(MEDIUM, correctness/honesty)*

`test/test_exemplar_critpoints.jl` still asserts an "active learning beats random for
enumeration" advantage that the critical-point spec's own as-built section explicitly retracts
("extraction-limited, not acquisition-limited"). Delete or rewrite this exemplar to assert only
what is robust and non-contradictory — that the survey recovers the known critical points from a
coverage sample, and that the *targeted* `transition_state` (the genuine active-learning win)
converges — matching the documented honest ordering. Keep the harmless "finds the single
minimum of a quadratic bowl" testset.

## Testing strategy

Each workstream lands test-first (TDD):
- **Validation:** each public entry point rejects mismatched-length, empty, NaN/Inf, reversed-box,
  dimension-mismatch, and bad-label inputs with the intended `ArgumentError`/message.
- **Reproducibility:** two `run!`s with the same seeded `rng` produce identical query sequences;
  `transition_state` is deterministic under a fixed `rng`.
- **Type stability:** `@inferred` (or `Test.@inferred`/`JET`) on `observe!`/`predict`/the hot
  acquisition path; assert `eltype(queried_points(al)) != Any`.
- **Convergence gating:** a walker on a ramp (no interior critical point) returns `converged=false`
  and a boundary point; `classify` is not called on / not trusted for non-converged points.
- **Scale invariance:** `classify` returns the correct Morse type on a surface scaled by `1e-3`
  and by `1e3` (the previously-failing cases), without pre-normalization.
- **Kernel-generic fit:** `fit` of a Matérn-5/2 `ExactGP` succeeds and improves `nlml`; an
  unsupported kernel errors with the supported-set message. Add the missing Mooncake-vs-finite-
  difference check on the two-parameter penalized loss (`[logℓ, logσ²]` + MAP penalty).
- **Multi-output spine (Phase 1.5):** a `d=2` `ExactGP` conditioned on vector observations returns
  a length-2 `predmean`; its `α` is `n×2`; its predictive `var` is identical across the two outputs
  (shared-kernel invariant) and equals the `d=1` result; incremental `update` matches a
  from-scratch refit; the representation matches gp-ude `ExactGPField` (`d`, `n×d` `α`, `_chol`).
  `d=1` behavior is byte-for-byte unchanged from the scalar path.

The full suite must stay green; the previously-pre-normalized `classify`/saddle tests are
updated to exercise the scale-invariant and convergence-gated behavior directly.

## Risks & mitigations

- **Breaking return-shape change (1.4):** only in-repo consumers today; do it now, before Phase 2
  promotion widens the blast radius.
- **Typed `ActiveLearner` empty-state (1.3):** the recommended typed-default + escape-hatch keeps
  the common path ceremony-free; the alternative (rebuild on first observe) is rejected as magic.
- **Label coercion (1.2/D2):** coercion is convenience, not silent correctness loss — the error
  path on unknown encodings is explicit.

## Decisions (resolved)

- **D1** — storage: **mutable struct, parameterize on data `{TX,TY}` only** (`gp`/`acq` stay
  abstract fields for in-place reassignment); `TY` free to span scalar / `Bool` / `Vector{Float64}`;
  empty-state via inferred-default + typed/seed-data escape-hatch constructors. *Resolved.*
- **D2** — `LaplaceGP` labels: coerce `{0,1}`/`{-1,+1}`→`Bool`, error otherwise. *Resolved.*
- **D3** — walker return shape: named tuple with `converged`/`residual`. *Resolved.*
- **D4** — `fit` supported kernels: SqExp + Matérn-3/2 + Matérn-5/2. *Resolved.*
- **Multi-output** — generalize the `ExactGP` spine to `n×d` weights now (Phase 1.5), aligned to
  gp-ude `ExactGPField`; multi-output *acquisitions* deferred. *Resolved.*
