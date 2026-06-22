# Critical-Point Survey via Derivative-GP + Component Straddle

**Date:** 2026-06-21
**Status:** Implemented (`worktree-critical-points`). **The "Post-exploration revision" section
below is superseded — read "As-built (final)" at the end for what was actually built and found;
the cosine "success" it describes turned out to be artifactual.**
**Type:** Exploratory example / Capability-A extension (ahead of the post-Capability-B example pass)

## Goal

Find and **classify all critical points** of a function `f: ℝ^d → ℝ` (minima, maxima,
saddles) by treating the gradient `∇f` as a Gaussian-process-derived vector field and
using an active-learning acquisition to localize its zeros `{x : ∇f(x) = 0}`. Each found
point is labeled by **Morse index** (number of negative Hessian eigenvalues):
`0 → min`, `d → max`, otherwise `saddle`.

Single acquisition, single deliverable: locate the critical points and classify them.
No acquisition comparison (see "Scope" below).

## Prior-art grounding

The core mechanism is **not novel** — and that is fine for an example; it means we
follow a citable method rather than invent one.

- **Inatsu, Sugita, Toyoura, Takeuchi (2020), "Active Learning for Enumerating Local
  Minima Based on Gaussian Process Derivatives," *Neural Computation* 32(10).** Models
  `f` with a GP, derives the `∇f` process analytically (f-values observed, gradients
  *not* observed), and actively samples using confidence intervals on the `∇f`
  components and on the Hessian `λ_min` to enumerate `{∇f=0 ∧ λ_min>0}`. §4.2 explicitly
  states the method extends to saddles/maxima by changing the Hessian condition (no
  experiment shown). Same author as the `RandStraddle` (2024) already in this package.
- **Scalar level-set lineage:** Bryan & Schneider (2005, Straddle) → Gotovos/Krause
  (2013, LSE confidence-bound classifier) → Inatsu (2024, randomized straddle). This
  package already implements `Straddle`, `RandStraddle`.
- **Chemistry (Jónsson/Koistinen 2017–2026, GP-NEB / GP-dimer):** mature derivative-GP
  saddle search, but finds *one* saddle per run via path topology; never enumerates or
  classifies.

**How this example relates.** We *demonstrate* what Inatsu's framework covers but did
not run — finding **all** critical-point types in one sweep and classifying by Morse
index — using the simpler **Bryan-Straddle heuristic** on the gradient components
(`βσᵢ − |μᵢ|`) rather than Inatsu's CI-classifier, and classifying post-hoc from the
posterior-mean Hessian. Honest framing: a faithful-in-spirit demonstration with a
heuristic acquisition, not a reproduction of Inatsu's CI machinery or its guarantees.

### Same as Inatsu / different / not attempted

- **Same:** f-only observation + analytic `∇f` process; per-component treatment of the
  gradient; Hessian-eigenvalue classification; active loop to enumerate; low-D.
- **Different:** Bryan-Straddle heuristic acquisition (not the Gotovos/Inatsu
  CI-classifier — note the per-component LSE ambiguity at threshold 0 *equals* the
  straddle score, so the genuine distinction is the `λ_min` CI, which we do not build);
  posterior-**mean** Hessian for classification (not a `λ_min` confidence interval);
  classify all Morse indices post-hoc rather than targeting one type.
- **Not attempted (deferred):** the `λ_min` confidence interval, the CI-classifier
  acquisition, any acquisition A/B, and gradient observations (Phase 2).

## Why component-wise (not a norm)

`E‖∇f‖² = ‖μ_∇‖² + tr(Σ_∇)` is a poor *acquisition*: both terms are positive, so
minimizing it flees uncertainty and maximizing it chases large gradients — neither gives
explore/exploit tension. Straddling the scalar field `g=‖∇f‖²` at level 0 is degenerate:
`g≥0`, so `0` is its minimum, touched tangentially (`∇g=2H∇f→0` at `x*`), never crossed
transversally.

Component-wise works because near a nondegenerate critical point `x*`,
`∇f(x) ≈ H(x−x*)`, so **each** `∂f/∂xᵢ` is affine and crosses zero *transversally*
through a `(d−1)`-surface; the `d` independent surfaces meet in the single point `x*`.
This decomposes the hard codim-`d` isolated-zero problem into `d` well-posed codim-1
level-set problems the existing `Straddle` machinery already solves. (FD- and
sweep-verified by review.)

## Architecture

Module is `Magpie`. `ExactGP` fields (`prior, x, δ, C, α, noise`) are package-internal
and used directly. Prior kernel is `with_lengthscale(SqExponentialKernel(), ℓ)` →
`k(x,x')=exp(−‖x−x'‖²/(2ℓ²))`, unit variance, `ℓ = _lengthscale(g.prior.kernel)`.

### 1. Derivative-predict helper (`src/`, new) — the only new numeric piece

Given an `ExactGP` (RBF) and query `x ∈ ℝ^d`, return:

- `μ_∇(x) ∈ ℝ^d` — posterior-mean gradient = `(∂ₓ k(x, X)) · α`
- `diag Σ_∇(x) ∈ ℝ^d` — marginal posterior variance of each `∂f/∂xᵢ`
  = `1/ℓ² − diag( Ksᵀ C⁻¹ Ks )`, computed via the existing `diag_Xt_invA_X(C, Ks)`
- `H̄(x) ∈ ℝ^{d×d}` — posterior-mean Hessian = `(∂²ₓ k(x, X)) · α`

RBF derivative blocks (unit variance, lengthscale `ℓ`), with `r = x − x'` (review-verified):
- `∂k/∂xᵢ = −(rᵢ/ℓ²) k`
- query-Hessian `∂²k/∂xᵢ∂xⱼ = k · [ rᵢrⱼ/ℓ⁴ − δᵢⱼ/ℓ² ]`
- mixed `∂²k/∂xᵢ∂x'ⱼ = k · [ δᵢⱼ/ℓ² − rᵢrⱼ/ℓ⁴ ]` → `δᵢⱼ/ℓ²` at `x=x'` (prior grad-var `1/ℓ²`)

**Shape note (review):** for `diag_Xt_invA_X(C, Ks)` the cross-cov `Ks` must be `n×d`
with column `i` = the length-`n` vector `∂ᵢk(x,X)`. The `∂k` block falls out `d×n`, so
`permutedims` it. This is the single-query form → loop over query points.

**ponytail:** RBF only; hardcode the derivative algebra. Generalize only when a second
kernel actually needs it.

### 2. `GradStraddle` acquisition (`src/acquisitions.jl`, new)

An `AcquisitionFunction` callable `(g, x)` (parallels `Straddle`'s `(g,x)` form):

```
score(x) = ∑ᵢ [ β·√(Σ_∇)ᵢᵢ − |μ_∇,ᵢ(x)| ]
```

**Sum, not min** (review): `min` is dominated by the most-resolved component and starves
half-resolved critical points (one component pinned to 0, another still uncertain → the
`min` abandons the point). The sum keeps explore/exploit tension on every component, and
each term still penalizes distance-from-zero per component (large `|μᵢ|` → negative
contribution). Plugs into the existing `acquire(g, a; over=Box)` and `ActiveLearner` with
no loop changes.

### 3. Active loop (reused)

`ActiveLearner`: seed → `acquire` (via `GradStraddle` over a `Box`) → `observe!` f →
`fit!` → repeat to budget via `run!(al, f; budget, over)`. No loop changes.

### 4. Extract + classify (exemplar-side helper)

After the budget:
1. On a fine grid, keep points where all `d` component CIs contain 0
   (`|μ_∇,ᵢ| ≤ β√(Σ_∇)ᵢᵢ` ∀i) — the candidate set.
2. Polish each candidate with a few Newton steps on the GP-mean gradient: solve the
   linear system `H̄ \ μ_∇` (do **not** form `H̄⁻¹`); clamp / fall back to gradient
   descent on `∑|μ_∇,ᵢ|` if `H̄` is near-singular.
3. Deduplicate polished points: `unique(round.(x; digits=2))` within tolerance ε. (No
   clustering — the 9 Himmelblau points are >1 apart; clustering would do nothing.)
4. Classify each by Morse index from `eigvals(Symmetric(H̄))` (guaranteed real). Label
   only when `|λ| > ε_morse` (e.g. `1e-3`) for all eigenvalues; otherwise report
   "unclassified" rather than risk a wrong label on a borderline Hessian.

### 5. Validation = the one runnable check (`test/`, exemplar)

**Himmelblau** `f=(x²+y−11)²+(x+y²−7)²` on `[−5,5]²`: **9 critical points, known analytic
locations** — 4 minima (f=0; ≈(3,2),(−2.805,3.131),(−3.779,−3.283),(3.584,−1.848)), 1
maximum at (−0.2708,−0.9230) (f≈181.6), 4 saddles. Observe f only, run the loop with
`GradStraddle`, assert all 9 recovered near analytic locations with correct Morse index.
(Ground truth confirmed by independent Newton sweep during review.)

## Scope

- **In:** the derivative-predict helper, `GradStraddle`, the find+classify pipeline, the
  Himmelblau exemplar.
- **Out (deferred, referenced as prior art only):** the `λ_min` confidence interval and
  a CI-classifier (`GradLSE`) acquisition; any acquisition A/B comparison; gradient
  observations (`f+∇f`, a `(1+d)n` augmented ill-conditioned Gram matrix — Phase 2);
  kernels other than RBF; promoting the extract/classify helper into `src/` (do that only
  if a second consumer appears).

## Placement

- `src/`: the derivative-predict helper + `GradStraddle` (a real reusable Capability-A
  addition; the helper lives wherever the acquisition lives — `src/acquisitions.jl` or a
  small companion file).
- `test/`: the Himmelblau exemplar with the find+classify assertion.

## Open items to verify during implementation

- `Ks` shape (`n×d`, component-indexed columns) and the per-query loop for `Σ_∇`.
- Conditioning of `Σ_∇` in data-dense regions (`_chol`'s `check=false` already guards
  tiny-negative pivots; add a noise floor if needed).
- Newton-polish robustness near singular `H̄` (linear solve + clamp/GD fallback;
  `ε_morse` classification threshold to avoid mislabeling).
- Whether the active loop places enough data near saddles/maximum (not just minima) for
  the mean Hessian to classify them correctly — if not, increase budget or seed coverage.

## Post-exploration revision (2026-06-21) — SUPERSEDED

> **Superseded by "As-built (final)" below.** This section's central claim — that the
> `cos(x)+cos(y)` active loop is "the clean active-learning success" — was later shown to be an
> **artifact** of a brittle extraction filter that handicapped the random baseline. With robust
> extraction, uniform random ties or beats the active loop on these small-domain problems. Kept
> for history; do not treat its conclusions or deliverables as current.

Implementing Task 4 (Himmelblau exemplar) surfaced that the original target was the wrong
shape, and a controlled diagnosis re-scoped the example. Recorded honestly:

**Himmelblau does not work as a *single-assertion* active exemplar — and the bottleneck is
the acquisition, not the kernel/extraction.** Layer-by-layer, controlled (active vs.
uniform-random, budget-swept, coverage-measured):
- **Kernel ✓** — RBF on a dense grid recovers all 9 (incl. the sharp minima). The earlier
  "stationary RBF can't represent the minima" guess was *wrong*; it was the extraction.
- **Extraction** — the CI candidate-filter (`|μ∇| ≤ β√Σ`) is **brittle under near-
  interpolation** (`Σ_∇→0` shrinks the band to ~0, rejecting grid points *near* a sharp
  minimum). Newton-polish-from-grid (no CI gate) is more robust. The shipped `critical_points`
  keeps the CI-filter because it is clean on the smooth cosine; Newton-from-grid is the
  documented upgrade for harder landscapes.
- **Budget ✗ not it** — uniform random recovers 9/9 at ~230 points.
- **Acquisition = the frontier.** Deterministic `GradStraddle` **mode-collapses** on
  Himmelblau (the constant `−|μ∇|` term pins the argmax to one region on steep walls):
  coverage **1/9** even at 630 samples. Two coverage-forcing cures: **randomization**
  (`RandGradStraddle`, ~6/9 mean) and **Inatsu CI-eviction** (7/9, self-terminates early →
  lower recall). Neither matches uniform random's 9/9 — the residual exploit bias is open work.

**Revised deliverable (implemented):**
- **Primary exemplar:** `cos(x)+cos(y)` on `[-4,4]²` — smooth, single-scale, analytic 9
  critical points (4 min/1 max/4 saddle). Active loop with `GradStraddle` + `refit_every`
  recovers **9/9**, zero spurious. This is the clean active-learning success.
- **`RandGradStraddle`** (`src/acquisitions.jl`): randomized-band gradient straddle mirroring
  `RandStraddle`; the buildable anti-mode-collapse demonstration.
- **Himmelblau stress-case** (exemplar): asserts the coverage ordering (det ≤2 < rand <
  random ≥8), documenting the acquisition frontier honestly rather than hiding it.

**Other learnings (documented, not built):**
- **Observation standardization** (z-score, or `log1p` for non-negative wide-range `f`) is a
  precondition the original spec omitted — essential for large-dynamic-range targets; it is
  location/Morse-type-preserving (affine/monotone), so `critical_points` is unchanged.
- **KernelFunctions (v0.10.67) has no derivative-kernel type**, but its kernels are
  AD-differentiable in their inputs — an **AD-generic `grad_predict`** (ForwardDiff on the
  kernel) would support any kernel (e.g. Matérn-5/2) without hand-derived blocks, *except*
  the `r=0` prior-variance term (Matérn `‖·‖` singularity → NaN; needs the analytic constant
  `5/3ℓ²`). A future extension; RBF stays the v1 path.
- Matérn was a red herring for Himmelblau's minima (the kernel was never the problem).

## As-built (final)

What was actually built (`worktree-critical-points`, suite 77/77) and — more importantly —
what the investigation found. The headline claim of the superseded revision did not survive
scrutiny; the work below is the honest replacement.

### Engine changes (reusable, independent of this example)

1. **`fit` optimizes σ_f² as well as ℓ.** Previously `fit` tuned only the lengthscale, with
   signal variance frozen at 1 (`src/fit.jl`). `grad_predict` then hardcoded prior gradient
   variance as `1/ℓ²`, so σ_f² did **not** cancel in the `GradStraddle` explore/exploit ratio
   (unlike ordinary `Straddle` on f-values, where it does) — miscalibrating the acquisition on
   any non-unit-scale function. Now `fit` recovers `σ_f²·withlengthscale(SqExp, ℓ)`; the manual
   `log1p`/z-score standardization the old revision called a "precondition" is no longer needed
   for scaled targets. (Measured: 50×-scaled cosine recovers without standardization.)

2. **`grad_predict` is kernel-generic via AD.** Replaced the hand-written RBF derivative blocks
   with `ForwardDiff` on `predmean` (gives μ∇ and the mean-Hessian for *any* kernel — this is
   also the answer to "what if there's no closed-form Hessian") and AD on the kernel for the
   gradient-variance cross term. The only kernel-specific piece is the prior gradient-variance
   constant where AD is singular at coincidence (Matérn's `‖·‖`): AD-by-default, with an analytic
   override (`5/3ℓ²·σ²` for Matérn-5/2). FD-verified on SqExp and Matérn-5/2.

3. **`LocalPenalization` acquisition (`src/acquisitions.jl`).** A composable soft-penalty wrapper
   adding `Σⱼ log Φ((‖x−xⱼ‖ − c·ℓ)/(s·ℓ))` to any base acquisition over the observed history.
   Adapts Local Penalization (González et al. 2016): the soft probit envelope transfers, but the
   radius is the **fitted lengthscale `c·ℓ`** (a redundancy scale), not LP's optimum-seeking
   `(M−μ)/L` — that formula assumes a scalar being maximized and does not transfer to gradient-zero
   search (verified against the paper). `ActiveLearner.acq` is now abstractly typed so the loop's
   live history can be wrapped: `al.acq = LocalPenalization(al.acq, al.Xs)`.

4. **Robust extraction.** The exemplar's `critical_points` now does multi-start Newton-from-grid
   (no CI candidate gate). The old `|μ∇| ≤ β√Σ∇` filter is brittle once the gradient is
   well-resolved: `Σ∇ → 0` shrinks the band and rejects grid points merely near a zero.

### The exemplar (honest)

`f(x) = (x₁²−1)² + (x₂²−1)²` — 9 critical points (4 minima, 1 max, 4 saddles) all inside
`[-1,1]²`, searched over the **large** box `[-6,6]²` (the centre is ~1/36 of the area). This is
the regime where active learning genuinely beats random:

- **uniform random** spreads across the whole box; only a handful of points reach the informative
  centre → med **2/9** recovered.
- **plain GradStraddle** mode-collapses (resamples one spot) → fails.
- **GradStraddle + LocalPenalization** focuses on the centre (the straddle is repelled from the
  steep outer walls toward the gradient zeros) *and* spreads within it (the penalty) → med **9/9**.
  Both pieces are required.

### Findings (measured; the honest core of this work)

- **σ_f²-calibration is correct but does not rescue Himmelblau.** A 6-seed sweep over
  {raw, log1p+z} × {frozen-σ², fitted-σ²} left Himmelblau coverage flat (~2.5–3.5/9). The
  bottleneck is the **acquisition's exploration on a multiscale landscape**, not calibration.
- **Mechanism of the collapse (diagnosed):** the deterministic `GradStraddle` argmax pins to a
  box corner — 99/150 queries in one cell — because the `−|μ∇|` exploit term makes ~62% of a
  steep domain repulsive, collapsing the argmax onto the flattest empty region; and stacking
  queries at one point never resolves its gradient (which needs spatial spread). `LocalPenalization`
  fixes exactly this.
- **Active learning does NOT beat random for low-D *exhaustive* enumeration on a small domain.**
  Once extraction is robust, uniform random ties cosine (9/9) and beats the active loop on
  Himmelblau at tight budgets. Exhaustive enumeration needs *global* coverage, which uniform
  random is near-optimal at in low-D. The active loop's advantage appears only with **localized
  targets / scarce budget relative to domain** (hence the large-domain exemplar above). The
  earlier "active beats random" was an extraction artifact.
- **Inatsu (2020) relationship:** the example uses a Bryan-straddle heuristic + posterior-mean
  Hessian, not Inatsu's CI-classifier acquisition + λ_min CI. Region-eviction (Inatsu's idea) and
  the λ_min-CI classifier remain the principled, un-built upgrades.

### Deferred / open

- λ_min confidence-interval classification (vs the posterior-mean Hessian point estimate).
- Inatsu-style targeted region-eviction (evict only gradient-resolved regions) vs the current
  blanket `LocalPenalization`.
- Higher-dimensional enumeration (grid extraction does not scale past ~3-D; needs sample-based
  multi-start). A separable d=3 test showed random still competitive when minima are lattice-spread.

---
## As-built (final, 2026-06-22) — supersedes the original survey design

The contrived critical-point *survey* (separable double-well; large-box "active-beats-random" framing) is superseded. Spikes + reconnaissance established:

- **Realistic targets, shipped as Literate examples:** the Müller–Brown potential (the standard transition-state benchmark) and the Maunga Whau volcano DEM (real terrain; all three Morse types).
- **Core capability (the product):** kernel-generic derivative-GP extraction — `grad_predict` → multi-start Newton → Morse `classify` — recovers and labels all critical points from coverage-sampled f-values (sub-0.05 on Müller–Brown). Gradient-variance pruning + interior masking drop spurious under-sampled zeros.
- **Honest negative result:** active learning does NOT beat random for scarce-budget critical-point *enumeration*. Verified across five acquisitions (including a novel standardized stationarity density N(0; μ∇, Σ∇)) and two topologies, cold/warm-start/synthetic. Mechanism: it is **extraction-limited**, not acquisition-limited — the Newton extractor needs spatial coverage to represent every basin, so any acquisition that concentrates the budget starves the unvisited basins.
- **The genuine active-learning win:** *targeted* transition-state search — `saddle_walk` (min-mode/dimer walk on the GP-posterior-mean field) + `transition_state` (seed from two known minima → predict-saddle → evaluate → update). Localizes the transition state in ~10 evaluations (err ~0.03) where random fails at T ≤ 30 (err ~0.45). min-mode is the robust default; Newton-from-midpoint diverges on misaligned pairs.
- **Package fix:** `fit` over-smooths at small n (pure MLE drives ℓ up, can hit the bound); fixed by a default-on MAP lengthscale prior centered on the initial ℓ (`ℓ_prior=:auto`).

Commits (on worktree-critical-points): b0d4a3d (fit prior), 16aafb1 (saddle search), 4e6dbd7 (MB example), a3697dd (volcano example). Full suite 85/85 green; docs build clean.
