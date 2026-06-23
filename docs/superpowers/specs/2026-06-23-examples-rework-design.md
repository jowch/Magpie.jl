# Worked Examples Rework — Design

**Date:** 2026-06-23
**Status:** Design — for user review before an implementation plan.
**Relates to:** the multi-output spine and destructure-based `fit` added this branch; [`docs/src/`](../../src) (Documenter site); the [Matérn derivative-variance note](2026-06-23-matern-derivative-variance.md) (r²-Taylor path); the existing `test/test_exemplar_*.jl` and `test/test_saddle.jl`.

## Motivation

The current exemplars live as buried test files and read as *feature demos* on toy problems (unit circle, unit disk, "enumerate all 9 critical points"). They demonstrate the package but don't look like problems a user would actually bring. This phase reworks them into **realistic, narrative-driven Literate.jl tutorials** that render to the docs site and still run as regression tests — and, in doing so, exercises the new capabilities added this branch (ARD / composite kernels, the destructure-based `fit`, and a custom r²-Taylor Matérn kernel).

**Tenor:** collegial and diplomatic, plain not academic. **The narrative is structural** — it lives in the section arc (problem → why it's hard → how we model it → build the approach → results/diagnostics), not in the prose. **The prose stays plain and expository** — e.g. "In chemistry, locating a transition state is expensive because each energy evaluation is a DFT calculation. We can model this kind of problem with a GP and…" — grounding the domain and stating what we do, without dramatization or flowery storytelling. Let the structure carry the story; keep sentences economical.

## Format & tooling

Each example is a **Literate.jl source** (`examples/` or `docs/literate/`) that renders to a docs-site page **and** is executed in CI / the test suite (so it can't rot). Adds `Literate` and a plotting backend (e.g. `CairoMakie` or `Plots`) as docs/test deps. Ground-truth assertions stay in the tutorials (lightweight `@assert`/`@test`) so they double as regression tests.

## The three examples

### 1. Müller-Brown transition state (headline — deep)

**Narrative:** You've found *one* stable state of a system (one minimum of an expensive potential energy surface). You want the **transition state** — the escape saddle — leading out of it, spending as few expensive evaluations as possible. NEB/string methods need *both* endpoints; you only have one, so this is a **single-ended saddle search** (the dimer / min-mode-following family), with a GP surrogate replacing the dimer's expensive inner-loop force calls (the "GP-dimer", ~10× fewer oracle calls is the published headline).

**Task & ground truth:** Start from minimum **MA ≈ (−0.558, 1.442)**; locate the escape saddle **S1 ≈ (−0.822, 0.624)** (index-1 saddle into the MC basin). Müller-Brown is the standard 2D benchmark; the analytic form, all three minima, and both saddles are tabulated (see the transition-state research findings). `test_saddle.jl` already defines the potential, normalization, ground-truth constants, and a fixed-feature kernel — reused as the scaffold.

**Method (composed inline in the tutorial, visibly):** condition an `ExactGP` on a few points near MA → run a min-mode walk (`saddle_walk`) on the GP mean to predict the saddle → choose the next true-PES evaluation by uncertainty (`GradStraddle` / `acquire`, gated on gradient variance) → re-condition → repeat until converged → verify it's a first-order saddle. Every query and acquisition value stays accessible for plotting (this also sidesteps the audit's "the `transition_state` loop is invisible" friction — we don't call the opaque two-endpoint `transition_state`).

**Kernel — the r²-Taylor extensibility demo:** the PES is rough and anisotropic, so the physically honest prior is an **anisotropic Matérn**. Plain KernelFunctions Matérn NaNs under AD at coincidence, so the tutorial **defines a custom Matérn kernel evaluated in r² with a polynomial-Taylor branch near 0** (the CovarianceFunctions.jl technique, MIT — attribute it). This is a worked extensibility example *and* the "apply the r²-Taylor here" deliverable. **Verified by spike** (2026-06-23): a custom ARD Matérn-3/2 of this form gives finite, correct `grad_predict` output — isotropic `Σfar = 3/ℓ²`, anisotropic `Σfar = [3σ²/ℓ₁², 3σ²/ℓ₂²]` exactly — with **zero changes to `grad_predict`/`_prior_grad_var`** (the kernel is smooth-in-r², so the generic AD path just works, including per-dimension).

**Analyses (the BO-chemistry checklist):**
- PES contour with the five critical points labeled.
- GP surrogate evolution (posterior mean ± std) at increasing N, with sampled points overlaid.
- The oracle-evaluation path climbing out of MA toward S1.
- Convergence of the located saddle to S1 (coordinate/energy error vs # evaluations) against a baseline (random sampling; optionally a "classical dimer" call-count contrast).
- **The correctness diagnostic chemists expect: the Hessian at the located point has exactly one negative eigenvalue** (index-1 saddle), plus the min-mode eigenvector along the path.
- Reaction-energy profile along the path with the GP uncertainty band.
- A summary table (coordinate error, energy error, oracle count, Hessian index).

### 2. Level-set / feasibility boundary (light modernization of A1)

**Narrative:** find the threshold contour of an expensive response — a feasibility / process window — in few evaluations. Reframe the unit-circle toy as a recognizable 2D feasibility problem whose axes have **different physical scales**, motivating an **ARD squared-exponential** kernel (kernel composition, per the modernization goal). `Straddle` acquisition.

**Analyses (standard, light):** posterior mean ± std; the acquisition surface; queries clustering on the boundary; estimated level set vs ground truth; level-set recovery (F1 / sign-recovery) and #queries-to-tolerance vs a random/grid baseline.

### 3. Decision-boundary classification (light modernization of A2)

**Narrative:** map a binary success/failure boundary (e.g. a deposition-stability window) over a 2D parameter space in few expensive trials. `LaplaceGP` + `BinaryBALD`, with an **ARD SE** kernel where sensible. Honest one-line note about the v1 `LaplaceGP` fit no-op.

**Analyses (standard, light):** posterior class-probability map with the decision boundary; the BALD acquisition surface (queries concentrate near the boundary); a learning curve (accuracy/AUC vs #queries) against a random and a max-entropy baseline.

## Scope, retirement, and API smoothing

- **Retire the enumeration framing.** `test_exemplar_critpoints.jl` (the "enumerate all 9 critical points" demo with self-retracted claims and private helpers duplicating public API) is not carried into the tutorials; the transition-state story supersedes it. Its useful machinery already lives in the public API and `test_saddle.jl`.
- **Light A1/A2.** Realistic framing + ARD/composition + a couple of standard plots + one baseline each — not a research treatment.
- **Minimal API smoothing — only what the examples need:** document the cold-start requirement (`acquire` needs at least one observation), note the `LaplaceGP` fit no-op in v1, and keep the MB loop inline so its queries/acquisition values are directly plottable. No broad refactor.
- **Multi-output examples are out of scope here** — acquisitions are `d>1`-guarded, and the compelling multi-output story is the GP-as-ODE-RHS of Capability B; multi-output examples belong there.

## Decisions (recorded)

- **Format:** Literate.jl tutorials, rendered to docs + run as tests.
- **MB reframe:** a *single* known minimum (MA) → find the escape saddle (S1) via single-ended min-mode search; **loop composed inline in the tutorial** from `saddle_walk` + `GradStraddle` + `acquire` + `grad_predict` (no new opaque src function).
- **r²-Taylor:** realized as a **custom anisotropic Matérn kernel defined in the MB tutorial** (extensibility demo), borrowing the r²-Taylor pattern from CovarianceFunctions.jl (MIT — reproduce attribution). Spike-verified; needs no change to `grad_predict`.
- **A1/A2:** light modernization, using ARD SE / kernel composition where sensible.
- **Tenor:** collegial, narrative-driven.

## Open items for the plan

- Plotting backend choice (CairoMakie vs Plots) and docs/CI wiring for running Literate examples as tests.
- Whether the custom r²-Taylor kernel stays tutorial-local (recommended — it's an extensibility demonstration) or is promoted to a small `src/` helper later.
- Baseline depth for MB (random only, vs adding a classical-dimer call-count contrast).
- The level-set / classification realistic scenarios' exact closed-form ground truth (kept analytic, no external data).

---

## Scope correction (as-built, 2026-06-23)

The worked examples **already exist** as developed Literate tutorials in `examples/` (rendered via `docs/make.jl`, run as CI anti-rot tests, **Plots.jl** backend — already wired). So this is **rework of four existing files**, not creation, and the backend/tooling decisions are already made (use Plots; no new infra). Per-file plan:

- **`examples/muller_brown.jl`** (339 lines) — major surgery. Currently does *both* global critical-point enumeration *and* transition-state search between *two* known minima (`transition_state(mbt, MB_min, MC_min)`), with an "honest detour" active-vs-random enumeration benchmark and a convergence-comparison plot. **Cut** the global-enumeration half, the honest-detour benchmark, and the two-minima convergence plot (~150 lines). **Reframe** to a *single* known minimum → escape saddle via a single-ended min-mode/GP-dimer loop composed **inline**. **Swap** the fixed isotropic SE (its z-score/clip preprocessing currently substitutes for per-axis scaling) for the **custom anisotropic r²-Taylor Matérn** (spike-verified: finite + exact ARD `grad_predict`). Keep the potential, truth constants, preprocessing, and the saddle-walk animation pattern.
- **`examples/levelset_straddle.jl`** (135 lines) — light. Change the unit circle to an **ellipse** so an **ARD SE** kernel (`SqExponentialKernel() ∘ ARDTransform([ℓ1,ℓ2])`, fittable via `fit`) is genuinely motivated; +1 realistic-framing sentence. Keep the loop, figures, and `#src` assertions (retune thresholds).
- **`examples/bald_classification.jl`** (187 lines) — minimal. **Keep isotropic SE** — explored and confirmed ARD is not sensible here (`fit` is a no-op for `LaplaceGP` so ARD can't be recovered; hand-set ARD gave 0.559 vs isotropic 0.561 on an anisotropic boundary, within noise). Trim the over-packed opening (three cost examples → one) and the long Figure-2 comment.
- **`examples/volcano_terrain.jl`** (150 lines) — keep + light ARD modernization. Real volcano DEM Morse critical points (kriging = GP); the only example that calls `fit`. Swap `SqExponentialKernel() ∘ ScaleTransform(6.0)` → ARD SE (still fitted by `fit`). After MB drops its enumeration half, this becomes the sole "recover all Morse critical points" example — a clean complement. Keep narrative and data.

**Example count: four** (the spec's original three + `volcano_terrain.jl`). The `test/test_exemplar_*.jl` files are the *regression tests*; the `examples/*.jl` are the rendered tutorials — reconcile the exemplar tests with the reworked tutorials so both stay green.
