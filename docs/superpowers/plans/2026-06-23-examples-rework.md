# Worked Examples Rework Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the four existing Literate.jl tutorials in `examples/` into realistic, problem-driven worked examples that exercise this branch's new capabilities (ARD/composite kernels, destructure-based `fit`, `LaplaceGP` fitting, a custom r²-Taylor Matérn kernel).

**Architecture:** Edit the existing `examples/*.jl` Literate sources (rendered by `docs/make.jl`, run as CI anti-rot tests via `julia --project=docs examples/<file>.jl`, Plots.jl backend). The headline `muller_brown.jl` is refocused on single-ended transition-state search from *one* known minimum, using a tutorial-local custom anisotropic Matérn kernel (r²-Taylor). The other three get lighter modernizations. No `src/` changes are required (Plan A already added `LaplaceGP` fitting; the custom kernel lives in the tutorial).

**Tech Stack:** Julia ≥1.10, Magpie, KernelFunctions.jl, Plots.jl, Literate.jl, Documenter.jl.

Source of truth: [`docs/superpowers/specs/2026-06-23-examples-rework-design.md`](../specs/2026-06-23-examples-rework-design.md) (read it, incl. the "Scope correction (as-built)" section).

## Global Constraints

- **Tenor: narrative is structural, prose is plain.** The story lives in the section arc (problem → why it's hard → how we model it → build the approach → results/diagnostics); the prose is plain and expository ("In chemistry, locating a transition state is expensive because each energy evaluation is a DFT calculation. We can model this kind of problem with a GP and…"), not dramatized. Keep sentences economical. Trim overwrought existing passages.
- **Each reworked example must run standalone** as an anti-rot test: `julia --project=docs examples/<file>.jl` exits 0 (its `#src` `@assert`/`@test` lines pass). Instantiate the docs env first if needed: `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`.
- **The full package suite stays green** (`julia --project=. -e 'using Pkg; Pkg.test()'`, currently 161) — the `test/test_exemplar_*.jl` regression tests are separate from `examples/*.jl` and should remain passing; do not break them.
- **ARD/kernel composition only where it's genuinely motivated and fittable** — `levelset_straddle.jl` (ellipse) and `volcano_terrain.jl` (DEM) use ARD SE via `fit`; `bald_classification.jl` stays isotropic (symmetric truth; ARD not sensible) but now *fits* its lengthscale (Plan A made `LaplaceGP` fittable); `muller_brown.jl` uses the custom anisotropic Matérn.
- **Plots.jl backend** (already a docs dep). `ENV["GKSwstype"]="100"` for headless rendering (already in the files).
- Format with Runic before each commit; run the example standalone AND the full suite before each commit.

## File Structure

| File | Change | Task |
|---|---|---|
| `examples/muller_brown.jl` | major: custom anisotropic r²-Taylor Matérn kernel; single-ended escape-saddle from one minimum; drop enumeration/two-minima halves; BO-chemistry analyses | 1 |
| `examples/levelset_straddle.jl` | light: ellipse target + ARD SE (fitted); +1 framing sentence | 2 |
| `examples/bald_classification.jl` | light: keep isotropic but **fit** the lengthscale; trim prose | 3 |
| `examples/volcano_terrain.jl` | light: ARD SE via `fit` | 4 |
| `docs/make.jl` | update the `muller_brown` example title if framing changed | 1 |

---

### Task 1: `muller_brown.jl` — single-ended transition-state search (headline)

**Files:**
- Modify: `examples/muller_brown.jl` (full rework)
- Modify: `docs/make.jl:12` — example title

**Interfaces:**
- Consumes: `ExactGP`, `update`, `grad_predict`, `saddle_walk`, `classify`, `GradStraddle`, `acquire`, `Box`, `grid_points` (all existing public API); the MB constants/normalization already in the file (`mullerbrown`, `mbt`, `box`, `MC_min`, `S2`, `NOISE`, etc.).
- Produces: a tutorial-local `TaylorMatern32` kernel and an inline `escape_saddle` loop (both verified by spike 2026-06-23).

**Context — what to keep vs cut (from the audit):** keep the potential definition, `Vclip`/contour intro plot, the `CEIL`/z-score normalization (`mbt`), `box`, and the five truth constants (`MA_min`, `MB_min`, `MC_min`, `S1`, `S2`). **Cut** the global critical-point enumeration half (the `critical_points`/`cov_pts`/`plt_recovered` section), the "honest detour" active-vs-random enumeration benchmark, and the two-minima `transition_state(mbt, MB_min, MC_min)` convergence-comparison section (the `targeted_hist`/`random_err`/`raw_saddles`/`plt_conv` machinery). The new story is single-ended: *given one known minimum, find its escape transition state.*

- [ ] **Step 1: Define the custom kernel (with the verification assertion)** — after the setup/normalization section, add (spike-verified to give finite, correct, anisotropic `grad_predict`):

```julia
# ## A custom anisotropic Matérn kernel
#
# The potential is rough and the reaction valley is narrow across the path but flat along it, so an
# anisotropic Matérn is the honest prior. Plain `MaternKernel` is non-differentiable at r=0, which
# breaks the gradient-variance term `grad_predict` needs (the squared distance hides a √); we sidestep
# it by evaluating the kernel in r² with a short Taylor branch near 0 (technique from
# CovarianceFunctions.jl, MIT-licensed). One lengthscale per input axis gives the anisotropy.

import KernelFunctions as KF
struct TaylorMatern32{V} <: KF.Kernel       #src (defined for the docs build; shown in prose above)
    invℓ::V
end
function (k::TaylorMatern32)(x, y)
    s = sum(abs2, (x .- y) .* k.invℓ)                       # r² = Σ ((xᵢ-yᵢ)/ℓᵢ)²
    if s < 1.0e-6
        return 1 - 1.5 * s - 1.125 * s^2                    # Taylor of ψ(r²) near 0 (AD-smooth)
    else
        r = sqrt(s)
        return (1 + sqrt(3) * r) * exp(-sqrt(3) * r)
    end
end

mbkernel() = TaylorMatern32(1.0 ./ [0.35, 0.45])            # anisotropic feature scales (x, y)

## verify the custom kernel gives a finite, anisotropic prior gradient variance         #src
let gchk = update(ExactGP(mbkernel(); noise = NOISE), grid_points(box; per_axis = 6), mbt.(grid_points(box; per_axis = 6)))   #src
    _, Σ, H = grad_predict(gchk, [0.0, 0.5])                #src
    @assert all(isfinite, Σ) && all(isfinite, H)           #src custom r²-Taylor kernel is AD-clean in grad_predict
end                                                         #src
```

(Note: the existing file defines `mbkernel()` as isotropic SqExp; replace that definition with this one. Keep `buildgp` if used, or inline `update(ExactGP(mbkernel(); noise=NOISE), pts, mbt.(pts))`.)

- [ ] **Step 2: Write the single-ended escape-saddle loop** — the heart of the rework (verified by spike: from `MC_min`, ~33 evals → predicted saddle 0.009 from S2, classified `:saddle`):

```julia
# ## Finding the escape transition state from one known minimum
#
# We know one stable state (`MC_min`, the deep right basin) and nothing else. To find the transition
# state leading out of it we climb the softest local mode (a gentlest-ascent / dimer walk) on the GP
# *mean*, evaluate the true potential where the walk predicts the saddle plus one GradStraddle point
# to reduce gradient uncertainty along the way, re-condition, and repeat. The GP surrogate replaces the
# expensive inner force evaluations a classical dimer would need.

using LinearAlgebra: eigen, Symmetric
function escape_saddle(f, m0, kernel, box; budget = 14, noise = NOISE, rng = Random.default_rng())
    pts = [clamp.(m0 .+ 0.08 .* randn(rng, 2), box.lb, box.ub) for _ in 1:5]
    push!(pts, collect(float.(m0)))
    g = update(ExactGP(kernel; noise = noise), pts, f.(pts))
    hist = Tuple{Int, Vector{Float64}, Symbol}[]
    res = (; x = collect(float.(m0)), H = zeros(2, 2))
    for t in 1:budget
        v = eigen(Symmetric(grad_predict(g, m0)[3])).vectors[:, 1]    # softest mode at the basin
        # climb both signs on the surrogate; keep the branch that reaches an index-1 saddle
        walks = [saddle_walk(g, clamp.(m0 .+ 0.2 .* s .* v, box.lb, box.ub); box = box) for s in (1.0, -1.0)]
        saddles = filter(w -> classify(w.H) == :saddle, walks)
        res = isempty(saddles) ? argmin(w -> w.residual, walks) : argmin(w -> w.residual, saddles)
        xacq = acquire(g, GradStraddle(β = 1.96); over = box)         # explore high gradient-uncertainty
        g = update(g, [res.x, xacq], f.([res.x, xacq]))
        push!(hist, (length(pts) + 2t, copy(res.x), classify(res.H)))
    end
    return res, g, hist
end

Random.seed!(1)
res, gfit, hist = escape_saddle(mbt, MC_min, mbkernel(), box; budget = 14)
@info "escape transition state" predicted = round.(res.x; digits = 4) truth_S2 = S2 err = round(norm(res.x .- S2); digits = 4) kind = classify(res.H)
```

- [ ] **Step 3: Add the analyses (plain prose + Plots), in this order.** Each is a short section; keep prose plain. Use the existing `Vclip`/contour helpers where present.

1. **The landscape (orientation):** filled contour of the (clipped) potential with the five critical points marked (minima as circles, saddles as ×), and `MC_min` highlighted as "the one we know." (Reuse the existing intro contour plot; add the markers.)
2. **Surrogate growth:** GP posterior mean (and ±std via `mean_and_var`) at two or three snapshots of the loop (e.g. after 5, 15, 30 evaluations), with sampled points overlaid — show the surrogate filling in along the climb. (Capture intermediate GPs by running `escape_saddle` with smaller budgets, or snapshot inside a copy of the loop.)
3. **The climb:** the true-PES contour with the sequence of evaluation sites as a path out of `MC_min` toward S2 (use `queried_points`/the loop's accumulated `pts`, or plot `[h[2] for h in hist]`).
4. **Correctness diagnostic (the key check):** the Hessian at the located point has exactly one negative eigenvalue. Compute and show:
```julia
λ = eigen(Symmetric(grad_predict(gfit, res.x)[3])).values
@info "Hessian eigenvalues at the located saddle" λ = round.(λ; digits = 3) index = count(<(0), λ)
## a first-order transition state has Morse index 1                                  #src
@assert classify(grad_predict(gfit, res.x)[3]) == :saddle                            #src located point is an index-1 saddle
```
5. **Convergence vs a baseline:** localization error `‖predicted − S2‖` vs evaluation count from `hist`, against a random-sampling baseline (uniform points in `box`, GP built, nearest index-1 saddle of the mean to S2). Keep this lighter than the old 8-seed benchmark — one curve each is enough to make the point.
6. **A short takeaway** (plain): the GP-surrogate single-ended search finds the transition state from one basin in a few dozen evaluations; the diagnostic confirms it is a genuine index-1 saddle.

- [ ] **Step 4: Add the ground-truth `#src` assertions** at the end (mirroring the old file's, adapted to single-ended):

```julia
@assert classify(res.H) == :saddle                          #src located an index-1 saddle
@assert norm(res.x .- S2) < 0.1                             #src and it is the escape TS S2
```

- [ ] **Step 5: Update the example title** — in `docs/make.jl:12`, change:

```julia
push!(EXAMPLES, ("Müller–Brown: critical points & transition state", "muller_brown.jl"))
```

to:

```julia
push!(EXAMPLES, ("Müller–Brown: a transition state from one known minimum", "muller_brown.jl"))
```

- [ ] **Step 6: Run the example standalone, then the full suite**

Run: `julia --project=docs examples/muller_brown.jl` → Expected: exits 0 (all `#src` assertions pass; figures render headlessly).
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green (the example is not in `Pkg.test()`, but confirm nothing regressed).

- [ ] **Step 7: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add examples/muller_brown.jl docs/make.jl
git commit -m "docs(examples): Müller–Brown — single-ended transition state from one minimum (custom anisotropic r²-Taylor Matérn)"
```

---

### Task 2: `levelset_straddle.jl` — ellipse + ARD SE

**Files:**
- Modify: `examples/levelset_straddle.jl`

**Interfaces:**
- Consumes: `ActiveLearner`, `observe!`, `run!`, `posterior_gp`, `queried_points`, `Straddle`, `Box`, `fit` (existing).

- [ ] **Step 1: Reframe the target to an anisotropic ellipse** — replace the unit-circle `f(x) = ‖x‖ - 1` with an ellipse whose axes differ, so ARD is genuinely motivated:

```julia
## a feasibility boundary that is wider in x₁ than x₂ — different physical scales per axis
f(x) = (x[1] / 1.6)^2 + (x[2] / 0.8)^2 - 1.0
```

Add one plain framing sentence, e.g.: "In screening tasks the feasible region is often an anisotropic contour — wider along one control than another — so we let the GP learn a separate lengthscale per axis (ARD)."

- [ ] **Step 2: Use an ARD SE kernel and fit it** — replace the isotropic `with_lengthscale(SqExponentialKernel(), 0.5)` construction with ARD, and fit the lengthscales:

```julia
al = ActiveLearner(ExactGP(with_lengthscale(SqExponentialKernel(), [0.5, 0.5]); noise = 1.0e-4), Straddle(h = 0.0))
```

Keep `run!(al, f; budget = 40, refit_every = 10)` — `refit_every` now fits the two ARD lengthscales. After the loop, show the recovered lengthscales (plain): `@info "recovered ARD lengthscales" ℓ = round.(1 ./ posterior_gp(al).prior.kernel.kernel.transform.v; digits = 3)` (longer along x₁, the wider axis).

- [ ] **Step 3: Keep the figures and assertions** — the posterior-mean + level-set figure and the acquisition-surface figure stay; retune the `#src` recovery thresholds if needed for the ellipse (sign-match recovery on the grid; queries near the boundary). Keep the same assertion structure.

- [ ] **Step 4: Run standalone, then the full suite**

Run: `julia --project=docs examples/levelset_straddle.jl` → Expected: exits 0.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green.

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add examples/levelset_straddle.jl
git commit -m "docs(examples): level-set — anisotropic ellipse + fitted ARD SE kernel"
```

---

### Task 3: `bald_classification.jl` — fit the classifier, trim prose

**Files:**
- Modify: `examples/bald_classification.jl`

**Interfaces:**
- Consumes: `LaplaceGP`, `BinaryBALD`, `ActiveLearner`, `observe!`, `run!`, `posterior_gp`, `Box`, `fit` (Plan A made `LaplaceGP` fittable).

- [ ] **Step 1: Keep the isotropic kernel but fit it** — the checkerboard truth is symmetric, so ARD is not sensible (explored: hand-set ARD ≈ isotropic). Instead demonstrate the new capability that *is* sensible: fitting the classifier's lengthscale. Change the `run!` to refit:

```julia
run!(al, label; budget = 40, refit_every = 10, over = box)   # LaplaceGP now fits its lengthscale (Plan A)
```

Add a plain sentence: "Earlier the classifier's lengthscale was fixed; we now let it fit the data by re-optimizing the Laplace evidence every few queries." After the loop, report the fitted lengthscale: `@info "fitted lengthscale" ℓ = round(Magpie._lengthscale(posterior_gp(al).prior.kernel); digits = 3)`.

- [ ] **Step 2: Trim the prose** — shorten the over-packed opening (keep one cost example, e.g. drug-discovery, drop the other two) and the long Figure-2 inline comment (~3 lines). Keep the learning-curve methodology (paired seeds, ±SD ribbons) and the figures.

- [ ] **Step 3: Keep the assertions** — the learning-curve `#src` assertions (BALD beats random, reaches accuracy by budget) stay; re-run to confirm they still pass with the fitted kernel (fitting should help or match).

- [ ] **Step 4: Run standalone, then the full suite**

Run: `julia --project=docs examples/bald_classification.jl` → Expected: exits 0.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green.

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add examples/bald_classification.jl
git commit -m "docs(examples): BALD classification — fit the classifier lengthscale; trim prose"
```

---

### Task 4: `volcano_terrain.jl` — ARD SE via fit

**Files:**
- Modify: `examples/volcano_terrain.jl`

**Interfaces:**
- Consumes: `ExactGP`, `update`, `fit`, `critical_points`/`newton_polish`/`classify` as currently used.

- [ ] **Step 1: Switch the kernel to ARD SE** — replace `with_lengthscale(SqExponentialKernel(), ...) ∘ ScaleTransform(6.0)` (or the current isotropic construction) with an ARD SE, still fitted by `Magpie.fit(g; restarts=5)`:

```julia
g = update(ExactGP(with_lengthscale(SqExponentialKernel(), [1 / 6, 1 / 6]); noise = 0.01), pts, h.(pts))
g = Magpie.fit(g; restarts = 5)   # fits per-axis lengthscales for the terrain
```

Add a plain sentence noting the terrain's mild east–west vs north–south anisotropy and that ARD fits a separate scale per direction. Keep the narrative, data loading, interior/variance filters, and the recovered-critical-points figure.

- [ ] **Step 2: Keep the assertions** — the `#src` checks (≥1 max/min/saddle recovered; all interior) stay; re-run to confirm they pass with the ARD-fitted kernel.

- [ ] **Step 3: Run standalone, then the full suite**

Run: `julia --project=docs examples/volcano_terrain.jl` → Expected: exits 0.
Run: `julia --project=. -e 'using Pkg; Pkg.test()'` → Expected: green.

- [ ] **Step 4: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add examples/volcano_terrain.jl
git commit -m "docs(examples): volcano DEM — fitted ARD SE kernel"
```

---

### Task 5: Build the docs and final reconciliation

**Files:**
- Possibly modify: `examples/*.jl` (only if the docs build surfaces an issue), `docs/make.jl`

- [ ] **Step 1: Run all four examples standalone** (anti-rot):

```bash
for f in muller_brown levelset_straddle bald_classification volcano_terrain; do julia --project=docs examples/$f.jl || echo "FAILED: $f"; done
```
Expected: all exit 0.

- [ ] **Step 2: Build the docs** (renders the Literate sources + API):

Run: `julia --project=docs docs/make.jl`
Expected: builds without error (warnings allowed per `warnonly=[:missing_docs]`); the four example pages and the updated Müller–Brown title appear.

- [ ] **Step 3: Full package suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: green (161+). The `test/test_exemplar_*.jl` regression tests are unchanged and still pass.

- [ ] **Step 4: Commit any final fixups**

```bash
git ls-files -z -- '*.jl' | xargs -0 /home/jonathanchen/.julia/bin/runic --inplace
git add -A
git commit -m "docs(examples): build-green reconciliation of reworked tutorials"
```

---

## Self-Review

**Spec coverage** (examples-rework design + scope correction → tasks):
- Müller–Brown refocused on single-ended transition state from one minimum, custom anisotropic r²-Taylor Matérn, BO-chemistry analyses, enumeration/two-minima halves cut → **Task 1** ✓ (loop + kernel spike-verified)
- Level-set ellipse + fitted ARD SE → **Task 2** ✓
- BALD classification stays isotropic but fits its lengthscale (Plan A), prose trimmed → **Task 3** ✓
- Volcano DEM ARD SE via fit → **Task 4** ✓
- Literate/Plots/anti-rot format preserved; docs build green → **Task 5** ✓
- Tenor (structural narrative, plain prose) → Global Constraints, applied per task ✓

**Placeholder scan:** the analyses in Task 1 Step 3 and the prose edits are specified as directional (plain-prose sections + Plots figures) rather than line-pinned — deliberate, per the tenor (don't over-specify prose/plots); all *logic* (the custom kernel, the `escape_saddle` loop, every `#src`/`@assert`, the kernel swaps, the `fit`/`refit_every` calls, the title edit) is given as complete, spike-verified code.

**Type consistency:** `TaylorMatern32(invℓ)` with `invℓ = 1.0 ./ [ℓx, ℓy]`; `escape_saddle(f, m0, kernel, box; budget, noise, rng) -> (res, g, hist)` where `res` has `.x`/`.H` (matching `saddle_walk`'s return) and `classify(res.H)` gives the Morse type. ARD lengthscale read-back uses `.kernel.transform.v` (the `ARDTransform` field is `.v`, confirmed in Plan for the multi-output work). `Magpie._lengthscale` reads a scalar isotropic lengthscale (Task 3).

## Notes

- The custom `TaylorMatern32` stays tutorial-local (an extensibility demonstration), per the design decision; promote to `src/` only if a later use case needs it.
- `test/test_exemplar_*.jl` (the in-suite regression tests) are intentionally left as-is; they remain the `Pkg.test()` backstop while `examples/*.jl` are the rendered, anti-rot'd tutorials.
