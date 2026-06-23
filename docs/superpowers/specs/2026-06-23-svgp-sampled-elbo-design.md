# SVGP-UDE Sampled-ELBO — Design Spec

**Status:** proposed (2026-06-23)
**Supersedes:** the SVGP training path of the v0.1 GP-UDE bridge (mean-field + local-trace surrogate).
**Round scope:** the combined SVGP follow-up arc — replace the SVGP-UDE objective, then enable SVGP+MultipleShooting and CompositeField+SVGP-residual as special cases, to the same correctness standard as the ExactGP path. **Optimize for correctness, not cost.**

---

## 1. Problem — why the current SVGP-UDE objective is inadequate

The GP is the ODE right-hand-side `du/dt = f(u)`. We observe **states** `(tᵢ, uᵢ)` (the time-integral of `f`), not the field — training integrates `f` through the solver and fits the state misfit (through-solver, `GaussAdjoint`+`MooncakeVJP`, outer Mooncake).

The current SVGP path (`svgp_elbo_loss` → `field_loss` → `shooting_data_term` → `regularizer`) optimizes:

```
L(v) = Σᵢ ‖u_meanfield(tᵢ) − Xᵢ‖²/(2σ²) + (Nd/2)log(2πσ²)   ← integrate the MEAN field
     + Σᵢ var_q(f(u(tᵢ)))/(2σ²)                              ← local "trace correction" at saved states
     + KL(q(u)‖p(u)) + priors
```

This is a **mean-field plug-in + local-collocation surrogate**, not the through-solver ELBO. The trace term treats each saved state as a *direct, noisy observation of the field at that point* — correct for SVGP regression (direct `f` observation), wrong here (field uncertainty propagates through the solver with a different covariance). Three concrete defects:

1. **MultipleShooting structurally drops the variational term.** `shooting_data_term(::MultipleShooting)` (`ext/MagpieSciMLExt.jl:218–246`) accepts `uvar` and never uses it. For ExactGP that is correct (`uvar ≡ 0`); for SVGP it means even a wired-up SVGP+MS would discard the field-variance correction — it stops being an ELBO.
2. **Train/predict model mismatch.** Training uses the mean field + local trace; prediction/propagation uses decoupled (Matheron) **samples** (`_svgp_pathwise_sampler`). Two models — calibration cannot be principled.
3. **Nonlinear under-calibration** (spike, LV: cov90 0.6–0.8) — corroborating, though confounded by under-training; weighted below the two structural facts.

## 2. The objective — sampled reparameterized ELBO

Replace the plug-in with a Monte-Carlo estimate of the **true through-solver ELBO**:

```
L(v) = −(1/S) Σ_{s=1}^{S} log p(X | states(f_s))  +  Σ_i KL(q_i ‖ p_i)  +  priors
```

where each `f_s` is a **decoupled (Matheron) sample** from the variational posterior, `states(f_s)` is the ODE solution with field `f_s`, and `log p(·)` is the Gaussian state-NLL already in `_gaussian_nll`. Per output dim `i`, the sample is

```
f_{s,i}(x) = wᵀφ(x) + Σⱼ k(x,Zⱼ)·v_{corr},   φ(x)=√(2/D)cos(ω'x+b)
ω = ε_ω/ℓ,  w = σ·ε_w,  u_s = L_ZZ(μ_i + L_{S,i}·ε_ind),  v_corr = K_ZZ⁻¹(u_s − Φw)
```

**Reparameterization (correctness requirement):** the noise `ε = (ε_ω, ε_b, ε_w, ε_ind)`, indexed by `(sample s, output i)`, is **frozen for the whole optimization run** (drawn once, seeded, captured at loss construction). This makes `L(v)` deterministic — required for LBFGS and for reproducible tests. Only `ε` is captured; all trained params (`logℓ, logσ, Z, μ, L_S`) flow through (R1).

**Variance reduction:** antithetic pairing (`ε` and `−ε`) by default; `S` is even.

**Feasibility:** confirmed (2026-06-23 spike, [[svgp-sampled-elbo-mooncake-feasible]]) — differentiates through `GaussAdjoint`+`MooncakeVJP`, Mooncake-vs-FD max rel.err 2.2e-4 across `logℓ/logσ/Z/μ/L_S`, including the trained-frequency RFF prior term.

## 3. Architecture — one objective, three consumers

The loss loops `S` samples, builds a per-sample `pf`, integrates via the **field-agnostic** `shooting_data_term` (single- or multiple-shooting), averages the data term, adds KL once. The shooting engine stays field-agnostic; it is fed *sampled* fields instead of the mean field.

- **Per-sample RHS / `pf` threading.** A new `field_rhs`-style builder for SVGP **samples**: thread `[logℓ, logσ, vec(ω_s), w_s, vec(Z), v_corr_s]` into `pf` (R1); capture the constant `b_s`. Relative `K_ZZ` jitter `field.jitter·σ²` (keeps the Mooncake Cholesky backward safe — same convention as the existing path). RHS adds the RFF prior + canonical update.
- **SVGP + SingleShooting (replaces mean-field).** Loop `S` samples, integrate each over `tspan`, average the per-sample state-NLL, add KL. The deterministic posterior (`predmean`/`var` via `svgp_moments`) is **unchanged** and remains the cheap prediction path.
- **SVGP + MultipleShooting.** Per-segment node `s0` (dout×S_seg) packed after the field prefix (`offset = length(field.v0)`, the established convention). Each sample is integrated per-segment from the shared `s0`; the data-NLL, continuity penalty, and anchor penalty are computed per-sample and **averaged over samples** (the penalties are part of the per-sample objective, so they hold in expectation). KL once. New `_train_loss`/`_train_init` methods for `(::SVGPField, ::MultipleShooting)`.
- **CompositeField + SVGP residual.** Each sample's RHS becomes `du = known(u,t) + f_s(u)` — identical to the existing Pathwise propagation loop (`_pathwise_integrate`, `:615`). Add `gpfield(::SVGPField, u, pf)` for the deterministic posterior-mean residual (used by `posterior`/cheap prediction, not the training objective). Remove the two `_assert_shooting_supported` guards for the now-supported combos.

**No-op for ExactGP/Composite-with-Exact:** the ExactGP path is unchanged (exact, no sampling). Only SVGP-inner training changes.

## 4. API surface

- `train!(field, data; nsamples=16, shooting=…, …)` — `nsamples` (even; antithetic) is the new knob; default 16. All other kwargs unchanged. Return contract unchanged (returns `field`).
- The `_assert_shooting_supported(::SVGPField, ::MultipleShooting)` and `(::CompositeField{SVGP})` guards are **removed** (those combos now work). No other guard changes.
- Exports unchanged. `kmeans_anchors` stays exported; decoupled-sampler internals stay un-exported (training uses them internally).
- **Breaking (intended):** SVGP+SingleShooting now trains the sampled ELBO, not the mean-field surrogate. Existing SVGP test expectations (`test_gpude_calibration.jl`, `test_gpude_noise.jl`) are **re-derived** under the new objective — a reviewed change, not a regression.

## 5. Correctness gates (the "professional standard")

Mirroring what stage2/calibration did for the Exact path. Each is a gated SciML test (`MAGPIE_TEST_SCIML=true`, see [[test-sciml-gate]]).

1. **Gradient gate** — Mooncake vs central-FD `< 5e-3` for: SVGP+SingleShooting sampled, SVGP+MultipleShooting sampled (incl. `∂loss/∂s0` liveness, as in `test_gpude_stage2.jl`), CompositeField+SVGP.
2. **Determinism gate** — `loss(v)` returns bitwise-identical values across repeated calls (ε frozen).
3. **Linear-field equivalence** — on a *linear* field `du=Au`, the sampled data term converges (S→large) to the analytic mean-field+trace value. For linear ODEs the state is a linear functional of the field, so the local-trace picture is exact: this validates the Monte-Carlo estimator against the known-correct analytic form in the regime where they must agree.
4. **`M=N` limit** — with inducing points at the data and the deterministic posterior, SVGP reproduces the ExactGP moments (extends the existing `test_gpude_svgp.jl` invariant).
5. **Calibration gate (the real one)** — held-out coverage cov90 ≈ 0.9 (`|cov−0.9| < 0.15`) on a **nonlinear** system (LV), for SingleShooting *and* MultipleShooting. This is where only the sampled ELBO is correct.
6. **Solver robustness** — the forward divergence sentinel (`:195`) and the relative-jitter backward `potrs` guard both hold when the RHS is a sampled field (the `dt_NaN` path observed in the recovery spike).

## 6. Scope / non-goals

- **In:** sampled-ELBO objective; SVGP+SingleShooting (replacement), SVGP+MultipleShooting, CompositeField+SVGP; `gpfield(::SVGPField)`; antithetic variance reduction; the six gates above; re-derived SVGP test expectations.
- **Out (deferred):** control variates beyond antithetic; adaptive `S`; multi-class BALD; the cost/scaling benchmark (separate from correctness — `bench/`); inducing-point count selection heuristics; PULL for CompositeField (Pathwise remains the supported propagator).
- **Cost:** `S×` solves per gradient is accepted — correctness over cost. A cost study is a separate, later artifact.

## 7. Risks / open questions

- **Gradient variance at small S** → antithetic + default `S=16`; the gradient gate uses a fixed seed so it is deterministic.
- **MS continuity-penalty semantics under sampling** → resolved: penalties are per-sample, averaged (Section 3).
- **Cost** → accepted; flagged, not optimized, this round.
- **Re-deriving existing test expectations** → done as part of the gates, reviewed (Section 4).
