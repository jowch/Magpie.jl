# GP-UDE Correctness Foundation — Design Spec

**Date:** 2026-06-22
**Branch:** `gp-ude` (Capability B)
**Status:** Approved design → ready for implementation plan
**Round:** Spec 1 of 2 (Spec 2 = "Ergonomic UDE API", separate cycle)

## Goal

Make every uncertainty number the GP-UDE bridge emits something a user can *believe*. After this round, each uncertainty quantity is either (a) genuinely trained by the data, (b) a numerically valid covariance, or (c) honestly documented as an assumption — and each property is pinned by a test that fails if it regresses.

This is the **correctness foundation** for the vision of a professional-grade, actually-useful GP-in-UDE package. Ergonomics (API surface, abstraction completeness, naming) is deliberately deferred to Spec 2; you cannot make a tool pleasant on top of numbers that are wrong.

## Scope

**In scope (6 sections):**
1. SVGP calibrated ELBO — train the variational variance with the data (expected-log-likelihood trace correction).
2. Calibration oracle — a differential coverage test that proves §1 worked, plus a reusable `pathwise_moments` helper.
3. PULL covariance PSD repair — replace the diagonal-only `eps()` clamp with a PSD-cone projection; full-history buffer default.
4. Per-dimension `σ_obs` — heterogeneous observation noise across output dimensions.
5. Divergence-guard honesty — document that it protects only the forward pass; add the missing trigger test.
6. Test hardening — `train!`→MultipleShooting test, divergence-guard test, tighten loose/tautological tests.

**Out of scope (deferred):**
- **State-space SVGP ELBO** (propagating field variance through the solver sensitivity into predicted state covariance). §1 ships the *field-space* expected-log-likelihood, which is the standard collapsed-SVGP-ELBO term and is correct for the GP-regression framing; the state-space version is a documented follow-up.
- Smooth divergence barrier (§5 option b) — the ADAM warm-up already finds the basin; gold-plating until proven needed.
- All Spec 2 items: `CompositeField` generality, `train!`×shooting support matrix, export trimming, naming, `train!` contract.

---

## Section 1 — SVGP calibrated ELBO

### Problem
The SVGP-UDE loss integrates **only the mean field**. `field_rhs(::SVGPField)` threads `α = L_ZZ'\μ` into the ODE param vector `pf`; the variational factor `L_S` enters the loss **only** through the KL regularizer (`regularizer(::SVGPField)` → `svgp_kl`). KL-minimization drives `L_S → I` (the whitened prior), so the SVGP predictive variance collapses to the prior:
`σ²_f(u) = k(u,u) − A'A + ‖L_S'A‖²` with `L_S = I` ⇒ `= k(u,u)` (prior variance everywhere).
Net effect: **vacuously wide, uninformative uncertainty** (over-coverage ≈ 1.0). The function is named `svgp_elbo_loss` and documented as "the exact ELBO," which is misleading — it is a mean-field MAP objective.

### Fix — field-space expected-log-likelihood trace correction
The collapsed-SVGP ELBO data term for a Gaussian likelihood is `Σₙ E_q[log p(yₙ|fₙ)] = Σₙ [log N(yₙ; μ_f(uₙ), σ²_obs) − σ²_f(uₙ)/(2σ²_obs)]`. The first term is already computed (Gaussian NLL on the mean trajectory). We add the missing trace term so the loss (= −ELL) gains:
```
data_term += Σₙ σ²_f(uₙ) / (2 σ²_obs)      # +sign: loss is the NEGATIVE ELL
```
evaluated at the saved trajectory states `uₙ = eachcol(Array(sol))` (saved at the data times `ts` via `saveat=ts` — confirmed). `L_S` now receives a data gradient: variance sharpens where data constrains the field, stays wide off-data.

### Design — `field_var` closure (preserves field-agnosticism)
`shooting_data_term` is documented as field-agnostic and does **not** receive the trained vector `v` (only `field`, `(pf, rhs!)`, `data`, `logσ_obs`). To thread the variance in without breaking that:

- Extend each field's `field_rhs(field, v)` to return a **third value**: a pure post-solve closure `uvar(u) -> Vector` giving per-output field variance at `u`.
  - **SVGPField:** `uvar` captures `(k, Z, L_ZZ, [L_S_i for i in 1:dout])` and returns `[σ²_{f,i}(u) for i in 1:dout]`, where `A = L_ZZ \ k(Z,u)` is shared across outputs (shared inducing set) and `σ²_{f,i} = k(u,u) − A'A + ‖L_Sᵢ'A‖²`.
  - **ExactGPField / CompositeField:** `uvar = _ -> zeros(d)` (no variational-variance term). Composite delegates to its inner field's closure (which is `zeros` for an Exact inner — the only currently-supported inner; SVGP-inner is a Spec 2 concern).
- `field_loss` destructures the 3-tuple and forwards `uvar` into `shooting_data_term` as a defaulted kwarg (`uvar = _ -> zeros(d)`), so MultipleShooting and the Exact path are unchanged (no-op).
- `shooting_data_term` accumulates generically, per output dimension (composes with §4's per-dim `σ_obs`):
  ```
  trace_j += sum(uvar(uₙ)[j] for uₙ in eachcol(A))      # per output j
  ...
  return _gaussian_nll(sse, Nd, logσ_obs) + Σ_j trace_j / (2 σ²_obs,j)
  ```

### AD safety (explorer-confirmed)
- **R1-safe.** R1 forbids capturing `α` *into the ODE rhs/pf*. `uvar` is evaluated post-solve, outside `pf`; `L_S` was never in `pf`. Capturing `L_S`/`L_ZZ`/`Z` in `uvar` is categorically not an R1 violation.
- **Mooncake-composable.** The outer Mooncake AD differentiates both `∂uₙ/∂θ` (from the `GaussAdjoint`+`MooncakeVJP` solve over `Array(sol)`) and the explicit `∂σ²_f/∂θ`. Standard chain rule through `Array(sol)`.
- **Needs a dedicated AD-safe `σ²_f`.** Do **not** reuse `svgp_moments` — its `max(zero, …)` clamp (src/gpude.jl:362, "prediction path only — no through-solver AD here") zeros the `L_S` gradient exactly where it matters. The loss-side variance must use either no clamp (jitter keeps it positive) or a smooth `softplus` floor.

### Watch-outs
- **Naming:** a `field_var(gps, u)` already exists (ext:406, PULL's marginal variance on built GPs). Name the new closure differently (e.g. `uvar` / `field_vartrace`) to avoid shadowing.
- **Sign:** the correction is `+ Σ σ²_f/(2σ²_obs)` in the loss (loss = −ELL). Verify against the derivation during implementation.
- **Rename:** `svgp_elbo_loss` is now an honest ELBO (data + trace + KL). Keep the name; update its docstring to state the field-space framing explicitly.

### Units / boundaries
- `uvar :: u -> Vector{<:Real}` (length = output dim). Pure, no solver, AD-clean.
- `field_rhs :: (field, v) -> (pf, rhs!, uvar)`. The only interface change is the added return value; the no-op default keeps every non-SVGP path byte-identical.

---

## Section 2 — Calibration oracle

### Problem
There is **no** test anywhere that asserts SVGP uncertainty is calibrated against truth. Existing SVGP tests check only finiteness, non-negativity, and KL/moment self-consistency. The `coverage` "perfect prediction → 1.0" test (test_eval.jl:26-34) is tautological and level-independent (`μ = truth` exactly ⇒ maha²=0 ⇒ 1.0 for any level/Σ). (Note: a genuine level-dependent calibration test for `N(0,I)` already exists at test_eval.jl:7-24 — `|c90−0.9|<0.05`, `|c50−0.5|<0.05` — that one is the model to reuse.)

### Fix — differential coverage oracle
A test that fails if `L_S` is not data-trained:
1. Generate ground truth from a known field (start with a **linear field** — closed-form propagated covariance is the cleanest oracle; extend to a nonlinear field validated by a dense Pathwise reference if useful).
2. Observe noisy states; train the SVGP **two ways**: (a) regularized-only (today), (b) trace-corrected (§1).
3. Propagate both via Pathwise (`SVGPField` → `posterior_sparsegps` → `Pathwise` array), convert to per-step `(μs, Σs)` with the new helper, and measure `coverage(truth, μs, Σs; level=0.9)` on **held-out** data (held-out initial conditions or a held-out trajectory segment).
4. **Assert calibrated AND sharp together:**
   - (a) regularized-only **over-covers** (coverage ≈ 1.0) with large mean predictive variance;
   - (b) trace-corrected coverage lands near nominal (e.g. `|cov − 0.9| < 0.15`, one round; tolerance set empirically) **and** mean predictive variance is materially smaller than (a).
   The conjunction (near-nominal coverage *and* reduced variance) is the real property — neither alone can be faked.

### Reusable helper — `pathwise_moments`
Add to `src/eval.jl`:
```
pathwise_moments(ens) -> (μs, Σs)
# ens :: N×d×T  (samples × dim × timestep)
# μs[k] = vec(mean(ens[:,:,k]; dims=1)); Σs[k] = cov(ens[:,:,k])
```
This converts a Pathwise ensemble into the `(μs, Σs)` `coverage` consumes. It is currently duplicated inline in 4 examples (LV/VdP/FHN/scale-forcing); the helper DRYs them and is a genuine public ergonomic add. Export it (eval helpers are already exported).

### Units / boundaries
- `pathwise_moments` is pure, no solver, no AD. One responsibility: ensemble array → per-step moments.
- The oracle test lives in `test/test_gpude_svgp.jl` (or a new `test/test_gpude_calibration.jl`), gated under the existing `MAGPIE_TEST_SCIML` flag (it trains).

---

## Section 3 — PULL covariance PSD repair

### Problem
`pull_propagate` (ext:436-439) clamps only **diagonal** entries to `eps()` when a variance goes negative; for d>1 this leaves Σ symmetric-but-indefinite. `coverage`'s `isposdef(Σ) || continue` guard then silently drops those steps from *both* numerator and denominator, distorting the metric. `eps()` is also a machine-epsilon floor unrelated to the field's variance scale. The per-step `@warn` (ext:438) spams the hot loop.

### Fix
- Replace the diagonal clamp with a **PSD-cone projection**: symmetric eigendecomposition, clamp eigenvalues at a **relative** floor (`~1e-12 · tr(Σ)/d`, or 0), reconstruct. Guarantees every Σ is a valid covariance, so `coverage` never silently drops a step.
- **AD-unconstrained** (explorer-confirmed): `pull_propagate`/`propagate` are forward-only, never differentiated (the eval module imports no AD; no caller wraps `propagate` in a gradient). `eigen` is safe here.
- Collapse the per-step `@warn` into a single aggregated count emitted once at the end (e.g. "PULL: projected N steps onto PSD cone"), so systematic failure is visible without hot-loop spam.

### Buffer default
Default `buffer` to **full history** (`typemax(Int)` / no truncation). Save points are O(20–100); the O(n²d²) telescope cost is trivial, and full history removes the silent downward-bias foot-gun. Keep `buffer` as an opt-in cap that warns **once** when it actually truncates. (Note: the current default-21-point `ts` is safely inside `buffer=20`; the risk is only at higher save resolution — full history closes it regardless.)

### Units / boundaries
- A small pure `project_psd(Σ; floor)` function (testable in isolation: feed an indefinite matrix, assert PSD out, assert it's the nearest PSD in Frobenius norm).

---

## Section 4 — Per-dimension `σ_obs`

### Problem
`_gaussian_nll` (ext:132) uses one scalar `σ²_obs = exp(2·logσ_obs)` pooling SSE across all output dimensions and trajectories. For multi-output systems with genuinely different per-component noise scales (the common real case), this misweights the fit and yields a pooled-RMS noise, not per-dimension noise.

### Fix — widen the hyper-prefix
Change the trained-vector hyper-prefix from `[logℓ, logσ, logσ_obs]` to `[logℓ, logσ, logσ_obs(1..d)]`, replacing `const NHYP = 3` with `nhyp(field) = 2 + outputdim(field)`, where `outputdim(::ExactGPField)=f.d`, `outputdim(::SVGPField)=f.dout`, and `CompositeField` forwards via `getproperty`.

The data term becomes per-output:
`Σ_j [ SSE_j/(2σ²_obs,j) + (N_j/2)·log(2π σ²_obs,j) ]`, accumulating per-row `SSE_j = sum(abs2, A[j,:] .- X[j,:])` via `sum(abs2, A .- X; dims=2)`.

### Ordered change-list (explorer-audited; the layout chokepoint absorbs most of it)
1. `src/gpude.jl:75` — `const NHYP = 3` → `outputdim` methods + `nhyp(f) = 2 + outputdim(f)`.
2. `src/gpude.jl:152` (`hyp`) — `logσ_obs = v[3]` → block `v[3:(2+L.d)]`. **[landmine L2]**
3. `src/gpude.jl:154` (`wmat`) — w-block offset `NHYP+1` → `2+L.d+1` (must move together with #2 or weights silently scramble — landmine L3).
4. `src/gpude.jl:163-167` (`unpack(::ExactGPField)`) — propagate vector `logσ_obs`.
5. `src/gpude.jl:245,248,252` (`svgp_Z`/`svgp_μ`/`svgp_Lsblk`) — `NHYP` → `2+f.dout`.
6. `src/gpude.jl:266` (`unpack(::SVGPField)`) — `v[3]` → `v[3:(2+f.dout)]`. **[landmine L2]**
7. `src/gpude.jl:132-139` (`ExactGPField` ctor) — `v0` emits `fill(logσ_obs0, d)`; accept a `d`-vector kwarg.
8. `src/gpude.jl:223-233` (`SVGPField` ctor) — `v0` emits `fill(logσ_obs0, dout)`.
9. `ext:132` (`_gaussian_nll`) — accept per-output `sse::Vector`, `Nd::Vector`, `logσ_obs::Vector`; return the summed per-output NLL. **Make the vectorized contract explicit** (landmine L4: a scalar method silently mis-broadcasting a vector `logσ_obs`).
10. `ext:143-165` (SingleShooting `shooting_data_term`) — per-row `sse` (`d`-vector) and `Nd`.
11. `ext:176-201` (MultipleShooting `shooting_data_term`) — same per-row decomposition.
12. `ext:44-49` (`field_loss`) — `logσ_obs = v[NHYP]` → block `v[3:(2+outputdim(field))]`. **[landmine L1]**
13. `ext:225-241` (MS `build_loss`) — `off = NHYP+nwL` → `2+field.d+nwL`; `logσ_obs` → block. **[landmine L1]**
14. Verify-only: `field_rhs`/`posterior` (ExactGPField) use only logℓ/logσ/w, not `logσ_obs`; offsets shift correctly once #3 lands.
15. Update layout-comment prose throughout.

**Landmines (silently wrong, no error):** the four scalar `v[3]`/`v[NHYP]` σ_obs reads (would use output-1's noise for all dims); the `wmat`/`NHYP` width-drift (scrambled weights via a still-valid reshape); the scalar `_gaussian_nll` leaf (non-scalar loss or silent mis-weight). The variable-length blocks (w/Z/μ/L_S/s0) are all `NHYP`-routed and shift safely once the chokepoint moves.

### Composition with §1
`uvar(u)` already returns a per-output vector, and the trace term divides per output by `σ²_obs,j` — §1 and §4 compose naturally.

### Units / boundaries
- `outputdim` / `nhyp` are the single layout chokepoint. The `pf` solve-vector layout is **unaffected** (`σ_obs` is never in `pf`).

---

## Section 5 — Divergence-guard honesty

### Problem
The guard `(size(A)==size(X) && all(isfinite, A)) || return 1e6` (ext:160/194) runs in the loss primal *after* the solve. It catches only a **forward** NaN/wrong-shape. The failure the surrounding comments actually worry about — Mooncake's Cholesky-solve **backward** throwing `SingularException` — propagates out of `DI.gradient` and never reaches this guard. The relative jitter (`exp(lognoise + 2logσ)`) is the **sole** backward protection. No test exercises the guard.

### Fix
- **Document** (comment + relevant docstrings) that the guard protects only the forward pass and that the relative jitter is the only backward guard — so jitter cannot later be weakened on the false assumption the guard covers it.
- **Add a test** that forces the guard to fire (params that drive the ODE to a non-finite state) and asserts the loss returns the finite sentinel rather than `NaN`/throwing.
- Defer the smooth-barrier idea (slope back toward feasibility) — ADAM warm-up already avoids divergence in practice; revisit only if a real training run stalls on the plateau.

---

## Section 6 — Test hardening

All cheap, all directly raise trust. Gated under `MAGPIE_TEST_SCIML` where they train.

1. **`train!`→MultipleShooting end-to-end test.** Today MS is exercised only at the `build_loss` level (test_gpude_stage2.jl); the `train!`→MS product path is unverified (and `train!` currently MethodErrors for SVGP+MS / mis-builds for Composite+MS — those *fixes* are Spec 2, but the ExactGPField+MS `train!` path should be tested now).
2. **Divergence-guard trigger test** (from §5).
3. **Tighten `σ_obs` recovery band.** Currently `0.5·σtrue < σ_obs < 2·σtrue` (test_gpude_noise.jl, both exact and SVGP). The pure NLL-normalizer oracle already recovers `½·log(SSE/Nd)` to `atol=1e-3`; tighten the through-solver band to a derived, much-narrower tolerance (empirically set, target ≪ 2×).
4. **Replace the tautological `coverage` "perfect→1.0" test** with a level-dependent assertion (or delete it in favor of the existing `N(0,I)` calibration test + §2's oracle).
5. **SVGP/SparseGP PULL quantitative oracle.** Currently a finiteness smoke test only (test_gpude_pull.jl:168-190). Add a real coverage/calibration assertion (reuses §2's `pathwise_moments` + the differential design).

---

## Testing strategy (summary)

| Property | Test | Fails if |
|---|---|---|
| SVGP variance is data-trained | §2 differential oracle (calibrated + sharp) | `L_S` stays at prior (trace term missing/broken) |
| Pathwise→moments conversion | `pathwise_moments` unit test | wrong reshape/axis |
| PULL Σ is always PSD | `project_psd` unit + PULL integration | indefinite Σ leaks to `coverage` |
| Per-dim `σ_obs` recovers heterogeneous noise | new multi-output recovery test (distinct per-dim noise) | pooled scalar misweights |
| Divergence guard fires | §5 trigger test | guard returns `NaN`/throws |
| `train!`→MS works | §6.1 e2e test | MS path regresses |
| `σ_obs` recovered tightly | §6.3 tightened band | bias slips through |

AD soundness (Mooncake-vs-FD gradient gates) for the new trace term: extend the existing `test_gpude_svgp_mo.jl` MO-ELBO gradient check to cover the trace-corrected loss (`relerr < 1e-3`).

## Future work (documented, deferred)

- **State-space SVGP ELBO** — propagate field variance through the solver sensitivity into predicted state covariance and place the likelihood there (couples training to a PULL-style propagation; AD-fragile). The follow-up to §1's field-space term.
- **Smooth divergence barrier** (§5 option b).
- **PULL for CompositeField** — currently errors; would need the combined `∂(known+GP_mean)` Jacobian.

## Risks / open questions

- **Trace-term tolerance (§2).** The "near-nominal coverage" band and the "materially sharper" threshold must be set empirically on the linear-field oracle — risk that a too-loose band re-creates a soft test. Mitigation: assert the *conjunction* (calibrated + sharp) and the *contrast* vs regularized-only, not an absolute band alone.
- **`σ_obs` band tightening (§6.3).** Through-solver recovery is noisier than the pure normalizer; the tightened band must be derived from a real run, not guessed.
- **Per-dim `σ_obs` layout change (§4)** touches the most code; the four landmines are silent-failure-mode, so each change site needs a paired assertion that the prefix width is read correctly.
