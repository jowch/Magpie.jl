# GP-UDE Ergonomic API — Design (Spec 2)

**Status:** Approved design, ready for implementation plan.
**Predecessor:** Spec 1 (`2026-06-22-gpude-correctness-foundation-design.md`) — the correctness
foundation (calibrated SVGP ELBO, PULL PSD projection, per-dim σ_obs). Spec 2 sits on top of it
and changes no numerics; it makes the now-correct Capability B *usable*.

## Goal

Make the GP-in-UDE (Capability B) public API as clean and tight as Capability A: one obvious way to
fit, reconstruct, and propagate a GP-UDE; a public surface that exposes intent (types + verbs +
strategies) and hides the loss/layout/math machinery; and every field × shooting combination either
works or fails with a typed, actionable error — no opaque `MethodError`s.

This is an **ergonomics + surface** round. It is behaviour-preserving for every numeric path that
exists today, except (1) one new capability — Composite(Exact inner) + MultipleShooting — and (2)
replacing today's opaque crashes with clear errors.

## Context: what exists today

Spec 1 left Capability B correct but sprawling. The current public surface (`src/Magpie.jl:40–45`)
exports ~23 Cap-B names, of which roughly half are loss/layout/SVGP-math internals. The fit workflow
has three sharp edges:

1. **`train!` dual contract.** `train!(field, data) -> (field, vopt)` *and* mutates `field.v0`.
   Callers thread `vopt` around even though `posterior(field)` / `propagate(field, …)` already read
   `field.v0`. The second return value is redundant baggage (and for MultipleShooting it carries the
   internal `s0` segment nodes a user never needs).
2. **Triple naming.** `posterior`, `posterior_gps`, `posterior_sparsegps` are three exported names for
   one concept, with not-loaded stubs (`src/gpude.jl:89–97`) and ext aliases.
3. **Incomplete shooting matrix.** `MultipleShooting` works only for `ExactGPField`. SVGP + MS throws a
   raw `MethodError` (no `_train_loss(::SVGPField, ::MultipleShooting)`); Composite + MS throws a
   missing-keyword error (`field_loss` never slices `s0`). The segmentation engine itself
   (`shooting_data_term(field, ::MultipleShooting, …)`) is already field-agnostic — only the wiring
   is missing.

Two things the earlier survey flagged as warts turned out to **already be clean** and need only a
guarding test, not a code change:

- `propagate(cf::CompositeField, …)` already reads `cf.known` internally (`ext:723`) — `known` is
  *not* passed twice.
- PULL on a CompositeField already raises a clear, explanatory error (`ext:718`); only Pathwise is
  supported there, by design (combined known+GP Jacobian for moment-matching is unimplemented).

## Decisions (locked)

| # | Decision |
|---|----------|
| A | `train!` returns just the mutated `field`. Drop the tuple; `vopt` threading disappears. |
| B | Keep only `posterior`. Hard-remove `posterior_gps` / `posterior_sparsegps` (exports, stubs, aliases). |
| C | Hard-trim the Cap-B export surface to types + verbs + strategies + `kmeans_anchors` + metrics. Internals lose only their `export` line (still callable as `Magpie.foo`). |
| D | CompositeField works with any inner field for its supported ops, else fails with a typed error. Lock the matrix with tests. |
| E | Enable **Composite(Exact inner) + MultipleShooting** by generalizing the loss to slice `s0` at `length(field.v0)`. **SVGP + MS** and **Composite(SVGP inner) + MS** → typed fail-clear error. |
| F | Fold in the Spec-1 deferred-Minor hygiene backlog. |
| — | Back-compat: **hard remove** (pre-1.0, no external users). No `@deprecate` shims. |
| — | `kmeans_anchors` **stays public** (standard-workflow input helper, parallels Cap-A's `grid_points`); decoupled sampler internals are un-exported. |

### Tracked follow-up (NOT in this spec)

**SVGP + MultipleShooting.** Deferred deliberately, but *tracked to revisit before the branch closes*
(user intent). It is not a simple wiring change: SVGP trains on the collapsed ELBO
(`svgp_elbo_loss`), a separate path from `shooting_data_term`. Segmenting it requires designing a
per-segment ELBO (where the trace correction lands per segment, one KL vs. per-segment, continuity)
and re-verifying both Mooncake AD and that the Spec-1 calibration win survives. Recorded in the
Spec-2 backlog as a named follow-up, not dropped.

## Design

### A. `train!` returns the field

`ext/MagpieSciMLExt.jl:330–350`. Change the final line from `return field, sol.u` to `return field`.
`field.v0 .= sol.u[1:length(field.v0)]` already stores the trained field prefix, so every downstream
reader (`posterior(field)`, `propagate(field, …)`) is unaffected. Update all call sites
(examples + tests) from `field, vopt = train!(…)` to `train!(field, …)` followed by `posterior(field)`
/ `propagate(field, …)`.

**Mutate-and-return contract, documented:** `train!` mutates `field.v0` in place and returns the same
`field` for chaining. There is no separate "trained vector" to carry. The docstring states this
explicitly.

### B. Single `posterior`

Remove `posterior_gps` and `posterior_sparsegps` entirely: the export line (`src/Magpie.jl:41`), the
not-loaded stubs and forward-declares (`src/gpude.jl:89–97`), and the ext alias bodies
(`ext:378–379`, `ext:431–432`). `posterior(field)` and `posterior(field, v)` remain the single
generic. Update call sites in examples/tests.

### C. Export hard-trim

New Capability-B export block (replaces `src/Magpie.jl:40–45`, lines 35–39 Cap-A untouched):

```julia
export GPField, ExactGPField, SVGPField, CompositeField, SparseGP, train!, posterior, propagate
export SingleShooting, MultipleShooting, PULL, Pathwise, kmeans_anchors
export coverage, field_error, recovery_metrics, ridge_slice, pathwise_moments
```

**Un-exported** (still defined and callable as `Magpie.foo`; only the `export` line is removed):
`FieldLayout`, `gpfield`, `solve_alpha`, `unpack`, `regularizer`, `svgp_kl`, `nLS`, `unpack_LS`,
`L_ZZ_factor`, `svgp_moments`, `DecoupledGPSample`, `build_decoupled_sample`.

**Hard-removed:** `posterior_gps`, `posterior_sparsegps` (see B).

(`svgp_var` is already internal — `Magpie.svgp_var`, never exported — so it needs no change.)

Any test that references an un-exported name must qualify it (`Magpie.unpack`, …). This is the
mechanical cost of the trim and is expected.

### D. CompositeField support matrix

CompositeField delegates `field_rhs` / `posterior` / `propagate` to its inner `gp`, so it composes
with any inner field for the ops that inner field supports. Lock the contract with a matrix test
asserting, for inner ∈ {ExactGPField, SVGPField}:

| op | Exact inner | SVGP inner |
|----|-------------|------------|
| `train!` + SingleShooting | works | works |
| `train!` + MultipleShooting | works (E) | typed error (E) |
| `posterior` | works | works |
| `propagate` Pathwise | works | works |
| `propagate` PULL | typed error (already) | typed error (already) |

No new code is needed for the "works/already-errors" cells beyond E; this section is a guarding
test that pins the behaviour so a future refactor can't silently break composition.

### E. Composite(Exact) + MultipleShooting; clear errors elsewhere

**Enable Composite(Exact inner) + MS.** The MS data term `shooting_data_term(field, ::MultipleShooting,
(pf, rhs!), data; s0, …)` is field-agnostic; the only Exact-specific piece is `build_loss`'s manual
`s0` slice (`ext:261–278`), which computes the offset as `2 + field.d + nwL` — i.e. exactly
`length(field.v0)`. Generalize the MS loss path so the `s0` block is sliced at `off = length(field.v0)`
for **any** field whose MS is supported, and route CompositeField(Exact inner) through it
(CompositeField delegates layout to `cf.gp`, so `length(cf.gp.v0)` is the offset; `_train_init`
already appends `s0` after the field prefix via `_init_vec(cf.gp, …)`).

**Fail-clear for the unsupported combos.** Add a typed guard so `train!(field, data;
shooting=MultipleShooting(…))` errors early with an actionable message when `field` is an `SVGPField`,
or a `CompositeField` whose inner `gp` is an `SVGPField`:

```
ArgumentError: MultipleShooting is not supported for SVGPField (here: inside a CompositeField).
Use SingleShooting for SVGP fields. (SVGP + MultipleShooting is a tracked follow-up.)
```

The guard lives at the `train!` / `_train_loss` dispatch boundary so it fires before any solve, and
replaces today's opaque `MethodError` / missing-`s0` crash.

### F. Spec-1 deferred-Minors polish

Fold the recorded backlog (`.superpowers/sdd/progress.md`) into this round:

- Add a direct `uvar(u) ≈ Magpie.svgp_var(prior, Z, L_ZZ, L_S, u)` equality assertion (closes a
  coverage gap; through-solver value already covered by the FD gate).
- Assert dout>1 σ_obs-slot gradients are live (per-dim σ_obs regression guard).
- Re-seed the RNG (`Random.seed!(…)`) before the SparseGP PULL `train!` to remove the latent
  cross-test RNG-state dependence; then tighten the SparseGP coverage band from `0.5 ≤ cov90 ≤ 1.0`.
- Add a comment at the SVGP jitter site noting the SE-constant-diagonal assumption tying
  `L_ZZ_factor`'s mean-diagonal jitter to the ext's exact-σ² jitter (would diverge for a
  non-stationary kernel).
- Drop the unused `ext = Base.get_extension(…)` binding and dead imports in
  `test/test_gpude_calibration.jl`.
- Silence the informational `dt_NaN` solver warning in the divergence-guard test via `@test_logs` (or
  a scoped logger), so CI output stays clean.

## Testing strategy

- **A (train! contract):** existing end-to-end tests already assert fit quality; update them to the
  one-value return and add one assertion that `train!(field, …) === field` and that `posterior(field)`
  after `train!` equals `posterior(field, field.v0)`.
- **B (naming):** grep guard — no `posterior_gps` / `posterior_sparsegps` remain; existing posterior
  tests pass through the single name.
- **C (exports):** a test asserting the un-exported names are *not* in `names(Magpie)` but *are*
  reachable as `Magpie.foo`; the public Cap-B names *are* exported.
- **D (matrix):** the support-matrix test above (works / typed-error per cell).
- **E (Composite+MS):** an end-to-end exemplar fitting a CompositeField(known, ExactGPField) with
  `MultipleShooting`, asserting recovery comparable to SingleShooting; plus `@test_throws ArgumentError`
  for SVGP+MS and Composite(SVGP)+MS.
- **F:** the individual assertions listed above.

Full suite (currently 221/221) must stay green. The AD/Mooncake tests are slow (~2 min suite); the
Composite+MS exemplar adds one through-solver fit.

## Non-goals

- SVGP + MultipleShooting (tracked follow-up, see above).
- Any change to the trained numerics: ELBO, trace correction, PULL/Pathwise math, per-dim σ_obs.
- A new high-level `fit_ude` factory — the cleaned `train!` + `posterior`/`propagate` *is* the entry
  point; a factory would be added surface for no current need (revisit if a one-call API is requested).
- Multi-class / additional likelihoods, inducing-point learning changes.

## Risk & ordering

Low risk overall; the only new numeric path is Composite(Exact)+MS, which reuses the verified
field-agnostic MS engine. Suggested task order: **C+B+A first** (mechanical surface changes touching
many call sites — do them together so the suite is updated once), then **E** (Composite+MS + guards),
then **D** (matrix tests over the now-final behaviour), then **F** (polish). The export trim (C) is the
highest-churn step; doing it before E/D means the new tests are written against the final names.
