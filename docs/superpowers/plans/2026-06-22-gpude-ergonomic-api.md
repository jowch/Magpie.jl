# GP-UDE Ergonomic API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Capability B's public API as tight and obvious as Capability A — one fit verb, one posterior name, a trimmed export surface, and every field × shooting combination either works or fails with a typed error.

**Architecture:** Six mechanical/low-risk tasks over `src/Magpie.jl`, `src/gpude.jl`, `ext/MagpieSciMLExt.jl`, the examples, and the test suite. No trained numerics change. The one new capability — `CompositeField(ExactGPField inner) + MultipleShooting` — reuses the already-field-agnostic segmentation engine (`shooting_data_term`); only loss-wiring is added.

**Tech Stack:** Julia ≥1.12, AbstractGPs.jl, KernelFunctions.jl, Optimization.jl (ADAM→LBFGS), Mooncake via DifferentiationInterface, OrdinaryDiffEq + SciMLSensitivity (the `MagpieSciMLExt` extension), Runic.jl formatting.

**Spec:** `docs/superpowers/specs/2026-06-22-gpude-ergonomic-api-design.md`.

## Global Constraints

- **AD is Mooncake-first** via DifferentiationInterface; keep factorizations dense; all Cholesky goes through the internal `_chol` chokepoint. (No change here, but new loss code must stay Mooncake-clean.)
- **R1:** never closure-capture `α`/`pf` into an ODE RHS; recompute in-loss and thread through `pf`. **R2:** `Array(sol)` before indexing a solution.
- **Pre-1.0, hard remove** removed names — no `@deprecate` shims.
- **`kmeans_anchors` stays exported**; the decoupled sampler (`DecoupledGPSample`, `build_decoupled_sample`) becomes internal.
- **Behaviour-preserving** for all existing numeric paths; the only new numeric path is `CompositeField(Exact)+MultipleShooting`, and the only behaviour *changes* are opaque crashes → typed errors.
- **Runic format** every changed `.jl` file before each commit: `git ls-files -z -- '*.jl' | xargs -0 runic --inplace`.
- Full suite (currently 221/221) stays green. Run a single file with `julia --project=. test/<file>.jl`; full suite `MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'` (~2 min).

## Final public Capability-B export surface (target of Task 3)

```julia
export GPField, ExactGPField, SVGPField, CompositeField, SparseGP, train!, posterior, propagate
export SingleShooting, MultipleShooting, PULL, Pathwise, kmeans_anchors
export coverage, field_error, recovery_metrics, ridge_slice, pathwise_moments
```

Un-exported (still defined, callable as `Magpie.foo`): `FieldLayout`, `gpfield`, `solve_alpha`, `unpack`, `regularizer`, `svgp_kl`, `nLS`, `unpack_LS`, `L_ZZ_factor`, `svgp_moments`, `DecoupledGPSample`, `build_decoupled_sample`. Removed entirely: `posterior_gps`, `posterior_sparsegps`.

---

### Task 1: `train!` returns the field

**Files:**
- Modify: `ext/MagpieSciMLExt.jl:349` (the `return` line)
- Modify (call sites): `examples/gp_ude_lotka_volterra.jl:69`, `examples/gp_ude_vanderpol.jl:78`, `examples/gp_ude_scale_forcing.jl:100`, `examples/gp_ude_identifiability.jl:61,116`, `examples/gp_ude_fitzhugh_nagumo.jl:82`, `test/test_gpude_protocol.jl:153`, `test/test_gpude_noise.jl:26,47,77`, `test/test_gpude_stage1.jl:13`, `test/test_gpude_stage2.jl:81`, `test/test_exemplar_B_grad.jl:39`
- Test: `test/test_gpude_protocol.jl` (add the contract assertion)

**Interfaces:**
- Produces: `train!(field, data; kw...) -> field` (the same, mutated `field`). The trained vector is `field.v0` (for `CompositeField`, forwarded to `cf.gp.v0`). There is no second return value.

**Conversion rule for call sites:** replace `field, v = train!(args)` with `train!(args)`, then replace every later use of the bound vector `v`/`vopt`/`vfit` in that scope:
- `posterior_gps(field, v)` → `posterior_gps(field)` (1-arg form reconstructs from `field.v0`; the rename to `posterior` happens in Task 2)
- `posterior_sparsegps(field, v)` → `posterior_sparsegps(field)`
- `unpack(field, v)` → `unpack(field, field.v0)`
- any other `…(…, v)` → `…(…, field.v0)`

Lines that already ignore the return (`train!(field, …)` with no binding, e.g. `test/test_gpude_pull.jl:197`, `test/test_gpude_calibration.jl:22-23`) need **no change**.

- [ ] **Step 1: Write the failing contract test**

Add to `test/test_gpude_protocol.jl` (inside the file's top-level testset, after the existing CompositeField block):

```julia
@testset "train! returns the mutated field (single return value)" begin
    rng = MersenneTwister(4)
    u0 = [1.0]; tspan = (0.0, 2.0); ts = collect(range(tspan...; length = 10))
    X = Array(solve(ODEProblem((du, u, p, t) -> (du[1] = -0.5u[1]), u0, tspan), Tsit5(); saveat = ts))
    Z = [[x] for x in range(0.2, 1.0; length = 4)]
    field = ExactGPField(SqExponentialKernel(), Z; d = 1)
    ret = train!(field, (ts, X); adam_iters = 20, maxiters = 10)
    @test ret === field                                   # returns the same field, not a tuple
    # field.v0 holds the trained vector: the 1-arg posterior (reads field.v0) matches the explicit form.
    @test predmean(posterior(field)[1], [0.5]) ≈ predmean(posterior(field, field.v0)[1], [0.5])
end
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `julia --project=. test/test_gpude_protocol.jl`
Expected: FAIL — `train!` currently returns `(field, sol.u)`, so `ret === field` is false (and `posterior` is still the old name, but the tuple mismatch fails first).

- [ ] **Step 3: Change the return value**

`ext/MagpieSciMLExt.jl:349`: replace

```julia
    return field, sol.u
```

with

```julia
    return field
```

- [ ] **Step 4: Convert every tuple-unpack call site**

Apply the conversion rule above. Representative edits:

`examples/gp_ude_lotka_volterra.jl:69,73`:
```julia
train!(field, (ts, Xnoisy); tspan, maxiters = 150, λ = 1 / (15 * 2), s = 0.5)
# ...
gps = posterior_gps(field)
```

`test/test_gpude_noise.jl:47,51` (the dout=2 block):
```julia
train!(
    field, (ts, Xnoisy); tspan, adam_iters = 800, maxiters = 200,
    λ = 1 / (15 * 2), s = 0.5
)
lo_vec = unpack(field, field.v0).logσ_obs
```

`test/test_gpude_stage2.jl:81`:
```julia
train!(field, (ts, X); shooting = MultipleShooting(nsegments = 4), adam_iters = 300, maxiters = 80)
```

`test/test_exemplar_B_grad.jl:39,43`:
```julia
Magpie.train!(field, (ts, target); tspan, adam_iters = 30, maxiters = 10)
# ...
gps = Magpie.posterior_gps(field)
```

Grep each listed file for the bound name after editing to confirm no dangling `vopt`/`vfit`/`v`/`vfit` reference to the removed binding remains:
`grep -nE '\b(vopt|vfit)\b' examples/*.jl test/*.jl`

- [ ] **Step 5: Update the `train!` docstring**

In `ext/MagpieSciMLExt.jl`, in the comment block above `function Magpie.train!` (around line 328), state the contract: "Mutates `field.v0` in place (for `CompositeField`, the inner `gp.v0`) and returns the same `field` for chaining. The trained parameters live in `field.v0`; there is no separate return vector."

- [ ] **Step 6: Format and run the suite**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: 222/222 (the new contract test adds one). All green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(gpude): train! returns the mutated field (drop redundant vopt)"
```

---

### Task 2: Single `posterior` name

**Files:**
- Modify: `src/gpude.jl:89-97` (remove the two forward-declares + two not-loaded stubs)
- Modify: `src/Magpie.jl:41` (drop `posterior_gps, posterior_sparsegps` from the export)
- Modify: `ext/MagpieSciMLExt.jl:378-379,431-432` (remove the four alias bodies), `:649,683` (internal uses → `posterior`)
- Modify (call sites): `examples/gp_ude_lotka_volterra.jl:73`, `examples/gp_ude_vanderpol.jl:82`, `examples/gp_ude_scale_forcing.jl:116`, `examples/gp_ude_identifiability.jl:118`, `test/test_exemplar_B_grad.jl:31,43`, `test/test_gpude_stage1.jl:5,17`, `test/test_gpude_stage2.jl:83`

**Interfaces:**
- Produces: `posterior(field)` and `posterior(field, v)` as the single generic for every `GPField`. `posterior_gps` / `posterior_sparsegps` no longer exist.

- [ ] **Step 1: Write the failing guard test**

Add to `test/test_gpude_protocol.jl` (after the Task 1 testset):

```julia
@testset "posterior is the only reconstruction name" begin
    @test !isdefined(Magpie, :posterior_gps)
    @test !isdefined(Magpie, :posterior_sparsegps)
    @test :posterior in names(Magpie)
end
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `julia --project=. test/test_gpude_protocol.jl`
Expected: FAIL — `posterior_gps`/`posterior_sparsegps` are still defined.

- [ ] **Step 3: Remove the core declares and stubs**

`src/gpude.jl`: delete the two lines

```julia
function posterior_gps end
function posterior_sparsegps end
```

and the two stub lines

```julia
posterior_gps(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
posterior_sparsegps(args...; kw...) = error("MagpieSciMLExt not loaded. Add `using OrdinaryDiffEq, SciMLSensitivity`.")
```

Update the adjacent comment (gpude.jl:91-92) to drop the alias mention: `# \`posterior\` is the canonical solver-free reconstruction generic (body in the ext).`

- [ ] **Step 4: Remove the export and the ext aliases**

`src/Magpie.jl:41`: change to
```julia
export unpack, regularizer, posterior
```
(`unpack`/`regularizer` are removed from the export in Task 3, not here.)

`ext/MagpieSciMLExt.jl`: delete the four alias lines (378-379, 431-432):
```julia
Magpie.posterior_gps(field::ExactGPField) = Magpie.posterior(field)
Magpie.posterior_gps(field::ExactGPField, v) = Magpie.posterior(field, v)
# ...
Magpie.posterior_sparsegps(field::SVGPField) = Magpie.posterior(field)
Magpie.posterior_sparsegps(field::SVGPField, v) = Magpie.posterior(field, v)
```
and their preceding `# Public aliases …` comment lines.

`ext/MagpieSciMLExt.jl:649`: `Magpie.propagate(Magpie.posterior_gps(field), u0, tspan; kw...)` → `Magpie.propagate(Magpie.posterior(field), u0, tspan; kw...)`
`ext/MagpieSciMLExt.jl:683`: `Magpie.posterior_sparsegps(field)` → `Magpie.posterior(field)`

- [ ] **Step 5: Convert call sites to `posterior`**

`posterior_gps(field)` / `posterior_gps(field, …)` → `posterior(field)`; `posterior_sparsegps(field)` / `posterior_sparsegps(field, …)` → `posterior(field)`. Examples:
- `examples/gp_ude_lotka_volterra.jl:73`: `gps = posterior(field)`
- `examples/gp_ude_scale_forcing.jl:116`: `sgps = posterior(field)`
- `test/test_gpude_stage1.jl:17`: `gps = Magpie.posterior(field)`
- `test/test_gpude_stage2.jl:83`: `gps = posterior(field)`
- `test/test_exemplar_B_grad.jl:43`: `gps = Magpie.posterior(field)`

Update the two testset *names* that mention `posterior_gps` (`test_gpude_stage1.jl:5`, `test_exemplar_B_grad.jl:31`) to say `posterior`.

Confirm none remain: `grep -rn 'posterior_gps\|posterior_sparsegps' src ext examples test` → only matches should be inside this plan/spec, none in code.

- [ ] **Step 6: Format, run suite, commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'
git add -A
git commit -m "refactor(gpude): single posterior name (remove posterior_gps/posterior_sparsegps)"
```
Expected: green (223 tests).

---

### Task 3: Export hard-trim

**Files:**
- Modify: `src/Magpie.jl:40-45` (Cap-B export block)
- Test: `test/test_gpude_protocol.jl` (export-surface assertion)

**Interfaces:**
- Produces: the final export surface listed at the top of this plan. Internals remain callable as `Magpie.foo`; tests that import them via `using Magpie: name` are unaffected (explicit import resolves regardless of export).

- [ ] **Step 1: Write the failing export-surface test**

Add to `test/test_gpude_protocol.jl`:

```julia
@testset "Capability-B export surface is trimmed" begin
    public = (:GPField, :ExactGPField, :SVGPField, :CompositeField, :SparseGP,
              :train!, :posterior, :propagate, :SingleShooting, :MultipleShooting,
              :PULL, :Pathwise, :kmeans_anchors)
    internal = (:FieldLayout, :gpfield, :solve_alpha, :unpack, :regularizer,
                :svgp_kl, :nLS, :unpack_LS, :L_ZZ_factor, :svgp_moments,
                :DecoupledGPSample, :build_decoupled_sample)
    exported = Set(names(Magpie))
    for n in public
        @test n in exported
    end
    for n in internal
        @test !(n in exported)          # not exported …
        @test isdefined(Magpie, n)      # … but still defined/callable as Magpie.n
    end
end
```

- [ ] **Step 2: Run it — expect FAIL**

Run: `julia --project=. test/test_gpude_protocol.jl`
Expected: FAIL — the internal names are still exported.

- [ ] **Step 3: Rewrite the Cap-B export block**

`src/Magpie.jl`: replace the current lines 40-45 (the block starting `export GPField, CompositeField, …` through `export coverage, …`) with:

```julia
export GPField, ExactGPField, SVGPField, CompositeField, SparseGP, train!, posterior, propagate
export SingleShooting, MultipleShooting, PULL, Pathwise, kmeans_anchors
export coverage, field_error, recovery_metrics, ridge_slice, pathwise_moments
```

(Capability-A export lines 35-39 are untouched.)

- [ ] **Step 4: Run suite to confirm internals still resolve**

```bash
MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: green. The `using Magpie: FieldLayout, gpfield, …` imports in `test_gpude_unit.jl`, `test_gpude_svgp.jl`, `test_gpude_protocol.jl`, `test_gpude_stage2.jl`, `test_gpude_svgp_mo.jl`, `test_gpude_noise.jl` continue to resolve (explicit import is export-independent). If any test used a now-internal name via bare `using Magpie` without explicit import, qualify it as `Magpie.name` — but the audit found none.

- [ ] **Step 5: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add -A
git commit -m "refactor(gpude): hard-trim Capability-B export surface to match Capability A"
```

---

### Task 4: `CompositeField(Exact) + MultipleShooting`; typed errors for SVGP+MS

**Files:**
- Modify: `ext/MagpieSciMLExt.jl:44-54` (`field_loss` — slice `s0` for MS); add `_ms_kwargs` helper near it; add `_assert_shooting_supported` + call it in `train!` (around line 337)
- Test: `test/test_gpude_stage2.jl` (add the Composite+MS recovery test and the two `@test_throws`)

**Interfaces:**
- Consumes: `shooting_data_term(field, ::MultipleShooting, (pf, rhs!), data; logσ_obs, s0, …)` (field-agnostic, already exists, `ext:209`); `length(field.v0)` (the field-param prefix length; `CompositeField` forwards `.v0` to its inner gp); `_init_vec(field, u_data, t_data, ms)` already appends `vec(s0)` after the prefix (`ext:288`).
- Produces: `train!(cf::CompositeField{<:Any,<:ExactGPField}, data; shooting = MultipleShooting(...))` trains successfully; `train!` with an `SVGPField` (bare or inside a `CompositeField`) + `MultipleShooting` throws `ArgumentError` before solving.

- [ ] **Step 1: Write failing tests**

Add to `test/test_gpude_stage2.jl` (it already imports `ExactGPField, FieldLayout, MultipleShooting`; add `using Magpie: CompositeField, SVGPField, SingleShooting, train!, posterior, predmean, kmeans_anchors` and `using OrdinaryDiffEq` at the top if not present):

```julia
@testset "Composite(Exact) + MultipleShooting trains (E)" begin
    rng = MersenneTwister(11)
    known(u, t) = -0.5 .* u                       # known linear decay
    f!(du, u, p, t) = (du .= known(u, t); du .+= 0.3; nothing)  # truth: residual = +0.3
    u0 = [1.0]; tspan = (0.0, 4.0); ts = collect(range(tspan...; length = 16))
    target = Array(solve(ODEProblem(f!, u0, tspan), Tsit5(); saveat = ts))
    X = target .+ 0.02 .* randn(rng, size(target))
    Z = kmeans_anchors(X, 6; rng = MersenneTwister(3))
    cf = CompositeField(known, ExactGPField(SqExponentialKernel(), Z; d = 1))
    ret = train!(cf, (ts, X); shooting = MultipleShooting(nsegments = 3),
                 adam_iters = 300, maxiters = 80, λ = 1 / 16)
    @test ret === cf                                          # returns the field (Task 1 contract)
    res = predmean(posterior(cf)[1], [0.5])                   # reconstructed residual GP mean
    @test isfinite(res)
    @test abs(res - 0.3) < 0.25                               # recovers the +0.3 residual, not 0
end

@testset "SVGP + MultipleShooting fails clearly (E)" begin
    Zs = [[x] for x in range(-1, 1; length = 5)]
    svgp = SVGPField(SqExponentialKernel(), Zs; dout = 1)
    ts = collect(range(0, 1; length = 8)); X = reshape(collect(range(1, 0.5; length = 8)), 1, 8)
    @test_throws ArgumentError train!(svgp, (ts, X); shooting = MultipleShooting(nsegments = 2))
    cf_svgp = CompositeField((u, t) -> zero(u), SVGPField(SqExponentialKernel(), Zs; dout = 1))
    @test_throws ArgumentError train!(cf_svgp, (ts, X); shooting = MultipleShooting(nsegments = 2))
end
```

- [ ] **Step 2: Run them — expect FAIL**

Run: `julia --project=. test/test_gpude_stage2.jl`
Expected: the Composite+MS test fails with a missing-keyword error (`field_loss` doesn't supply `s0`); the fail-clear test fails because `MethodError` is thrown, not `ArgumentError`.

- [ ] **Step 3: Add the `s0` slice to `field_loss`**

`ext/MagpieSciMLExt.jl`: directly above `function field_loss` (line 44), add:

```julia
# MultipleShooting stores per-segment initial nodes s0 (dout×S) AFTER the field-param prefix in v;
# SingleShooting has none. `length(field.v0)` is the prefix length (CompositeField forwards .v0 to gp).
_ms_kwargs(field, ::Magpie.SingleShooting, v, dout) = NamedTuple()
function _ms_kwargs(field, ms::Magpie.MultipleShooting, v, dout)
    off = length(field.v0)
    s0 = reshape(v[(off + 1):(off + dout * ms.nsegments)], dout, ms.nsegments)
    return (s0 = s0,)
end
```

Then change the `shooting_data_term` call inside `field_loss` (lines 49-52) to splat the s0 kwarg:

```julia
        return shooting_data_term(
            field, shooting, (pf, rhs!), data;
            logσ_obs = v[3:(2 + dout)], uvar = uv, _ms_kwargs(field, shooting, v, dout)..., kw...
        ) + Magpie.regularizer(field, v; kw...)
```

- [ ] **Step 4: Add the fail-clear guard**

`ext/MagpieSciMLExt.jl`: directly above `function Magpie.train!` (line 330), add:

```julia
# MultipleShooting is wired for ExactGPField (and CompositeField with an Exact inner) only.
# SVGP trains on the collapsed ELBO (a separate path) — segmenting it is a tracked follow-up.
_assert_shooting_supported(field, shooting) = nothing
_assert_shooting_supported(::SVGPField, ::Magpie.MultipleShooting) =
    throw(ArgumentError("MultipleShooting is not supported for SVGPField. Use SingleShooting for SVGP fields. (SVGP + MultipleShooting is a tracked follow-up.)"))
function _assert_shooting_supported(cf::Magpie.CompositeField, ms::Magpie.MultipleShooting)
    getfield(cf, :gp) isa SVGPField && throw(ArgumentError("MultipleShooting is not supported for a CompositeField with an SVGPField inner field. Use SingleShooting. (SVGP + MultipleShooting is a tracked follow-up.)"))
    return nothing
end
```

Inside `Magpie.train!`, immediately after `trajs = _as_trajectories(data)` (line 337), add:

```julia
    _assert_shooting_supported(field, shooting)
```

- [ ] **Step 5: Run the stage2 tests — expect PASS**

Run: `julia --project=. test/test_gpude_stage2.jl`
Expected: PASS — Composite+MS recovers the residual; both unsupported combos throw `ArgumentError`. (`ExactGPField + MS` still routes through `build_loss`, unchanged.)

- [ ] **Step 6: Format, run full suite, commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'
git add -A
git commit -m "feat(gpude): CompositeField(Exact)+MultipleShooting; typed error for SVGP+MS"
```

---

### Task 5: CompositeField support matrix (propagate axis)

**Files:**
- Test: `test/test_gpude_protocol.jl` (matrix test for the propagate dimension)

**Interfaces:**
- Consumes: `propagate(cf, u0, tspan; method, ts)` — Pathwise works for any inner field; PULL on a CompositeField raises a clear error (`ext:718`). `posterior(cf)` works for any inner field.

This task adds no production code — it pins the already-correct propagate behaviour so a future refactor can't silently break composition. The `train! × shooting` cells are covered by Task 4.

- [ ] **Step 1: Write the matrix test**

Add to `test/test_gpude_protocol.jl` (uses `propagate, PULL, Pathwise, posterior` — all exported; add `using Magpie: SVGPField, CompositeField, kmeans_anchors` to the file's import line if absent):

```julia
@testset "CompositeField support matrix (propagate axis)" begin
    rng = MersenneTwister(7)
    known(u, t) = -0.3 .* u
    u0 = [1.0]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length = 12))
    X = Array(solve(ODEProblem((du, u, p, t) -> (du .= known(u, t); du .+= 0.2), u0, tspan), Tsit5(); saveat = ts))

    # Exact inner: posterior works; Pathwise works; PULL errors clearly.
    Zx = [[x] for x in range(0.3, 1.0; length = 5)]
    cf_ex = CompositeField(known, ExactGPField(SqExponentialKernel(), Zx; d = 1))
    train!(cf_ex, (ts, X); adam_iters = 100, maxiters = 40)
    @test length(posterior(cf_ex)) == 1
    ens = propagate(cf_ex, u0, tspan; method = Pathwise(n = 16), ts = ts)
    @test size(ens, 3) == length(ts)
    @test_throws Exception propagate(cf_ex, u0, tspan; method = PULL(), ts = ts)

    # SVGP inner: train! is unsupported (fails clearly) — gpfield is ExactGPField-only. Tracked follow-up.
    # (Decision update: Composite(SVGP inner) was never trainable — opaque MethodError — so per the
    #  fail-clear+defer decision a shooting-agnostic ArgumentError guard rejects it; this pins that.)
    Zs = kmeans_anchors(X, 5; rng = MersenneTwister(2))
    cf_sv = CompositeField(known, SVGPField(SqExponentialKernel(), Zs; dout = 1))
    @test_throws ArgumentError train!(cf_sv, (ts, X); adam_iters = 10, maxiters = 5)
end
```

- [ ] **Step 2: Run it — expect PASS**

Run: `julia --project=. test/test_gpude_protocol.jl`
Expected: PASS (behaviour already correct). If the PULL cell does not throw, that is a real regression — stop and report, do not weaken the assertion.

- [ ] **Step 3: Format and commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
git add -A
git commit -m "test(gpude): pin CompositeField support matrix (propagate axis)"
```

---

### Task 6: Spec-1 deferred-Minors polish

**Files:**
- Test: `test/test_gpude_noise.jl` (per-dim σ_obs gradient liveness)
- Modify: `test/test_gpude_pull.jl` (RNG determinism + tightened band), `test/test_gpude_guard.jl` (`@test_logs` for `dt_NaN`), `test/test_gpude_calibration.jl` (drop dead bindings)
- Modify: `src/gpude.jl` (SE-constant-diagonal jitter comment at `L_ZZ_factor`)

Note: the backlog's "add a direct `uvar ≈ svgp_var` assertion" is **deliberately dropped** — the ext's `uvar` closure is *constructed from* `svgp_var`, so the assertion would be near-tautological, and the through-solver value is already covered by the FD-vs-Mooncake gate. (YAGNI.)

- [ ] **Step 1: Write the per-dim σ_obs gradient-liveness test (FD, no AD)**

Add to `test/test_gpude_noise.jl` (it imports `SVGPField, unpack`; the test needs the ELBO loss builder via the extension):

```julia
@testset "dout>1 σ_obs slots have live gradients" begin
    rng = MersenneTwister(5)
    Atrue = [-0.3 0.0; 0.0 -0.5]
    u0 = [1.0, 1.0]; tspan = (0.0, 3.0); ts = collect(range(tspan...; length = 20))
    Xc = Array(solve(ODEProblem((du, u, p, t) -> (du .= Atrue * u), u0, tspan), Tsit5(); saveat = ts))
    X = Xc .+ 0.03 .* randn(rng, size(Xc))
    Z = Magpie.kmeans_anchors(X, 6; rng = MersenneTwister(2))
    field = SVGPField(SqExponentialKernel(), Z; dout = 2)
    ext = Base.get_extension(Magpie, :MagpieSciMLExt)
    loss = ext.svgp_elbo_loss(field, [(ts, X)]; tspan = tspan)
    v = copy(field.v0)
    fd(i) = (vp = copy(v); vp[i] += 1.0e-5; vm = copy(v); vm[i] -= 1.0e-5; (loss(vp) - loss(vm)) / 2.0e-5)
    # v = [logℓ, logσ, logσ_obs(1..dout), …] ⇒ σ_obs slots are indices 3 and 4 for dout=2.
    @test abs(fd(3)) > 1.0e-6
    @test abs(fd(4)) > 1.0e-6
end
```

Run: `julia --project=. test/test_gpude_noise.jl` — expect PASS (both σ_obs slots influence the loss). If it FAILS, the per-dim σ_obs data term is not wired per-dimension — stop and report.

- [ ] **Step 2: RNG determinism + tightened band in the SparseGP PULL test**

`test/test_gpude_pull.jl`: at the top of the `@testset` that contains line 197 (the SparseGP PULL block), add a seed before the data/train block so the coverage value is deterministic:

```julia
    Random.seed!(9)
```

Run the file once: `julia --project=. test/test_gpude_pull.jl`, read the `@info "SparseGP PULL coverage" cov90` value `V`. Then tighten the band on the `cov90` assertion from `0.5 ≤ cov90 ≤ 1.0` to a pinned window around the measured value, e.g. `@test (V - 0.05) ≤ cov90 ≤ min(1.0, V + 0.05)` using the literal `V` you observed (round to 2 decimals). Keep `@test all(isposdef, Σs[2:end])`.

- [ ] **Step 3: Silence the `dt_NaN` warning in the guard test**

`test/test_gpude_guard.jl`: wrap the divergence-triggering loss evaluation (the call that emits the `dt_NaN` solver warning) in `@test_logs (:warn,) match_mode=:any` — or, if matching the exact warning is brittle, redirect with `Logging.with_logger(Logging.NullLogger()) do … end` around just that call. Keep the existing `@test L ≥ 1.0e6` assertion. Add `using Logging` if you take the logger route.

- [ ] **Step 4: Drop dead bindings in the calibration test**

`test/test_gpude_calibration.jl`: remove the unused `ext = Base.get_extension(Magpie, :MagpieSciMLExt)` binding if `ext` is not referenced, and remove any `using …` import not used in the file. Confirm with `grep -n 'ext\.' test/test_gpude_calibration.jl` (no matches ⇒ the binding is dead).

- [ ] **Step 5: Add the SE-constant-diagonal jitter comment**

`src/gpude.jl`: at the `L_ZZ_factor` definition (the relative-jitter site), add a comment noting the assumption:

```julia
# NOTE: relative jitter uses the prior's mean diagonal as the scale. This assumes a
# stationary kernel (constant k(z,z), e.g. SE) so the per-row jitter equals the ext's
# exact-σ² jitter. For a non-stationary kernel the two would diverge — revisit then.
```

- [ ] **Step 6: Format, run full suite, commit**

```bash
git ls-files -z -- '*.jl' | xargs -0 runic --inplace
MAGPIE_TEST_SCIML=true julia --project=. -e 'using Pkg; Pkg.test()'
git add -A
git commit -m "test(gpude): per-dim σ_obs gradient liveness + Spec-1 deferred-Minors polish"
```
Expected: full suite green (≈226 tests).

---

## Notes for the executor

- **Order matters:** 1 → 2 → 3 (surface changes; 2 must remove `posterior_gps` *and* its export together or precompile breaks) → 4 (capability) → 5 (matrix, asserts Task 4 behaviour) → 6 (polish).
- **Slow suite:** the AD/Mooncake tests dominate (~2 min). Each task runs the full suite once at its end; per-file runs during TDD are much faster.
- **No numerics change:** if any *existing* test's numeric assertion shifts, that's a regression in your edit, not an expected change — investigate, don't loosen the test.
