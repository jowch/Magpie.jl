# Kernel Parameterization — Design Note

**Date:** 2026-06-22
**Status:** **Decided** — `destructure`-based parameterization over user-built KernelFunctions kernels. Both AD spikes pass. Seeds a future phase (after Phase 1.5).
**Relates to:** [Phase 1 foundation hardening](2026-06-22-al-foundation-hardening-design.md) (Task 4 made `fit` kernel-generic over a whitelist; this is how that generalizes), and the gp-ude branch's field training.

## Problem

`fit` today flattens a kernel to exactly `[logℓ, logσ²]` and rebuilds it as `σ²·with_lengthscale(base, ℓ)`. That model can't express **ARD** (a lengthscale per input dimension), **composite kernels** (sums/products), or **extra shape params** (RationalQuadratic's α). We want extensible hyperparameter fitting — at least ARD and composites — without a parallel kernel DSL and without AD fragility.

The constraint is **not** ForwardDiff-ability; it's the *parameterization model*. The single-`(ℓ,σ²)` reconstruction, and the `_lengthscale`/`_outputscale` peelers it relies on (also used by `LocalPenalization` and `grad_predict`), assume one scalar-lengthscale base kernel.

## Ecosystem findings (KernelFunctions 0.10.67, verified against installed source)

- KernelFunctions exposes **no parameter interface of its own**; it is designed to *"interoperate with generic packages for handling parameters like ParameterHandling.jl and FluxML's Functors.jl."*
- Every kernel/transform is **`@functor`'d** (`ScaleTransform`, `ARDTransform`, `TransformedKernel`, `KernelSum`, `KernelProduct`) — so Functors can walk/rebuild any kernel tree.
- **ARD is first-class:** `with_lengthscale(base, ℓ::AbstractVector) → base ∘ ARDTransform(inv.(ℓ))`.
- A KernelFunctions kernel **holds its parameter values inside the struct** (it is like a Flux model, not a Lux model — params are baked into fields, not held separately).

## Decision: `destructure` the user-built kernel

The user builds the kernel they want with **ordinary KernelFunctions composition** — the full kernel zoo, ARD, sums, products. `fit` then treats that kernel `k` as a **structural template**: it `destructure`s `k` (Functors) into a flat parameter vector + a `rebuild` closure, optimizes the (log-transformed) vector, and `rebuild`s the kernel. We **use KernelFunctions directly and override the baked-in param values** via the round-trip; we do **not** define a parallel `KernelSpec` type hierarchy.

Chosen over the alternative (a `KernelSpec`/`build` hierarchy — see below) for one decisive reason: **no parallel tree of types to define and maintain.** Both are AD-clean (spikes below), so the choice is purely ergonomic, and "use KernelFunctions, add nothing" wins.

### Why this is AD-safe even though the kernel carries params

A KernelFunctions kernel is a param-carrying struct, so optimizing with new params means **reconstructing the struct** from a parameter vector inside the loss — `θ → nlml(update(ExactGP(rebuild(θ)), X, y))`. That reconstruction lands in the AD tape. Spike 2 confirms it is **Mooncake-clean**. (This is the one thing Lux avoids — its layer struct holds config only, and the trainable params are a separate NamedTuple read directly; we can't be that pure because a kernel *is* its params and AbstractGPs consumes a kernel object. We pay one struct reconstruction per loss eval; it's negligible next to the O(n³) Cholesky and is what `fit` already does today via `mkkernel`.)

## Interface

The **public API barely changes** — the user still just builds a kernel and calls `fit`; `destructure` is internal to `fit`:

```julia
# user builds ANY KernelFunctions kernel — ARD, composite, whatever
k = 2.0 * with_lengthscale(SqExponentialKernel(), ones(d)) + Matern32Kernel()

g = ExactGP(k; noise = 1e-4)        # unchanged constructor
g = fit(g, X, y)                    # internally: destructure(k) → optimize log-params → rebuild → cache
μ, σ² = predict(g, Xstar)           # uses the cached rebuilt kernel (no rebuild at predict time)
```

Internally, `fit`:
1. `θ_raw, rebuild = destructure(g.prior.kernel)` (Functors round-trip).
2. Optimize `logθ` with bounded LBFGS; the kernel for a trial point is `rebuild(exp.(logθ))`.
3. On convergence, `rebuild` once with the optimal params, re-condition the GP, cache the kernel.

**Constraints (the one loose end, and it's small).** `destructure` returns *raw* values, and for our target subset (RBF / Matérn / RationalQuadratic + sums/products) **all leaves are positive** (output scales and inverse-lengthscales). So optimize `log` of the flat vector and `exp` before `rebuild` — a one-line transform layer, **not** ParameterHandling. A future bounded param (e.g. GammaExponential's γ ∈ (0,2]) gets special-cased then.

**Frozen ("st") hyperparameters.** A hyperparameter you do *not* want to fit (e.g. a fixed noise, or a held lengthscale) is a leaf held *out* of the optimized set. Functors' `trainable`/`@functor` distinction is the mechanism; the flat vector then covers only trainable leaves. (This replaces the earlier `merge(st, ps)` framing — same idea, expressed through Functors' trainable-leaf selection.)

**Lux mapping.** `destructure` ≈ Lux `setup` (extract the params from the template), and the GP's `predict` ≈ Lux `apply`. The kernel `k` is the structural template (Lux `model`-like), but unlike a Lux model it carries param *values* that `rebuild` overrides.

## Spike results (both gating risks retired)

| Spike | Mechanism | Result |
|---|---|---|
| 1 | explicit `build(θ)` closure (the rejected `KernelSpec` design) | Mooncake-clean — ARD + Sum, relerr ~1.5e-10 / 2.1e-10 |
| 2 | **`destructure(k)` → optimize → `rebuild`** (the chosen design) on an ARD-RBF + Matérn composite | **Mooncake-clean — relerr 8.7e-10; `rebuild` reproduces `k` exactly** |

Spike 2 also settled **leaf selection**: destructuring `2.0·withℓ(SqExp,[0.5,1,2]) + 1.0·withℓ(Matern32,0.8)` yielded exactly **6 params** — two output scales, three ARD inverse-lengthscales, one Matérn inverse-lengthscale — with **no spurious `ν`** (Matern32 is a parameterless type). For the target subset, Functors extracts exactly the trainable hyperparameters and nothing fixed.

## Convergence with Capability B (gp-ude)

The gp-ude branch already trains an **explicit flat parameter vector** — `v0 = [logℓ, logσ, logσ_obs, vec(w)…]` with a documented layout (`FieldLayout`, `NHYP`), rebuilding the field from it inside the differentiated loss. `destructure`'s flat-vector + rebuild is the same philosophy, generalized to arbitrary kernels — so adopting it for the active-learning `fit` unifies both capabilities on one parameterization story, serving the "one spine" thesis.

## Considered alternative (rejected): `KernelSpec` / `build` hierarchy

An abstract `KernelSpec` with `init_params(spec)` / `build(spec, ps, st)` methods per family (RBF, Matern, Sum, Product), where `ps` is a NamedTuple and `build` constructs the kernel one-way. Advantages: constraints live in `build` (`exp` the log-params), no extraction/round-trip. **Rejected** because it requires defining and maintaining a parallel type tree mirroring KernelFunctions (`RBF` spec *and* `SqExponentialKernel`, `Sum` *and* `KernelSum`) — exactly the anti-sprawl risk we want to avoid — and Spike 2 showed the round-trip it was meant to avoid is AD-clean anyway. (Spike 1 is retained as evidence that this path is also AD-viable, should the destructure approach ever hit a wall.)

## Tradeoffs of the chosen approach

- **Raw-param constraints** — handled by the log/exp transform; trivial for the positive-only subset, needs per-param care if we add bounded params later.
- **Leaf-selection is per-kernel** — verified clean for the subset; expanding the supported families means confirming Functors exposes only trainable leaves (use `trainable` to exclude fixed ones).
- **Dependency** — needs Functors (already a transitive dep of KernelFunctions; promote to direct). The flat-vector flatten can be a ~10-line Functors helper, or `Optimisers.destructure` (battle-tested but pulls Optimisers), or ComponentArrays. **Open decision** — lean Functors-direct to avoid a heavy dep.
- **Per-iteration rebuild** — one kernel reconstruction per loss eval, same as `build` and as today's `mkkernel`; negligible vs the Cholesky.

## Open decisions (for the implementing phase)

1. **Destructure dependency:** Functors + a minimal flatten helper (lean), vs `Optimisers.destructure`, vs ComponentArrays.
2. **GP storage / source of truth:** does `ExactGP` keep the user's `k` as the template and the optimized flat params as the refit source of truth (cleaner refit/serialization), or just cache the rebuilt kernel and re-`destructure` on each `fit`? (Re-destructuring each `fit` is simplest and probably fine — `destructure` is a one-time setup cost per fit.)
3. **AD backend:** the metric-based `_default_ad` dispatch still applies to the rebuilt kernel; high-dim ARD naturally routes to Mooncake.

## Scope / phasing

A dedicated phase **after Phase 1.5** (multi-output spine). Not Phase 1. The gating AD risks are spiked and retired. Remaining work: make `fit` destructure-based (replacing the hardcoded `mkkernel`), the log-transform constraint layer, migrate `LocalPenalization`/`grad_predict`'s lengthscale access off the `_lengthscale` peeler (it assumes a single scalar lengthscale and breaks on ARD/composites), and fold in the metric-based `_default_ad`.
