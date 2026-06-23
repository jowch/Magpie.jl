# Kernel Parameterization — Design Note (explicit-parameter `(spec, ps, st)`)

**Date:** 2026-06-22
**Status:** Design exploration / decision record — seeds a future phase (after Phase 1.5). The gating AD risk is **spiked and resolved** (see below).
**Relates to:** [Phase 1 foundation hardening](2026-06-22-al-foundation-hardening-design.md) (Task 4 made `fit` kernel-generic over a whitelist; this note is how that generalizes), and the gp-ude branch's field training.

## Problem

`fit` today flattens a kernel to exactly `[logℓ, logσ²]` and rebuilds it as `σ²·with_lengthscale(base, ℓ)`. That model can't express **ARD** (a lengthscale per input dimension), **composite kernels** (sums/products), or **extra shape params** (RationalQuadratic's α). We want extensible hyperparameter fitting — at least ARD and composites — without blowing up complexity, and ideally without the AD fragility that comes from extracting parameters back out of a built kernel object.

The constraint is **not** ForwardDiff-ability; it's the *parameterization model*. The single-`(ℓ,σ²)` reconstruction, and the `_lengthscale`/`_outputscale` peelers it relies on (also used by `LocalPenalization` and `grad_predict`), assume one scalar-lengthscale base kernel.

## What the KernelFunctions ecosystem provides (v0.10.67, verified against installed source)

- KernelFunctions deliberately exposes **no parameter interface of its own**; its stated goal is to *"interoperate with generic packages for handling parameters like ParameterHandling.jl and FluxML's Functors.jl."*
- Every kernel/transform is **`@functor`'d** (`ScaleTransform`, `ARDTransform`, `TransformedKernel`, `KernelSum`, `KernelProduct`), so Functors can walk/rebuild any kernel tree.
- **ARD is first-class:** `with_lengthscale(base, ℓ::AbstractVector) → base ∘ ARDTransform(inv.(ℓ))`.
- The two recommended mechanisms differ only in *constraints*: **Functors** = tree-walking (you own log/positivity + trainable scoping); **ParameterHandling** = flatten/unflatten **+** a constraint layer (`positive()`, …) — the JuliaGPs standard, and the heaviest option.

The awkward part of both is the **reverse direction** — extracting params *out of* a built kernel and differentiating through an unflatten/reconstruct. `fit.jl` already carries a comment that it avoided ParameterHandling's unflatten "to stay AD-compatible."

## Decision: Lux-style explicit `(spec, ps, st)`, one-way `build`

Adopt the Lux.jl separation. Lux models hold **no** parameters/state; `ps, st = setup(rng, model)`, the forward pass is pure `model(x, ps, st) → (y, st_new)`, AD differentiates the explicit `ps`, and `ComponentArray(ps)` gives a flat AD-transparent vector for optimizers. The reason it's AD-robust: the differentiated object is a plain nested NamedTuple/array — **no functor-walk or unflatten-of-a-reconstructed-struct in the hot path.**

Mapped onto kernels:

| Lux | Our analogue |
|---|---|
| `model` (architecture, no params) | **`spec`** — kernel architecture; fixes the shape of `ps`, knows how to `build` |
| `ps` (NamedTuple of arrays) | **trainable hyperparameters**, unconstrained (log) space; `ComponentArray` for the flat optimizer view |
| `st` (non-trained) | **frozen hyperparameters** (see adaptation below) |
| `model(x, ps, st)` | **`build(spec, ps, st) → KernelFunctions.Kernel`** |

The flow is strictly **one-way** — params live in `ps`/`st` and flow *into* construction; we never peel them back out:

```
(spec, ps, st) ──build──▶ KernelFunctions.Kernel ──AbstractGPs.cov──▶ K ──▶ nlml
       └──────────────── AD differentiates ps through this chain ───────────────┘
```

### The `st` adaptation (important)

A kernel is a **pure** function of its hyperparameters — unlike a Lux `BatchNorm`, there is no running statistic the forward pass mutates. So Lux's "mutable state updated by the forward pass" does not literally exist at the kernel level. We **repurpose the `st` slot for *frozen* (non-trained) hyperparameters**:

- **`ps` = trained hyperparameters; `st` = frozen hyperparameters; `merge(st, ps)` = the full set `build` reads.**
- Freezing a hyperparameter (fix σ and fit only ℓ; or gp-ude's *fixed* `lognoise`) = move that field from `ps` to `st`. Spec and `build` are unchanged; AD never sees the frozen field.
- `st` is usually `(;)`; it stays in the signature for uniformity and freeze/unfreeze ergonomics.

The genuinely Lux-style *mutable* state in our world — the conditioned cache (`C`, `α`, data) and the loop `rng` — lives one level up in the **GP model / `ActiveLearner`**, NOT in the kernel. We deliberately do not conflate kernel `(spec, ps, st)` with GP conditioned state.

## Interface (pinned)

```julia
abstract type KernelSpec end

# Extension point — add a kernel family by adding these two methods:
#   init_params(s::KernelSpec) -> ps::NamedTuple        (unconstrained / log space)
#   build(s::KernelSpec, ps, st) -> KernelFunctions.Kernel

struct RBF    <: KernelSpec; dim::Int; ard::Bool; end
struct Matern <: KernelSpec; ν::Rational; dim::Int; ard::Bool; end   # ν ∈ (1//2,3//2,5//2): a constant, not a param
struct Sum{T<:Tuple}     <: KernelSpec; parts::T; end
struct Product{T<:Tuple} <: KernelSpec; parts::T; end

init_params(s::RBF) = (; logσ = 0.0, logℓ = s.ard ? zeros(s.dim) : 0.0)
build(s::RBF, ps, st) = (p = merge(st, ps);
    exp(2p.logσ) * with_lengthscale(SqExponentialKernel(), exp.(p.logℓ)))   # scalar→Scale, vector→ARD, for free

init_params(s::Sum) = NamedTuple{ntuple(i -> Symbol(:k, i), length(s.parts))}(init_params.(s.parts))  # nested
build(s::Sum, ps, st) = sum(build(p, ps[i], get(st, keys(ps)[i], (;))) for (i, p) in enumerate(s.parts))
```

- **Positivity constraints live in `build`** (`exp` the log-params) — KernelFunctions only ever sees valid positive values. No ParameterHandling.
- **ARD is free**: `with_lengthscale` dispatches scalar→`ScaleTransform`, vector→`ARDTransform`.
- **Composites compose**: `Sum`/`Product` specs map to `KernelSum`/`KernelProduct` with nested `ps`.
- The `fit` loss becomes `θ -> nlml(condition(build(spec, unflatten(θ), st), X, y))`, with `θ = ComponentArray(ps)` giving the flat ↔ named bridge AD-transparently.

## Convergence with Capability B (gp-ude)

The gp-ude branch already trains an **explicit flat parameter vector** — `v0 = [logℓ, logσ, logσ_obs, vec(w)…]` with a documented layout (`FieldLayout`, the `NHYP` constant), and `gpfield(field, u, pf)` rebuilds the field from `pf` inside the differentiated loss. That is the Lux explicit-parameter philosophy, hand-rolled. A `(spec, ps::NamedTuple/ComponentArray)` representation is the named, composable generalization of gp-ude's `v0` — so adopting it for the active-learning `fit` **unifies both capabilities on one parameterization philosophy**, serving the "one spine" thesis.

## Spike result (gating AD risk — resolved)

`scratch: spike_explicit_params.jl` — differentiate `θ -> nlml(update(ExactGP(build(θ)), X, y))` under **Mooncake** vs central differences, for the two cases the Functors/peeler approach struggled with:

```
ARD-RBF (d=3):     finite=true  relerr=1.55e-10
Sum RBF+Matern32:  finite=true  relerr=2.09e-10   (Matérn diagonal is fine under Mooncake; ForwardDiff would NaN)
```

The explicit-param `build` path is AD-clean under Mooncake for vector lengthscales and composites-with-Matérn. The design's central risk is retired.

## Tradeoffs

- **More upfront structure** than a Functors hack: a small `KernelSpec` hierarchy + `init_params`/`build` per family. This *is* the extensible foundation (extend-by-method, `AbstractGPModel`-style), but it's design work, not a one-liner.
- **Two representations** — `(spec, ps, st)` (trainable) and the built KernelFunctions kernel (evaluation). Lux lives with exactly this; manage the coupling (see open decision).
- **ComponentArrays.jl dependency** — small, SciML-core, AD-mature; lighter and more AD-proven than ParameterHandling for the flatten-to-vector need.
- **Optimization gets harder with richer kernels** (a sum-kernel marginal likelihood is multimodal) — more restarts/priors needed; orthogonal to the parameterization but real.
- **AD backend dispatch still applies**: the built kernel's metric picks ForwardDiff (SqEuclidean/DotProduct) vs Mooncake (Euclidean); high-dim ARD naturally routes to Mooncake (reverse-mode wins as param count grows). See the Phase-1 `_default_ad` chokepoint.

## Open decision (for the implementing phase)

**Where is the source of truth in the GP model?**
- **(A, recommended) `(spec, ps, st)` is canonical; the KernelFunctions kernel is a derived cache.** Cleaner refit/serialization, fully Lux-pure, unifies with gp-ude. Larger change to `ExactGP` (it grows `spec`/`ps`/`st`; `update`/`predict` build-and-cache the kernel).
- **(B) The built kernel stays primary; carry `(spec, ps)` only for refit.** Smaller change to `ExactGP`, but two half-sources of truth.

Recommendation: **A** for the foundation — it's the one place this ripples into the spine, and doing it properly once is cheaper than half-adopting it.

## Scope / phasing

This is a dedicated phase **after Phase 1.5** (multi-output spine). It is not Phase 1. The gating AD risk is already spiked; the remaining work is the `KernelSpec` hierarchy, the `ExactGP`/`fit` refactor per the open decision, and migrating `LocalPenalization`/`grad_predict`'s lengthscale access off the peelers onto `ps`/`spec`. When it lands, fold the metric-based `_default_ad` dispatch in at the same time.
