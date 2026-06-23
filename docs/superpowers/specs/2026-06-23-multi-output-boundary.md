# Multi-Output Boundary — Scope Note

**Date:** 2026-06-23
**Status:** Reference — records what Magpie's multi-output support covers, what it does not, and how to get the rest from the ecosystem today.
**Relates to:** [Phase 1.5 multi-output spine](2026-06-22-al-foundation-hardening-design.md) and [kernel parameterization](2026-06-22-kernel-parameterization-design.md).

## TL;DR

`ExactGP`'s `d>1` mode is the **independent, shared-kernel, isotopic** multi-output model — conceptually `IndependentMOKernel` on shared inputs. The coregionalized / mixed / per-domain MO kernels (`IntrinsicCoregionMOKernel`, `LinearMixingModelKernel`, `LatentFactorMOKernel`) are **not** a Magpie feature; they are available **today** through plain AbstractGPs + KernelFunctions via input augmentation. Adding Magpie's value-add (incremental `update`, `fit`, acquisitions) *on top of* coregionalized MO is deferred — no current capability pulls it.

## What Magpie covers

Phase 1.5 `ExactGP` with `d` outputs stores:

- one shared kernel, one shared `n×n` Cholesky `C`,
- weights as an `n×d` matrix (`α = C \ δ`, `δ` is `n×d`),
- a marginal **variance shared across outputs** (computed once).

This is exactly the `K ⊗ I` structure: `d` independent outputs, identical kernel, common inputs. It is the compact, efficient form of `IndependentMOKernel` for **isotopic** data (every output observed at every input), and it is what the gp-ude `ExactGPField` (an ODE RHS modeled as `d` independent component functions) needs.

## What it does not cover, and why the representation can't extend

The complex MO kernels all add cross-output structure that the `n×d`-weights / shared-variance form discards:

| Kernel | Gram | Captures | In `ExactGP`? |
|---|---|---|---|
| `IndependentMOKernel` | `K ⊗ I` | independent outputs, shared kernel | ✅ (compactly) |
| `IntrinsicCoregionMOKernel` (ICM) | `K ⊗ B` | cross-output covariance via PSD `B` | ❌ |
| `LinearMixingModelKernel` (LMC) | `Σᵢ Kᵢ ⊗ Bᵢ` | several latent processes, **a different kernel each** | ❌ |
| `LatentFactorMOKernel` (SLFM) | `Σᵢ Kᵢ ⊗ Bᵢ` (+ noise structure) | latent factors | ❌ |

The shared single `n×n` Cholesky and the one shared variance vector are a `K ⊗ I` commitment. ICM (`K ⊗ B`) and LMC (a sum of Kroneckers) need a fundamentally different factorization — either the full `nd×nd` augmented Gram or a Kronecker-structured solver. It is not an extension of the Phase 1.5 representation; it is a parallel one.

"Different kernels per output / per input domain" is the LMC/SLFM (different kernel per latent process) or heterotopic-data case — same conclusion.

## How to do coregionalized MO today (ecosystem)

KernelFunctions represents MO by **input augmentation**: an input becomes a `(x, output_index)` tuple. `MOInput(X, d)` builds the augmented input vector; the MO kernels evaluate on the tuples (e.g. ICM is `B[pₓ,p_y] · k(x,y)`); AbstractGPs conditions over the resulting `nd×nd` Gram with no special-casing. Verified end-to-end:

```julia
using AbstractGPs, KernelFunctions, LinearAlgebra
A = randn(d, d); B = A * A' + 0.1I                       # PSD coregionalization matrix
k = IntrinsicCoregionMOKernel(kernel = with_lengthscale(SqExponentialKernel(), 0.7), B = B)
Xmo = MOInput(X, d)                                       # (x, output) tuples; length == d·length(X)
post = posterior(GP(k)(Xmo, σ²), y)                       # y is the stacked nd-vector
μ, σ²_pred = mean_and_var(post, MOInput(Xtest, d))
```

So coregionalized **prediction** needs no Magpie code. What Magpie would add — incremental `update`/`update_chol`, hyperparameter `fit` (incl. fitting `B`), and the level-set/BALD acquisitions — is currently single-output / independent-MO only (the `d>1` guards on `fit`/`grad_predict`/`acquire` enforce this).

## Deferred (until a use case pulls it)

A "Magpie-flavored coregionalized MO" phase — an augmented-input `ExactGP` variant leaning on KernelFunctions' MO kernels, with Kronecker efficiency (`chol(K)` + `chol(B)` instead of `chol(K⊗B)`), `fit` over `B`, and MO acquisitions — is a real but **speculative** phase. Neither flagship needs it: Capability A (active learning / critical points) is single-output, and Capability B (GP-UDE) uses independent outputs. Per the anti-sprawl principle, it waits for a concrete driver: cross-correlated outputs, multi-fidelity, or genuinely per-domain kernels.

The documented next phase remains **Capability B (GP-in-SciML / GP-UDE bridge)**.
