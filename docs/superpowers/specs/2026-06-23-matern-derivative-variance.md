# Matérn Derivative Variance & the r²-Taylor Path — Research Note

**Date:** 2026-06-23
**Status:** Reference — records why `grad_predict` uses analytic per-family constants for the Matérn prior gradient variance, the generic upgrade path, and a factor-3 bug this investigation surfaced and fixed.
**Relates to:** [`src/derivatives.jl`](../../../src/derivatives.jl) `_prior_grad_var`/`_prior_grad_var_const`; the [multi-output boundary note](2026-06-23-multi-output-boundary.md); the scalar-ℓ requirement on derivative paths.

## The problem

`grad_predict` needs the per-component **prior gradient variance** `Var[∂ᵢf(x)] = −∂²_{zᵢ} k(z,x)|_{z=x}` — the diagonal of the kernel's query-Hessian at coincidence. Everything else in `grad_predict` (gradient mean, Hessian, the data term) is AD of the posterior mean and is fully kernel-generic. This one term is the only kernel-specific piece, and it is where the Matérn family breaks AD.

Two **distinct** Hessian problems are routinely conflated (both verified against sources, see Citations):
1. **Input-space** ∇²ₓk — what we need (derivative GPs / gradient acquisitions). Singular at r=0 for norm-metric kernels.
2. **Hyperparameter** ∂/∂ν (the Bessel `Kᵥ` order) — for marginal-likelihood optimization over a *continuous* smoothness ν.

We only ever use **fixed half-integer ν** (Matérn-1/2, 3/2, 5/2 → closed-form polynomial×exp kernels), so problem (2) — and the complex-step-on-Bessel method of Marin et al. (2022), arXiv:2201.00262 — **does not arise here** (no `Kᵥ` at runtime, ν never optimized). Our concern is purely (1).

## Why AD fails, and the generic formula

For a smooth (squared-distance) kernel — SqExponential, RationalQuadratic — `k` is a function of `r² = ‖z−x‖²`, so `ForwardDiff.hessian(z → k(z,x), x)` is finite at coincidence and `_prior_grad_var` is fully generic (works for ARD, composites). For a **norm-metric** kernel (Matérn, Exponential), `k` depends on `r = √(r²)`, and `∂²/∂z²` forms `√0` → AD returns `NaN`. So we fall back to an analytic constant.

Writing any isotropic kernel as `C(τ) = σ²·ψ(s)` with `s = ‖τ‖²/ℓ²` (ARD: `s = Σⱼ τⱼ²/ℓⱼ²`), a clean derivation gives the **generic coincidence formula**:

```
Var[∂ᵢf] = −2 σ² ψ′(0) / ℓᵢ²
```

where `ψ′(0)` is the first derivative of the kernel **w.r.t. squared distance** at 0. This is ARD-friendly (per-dimension `ℓᵢ²`) and composable. Evaluating `ψ′(0)`:

| kernel | ψ(s) near 0 | ψ′(0) | `Var[∂ᵢf]` |
|---|---|---|---|
| SqExponential | `e^{−s/2}` | −1/2 | σ²/ℓ² |
| Matérn-3/2 | `1 − (3/2)s + …` | −3/2 | **3σ²/ℓ²** |
| Matérn-5/2 | `1 − (5/6)s + …` | −5/6 | 5σ²/(3ℓ²) |

## Bug found and fixed (2026-06-23)

Deriving the constants from the generic formula reproduced SqExp (σ²/ℓ²) and Matérn-5/2 (5σ²/3ℓ²) — both matched the code and its tests — but Matérn-3/2 came out **3σ²/ℓ²**, while `_prior_grad_var_const` returned **σ²/ℓ²** (a factor of 3 too small). Confirmed by central differences of the actual KernelFunctions kernel (`Var[∂₁f] → 6.12 = 3/ℓ²` at ℓ=0.7, σ²=1; `→75 = 3σ²/ℓ²` at σ²=9, ℓ=0.6). It slipped because the Matérn-5/2 *value* is tested but the Matérn-3/2 value never was — the AD-fallback path is only reached for Matérn, and only M-5/2 had a value assertion.

**Fix:** `Matern32Kernel → 3σ² / ℓ²`, plus a value test in `test/test_derivatives.jl` mirroring the M-5/2 one. Impact of the bug: `grad_predict` under-estimated the prior gradient-explore variance by 3× for Matérn-3/2 GPs only (under-exploration in `GradStraddle`/derivative acquisitions far from data).

## Generic upgrade path (deferred): the r²-Taylor technique

The per-family analytic constant is correct for our scope (isotropic scalar-ℓ Matérn) but is exactly the kind of hand-derivation that produced the factor-3 error, and it does not extend to ARD/composite Matérn (this is the root of the scalar-ℓ requirement on `grad_predict` for Matérn kernels). The generic, ARD-friendly alternative is to compute `ψ′(0)` AD-transparently by **parametrizing the kernel in r² and using a Taylor-series branch near r²=0**, so AD differentiates a polynomial and never forms `√0`. This is the technique implemented (for kernel *values*) in **CovarianceFunctions.jl** (`src/stationary.jl`, MIT-licensed).

**Status: not adopted, deliberately.** Reasons:
- **Nothing pulls it.** The critical-points/derivative acquisitions use isotropic scalar-ℓ kernels; ARD/composite *derivative* GPs are not a current use case.
- **CovarianceFunctions.jl is not a viable dependency:** own kernel hierarchy (no KernelFunctions interop), ForwardDiff+SymEngine internally (Mooncake-hostile, conflicting with our Mooncake-first stance), Flux-0.13-pinned, ~15 heavy deps, unmaintained since Jan 2023. If adopted, **borrow the ~15-line r²-Taylor pattern** (MIT — reproduce the license/attribution), do not depend on the package.
- Implementing it now would be speculative per the anti-sprawl principle.

When ARD/composite Matérn *derivative* GPs are actually needed, the recipe is: evaluate each Matérn component in r² with a Taylor branch below a machine-eps threshold, then `Var[∂ᵢf] = −2σ²ψ′(0)/ℓᵢ²` falls out of AD generically — lifting the scalar-ℓ requirement on the derivative path.

## Citations (all verified)

- **Analytic derivative-kernel covariances and their r=0 limits** (what our constants encode):
  - Solak, Murray-Smith, Leithead, Leith, Rasmussen, "Derivative Observations in Gaussian Process Models of Dynamic Systems," NIPS **2002**, pp. 1033–1040. https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html
  - Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press 2006, **§9.4 "Derivative Observations."** https://gaussianprocess.org/gpml/
  - Eriksson, Dong, Lee, Bindel, Wilson, "Scaling Gaussian Process Regression with Derivatives," NeurIPS 2018. arXiv:1810.12283
- **Structured AD for derivative kernels** (the CovarianceFunctions.jl paper): Ament & Gomes, "Scalable First-Order Bayesian Optimization via Structured Automatic Differentiation," ICML 2022, PMLR 162:500–516. arXiv:2206.08366. **Caveat:** the r²-Taylor singularity-avoidance trick is in the package *code*, not stated in the paper body — cite it as an implementation technique, not a published result.
- **Hyperparameter ∂/∂ν (not relevant here):** Marin et al. (2022), "Differentiating the Matérn kernel w.r.t. the smoothness parameter," arXiv:2201.00262 — complex-step on the Bessel series; moot for fixed half-integer ν.
