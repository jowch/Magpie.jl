```@meta
CurrentModule = Magpie
```

# Magpie.jl

![Magpie.jl logo](assets/logo.svg)

*Composable Gaussian processes for active learning and dynamics —
a GP as a differentiable, uncertainty-aware component.*

> **Status: v0.1, experimental.** The incremental-GP spine and active-learning loop are
> implemented and tested; the GP-in-SciML bridge is the next milestone. APIs may move.

## What it is

Magpie is a Julia-ecosystem-native Gaussian process package where **the bridges are the
product**. It builds on [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl)
and [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl)
rather than reinventing the GP core, and treats a GP as a *differentiable,
uncertainty-aware* building block that composes with the rest of Julia — autodiff
(Mooncake), SciML, and ML.

Two capabilities hang off one shared, incrementally-updated GP spine:

- **Active learning** — a composable `fit → acquire → observe → update` loop with
  **level-set / boundary acquisitions** (Straddle, BALD-for-classification) that are
  not yet available in Julia's GP / Bayesian-optimization packages.
- **GP-in-SciML** *(next milestone)* — a GP as the right-hand side of an ODE, trained
  *through the solver*, for data-efficient, uncertainty-calibrated dynamics discovery.

## Ecosystem context

Magpie builds on [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) and [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) — it reuses their GP core and kernel library rather than reinventing them.

For optimization-style infill (expected improvement, upper confidence bound, SRBF), [Surrogates.jl](https://github.com/SciML/Surrogates.jl) and [BayesianOptimization.jl](https://github.com/jbrea/BayesianOptimization.jl) are mature choices; they target minima. Magpie's active-learning acquisitions (Straddle, BinaryBALD) target level sets and classification boundaries instead — a different objective these packages don't aim at.

For GP-as-ODE-field work, [GPDiffEq.jl](https://github.com/Crown421/GPDiffEq.jl) is the original proof of concept and deserves the credit; Magpie's Capability B (not yet built) continues that direction on a through-solver, Mooncake-trained path.

## Examples

See the **Examples** section in the navigation sidebar for worked vignettes.

## Install

Not yet registered — add from GitHub:

```julia
pkg> add https://github.com/jowch/Magpie.jl
```
