```@meta
CurrentModule = Magpie
```

# API Reference

The public surface of Magpie.jl. Names are grouped by role; see the
[Examples](generated/levelset_straddle.md) for worked usage.

## Models and inference

```@docs
AbstractGPModel
ExactGP
LaplaceGP
update
predict
predmean
nlml
fit
```

## Acquisitions

```@docs
AcquisitionFunction
MarginalAcquisition
Straddle
RandStraddle
BinaryBALD
resample
```

## Maximizing an acquisition

```@docs
AcquisitionDomain
Box
Points
AcqMaximizer
SobolPolish
grid_points
acquire
```

## Active-learning loop

```@docs
ActiveLearner
observe!
fit!
run!
posterior_gp
queried_points
all_data
```
