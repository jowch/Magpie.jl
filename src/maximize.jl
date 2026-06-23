using Optimization, OptimizationOptimJL, DifferentiationInterface
import Sobol

"""
    AcquisitionDomain

Supertype for the set an acquisition is maximized over: a continuous [`Box`](@ref)
(needs a search strategy) or an explicit [`Points`](@ref) set (argmax over enumerated
inputs).
"""
abstract type AcquisitionDomain end

"""
    Box(lb, ub) <: AcquisitionDomain

Axis-aligned box domain with lower and upper bounds `lb`, `ub` (per-dimension vectors).
"""
struct Box{T} <: AcquisitionDomain
    lb::T
    ub::T
    function Box(lb::T, ub::T) where {T}
        length(lb) == length(ub) ||
            throw(ArgumentError("Box bounds differ in length: lb has $(length(lb)), ub has $(length(ub))"))
        all(lb .≤ ub) ||
            throw(ArgumentError("Box requires lb .≤ ub; violated at dimension(s) $(findall(lb .> ub))"))
        return new{T}(lb, ub)
    end
end

"""
    Points(X) <: AcquisitionDomain

Finite candidate set: the acquisition is evaluated at each input in `X` and the best
is returned. `X` may be any point container (`Vector{Vector}`, `ColVecs`, …).
"""
struct Points{M} <: AcquisitionDomain
    X::M
end

"""
    AcqMaximizer

Supertype for strategies that maximize an acquisition over a continuous [`Box`](@ref).
"""
abstract type AcqMaximizer end

"""
    SobolPolish{A} <: AcqMaximizer

Two-stage maximizer for a [`Box`](@ref): score `n_candidates` low-discrepancy Sobol
points, then locally polish the best `n_restarts` of them with box-constrained LBFGS.

# Constructor

    SobolPolish(; n_candidates=2048, n_restarts=8, ad=AutoForwardDiff())

`ad` selects the DifferentiationInterface backend for the local polish gradients.
"""
struct SobolPolish{A} <: AcqMaximizer
    n_candidates::Int; n_restarts::Int; ad::A
end
SobolPolish(; n_candidates = 2048, n_restarts = 8, ad = AutoForwardDiff()) = SobolPolish(n_candidates, n_restarts, ad)

"""
    grid_points(box::Box; per_axis=50) -> Vector

Enumerate a regular grid over `box` with `per_axis` points along each dimension.
"""
function grid_points(box::Box; per_axis::Int = 50)
    d = length(box.lb)
    big(per_axis)^d > 1_000_000 &&
        throw(ArgumentError("grid of $(per_axis)^$(d) points exceeds 10^6; use SobolPolish() or a Points domain for high-D boxes"))
    axes = [range(box.lb[i], box.ub[i]; length = per_axis) for i in eachindex(box.lb)]
    return [collect(p) for p in Iterators.product(axes...)] |> vec
end

# Default maximizer per domain: enumerate Points; grid up to 2-D Boxes; SobolPolish above.
default_for(::Points) = nothing
default_for(b::Box) = length(b.lb) ≤ 2 ? Val(:grid) : SobolPolish()

"""
    acquire(g, a; over, maximizer=default_for(over))

Return the input in domain `over` that maximizes acquisition `a` under GP `g`. The
`maximizer` defaults to a sensible strategy for the domain (enumeration for
[`Points`](@ref), a grid for low-D [`Box`](@ref)es, [`SobolPolish`](@ref) otherwise).
"""
acquire(g, a; over, maximizer = default_for(over)) = _acquire(g, a, over, maximizer)

_argmax_over(g, a, X) = X[argmax([a(g, x) for x in X])]
_acquire(g, a, p::Points, _) = _argmax_over(g, a, p.X)
_acquire(g, a, b::Box, ::Val{:grid}) = _argmax_over(g, a, grid_points(b))

function _acquire(g, a, b::Box, m::SobolPolish)
    sobol_seq = Sobol.SobolSeq(b.lb, b.ub)
    candidates = [Sobol.next!(sobol_seq) for _ in 1:m.n_candidates]
    acq_values = [a(g, x) for x in candidates]             # scored once; reused for ranking + seeding
    ranked = sortperm(acq_values; rev = true)
    starts = candidates[ranked[1:min(m.n_restarts, length(candidates))]]
    best_point = candidates[ranked[1]]; best_value = acq_values[ranked[1]]
    for x0 in starts
        prob = OptimizationProblem(
            OptimizationFunction((x, _) -> -a(g, x), m.ad),
            collect(x0);
            lb = b.lb, ub = b.ub,
        )
        sol = solve(prob, Fminbox(LBFGS()))
        polished_value = -sol.objective
        if polished_value > best_value
            best_point, best_value = sol.u, polished_value
        end
    end
    return best_point
end
