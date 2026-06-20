using Optimization, OptimizationOptimJL, DifferentiationInterface
import Sobol

# An acquisition domain is either a continuous Box (needs a search strategy) or an
# explicit Points set (argmax over the enumerated inputs). Nominal types give clean
# dispatch + errors and accept any point container (Vector{Vector}, ColVecs, …).
abstract type AcquisitionDomain end
struct Box{T} <: AcquisitionDomain; lb::T; ub::T; end
struct Points{M} <: AcquisitionDomain; X::M; end

abstract type AcqMaximizer end
struct SobolPolish{A} <: AcqMaximizer; n_raw::Int; n_restarts::Int; ad::A; end
SobolPolish(; n_raw=2048, n_restarts=8, ad=AutoForwardDiff()) = SobolPolish(n_raw, n_restarts, ad)

function grid_points(box::Box; per_axis::Int=50)
    axes = [range(box.lb[i], box.ub[i]; length=per_axis) for i in eachindex(box.lb)]
    return [collect(p) for p in Iterators.product(axes...)] |> vec
end

default_for(::Points) = nothing
default_for(b::Box) = length(b.lb) ≤ 2 ? Val(:grid) : SobolPolish()

acquire(g, a; over, maximizer=default_for(over)) = _acquire(g, a, over, maximizer)

_argmax_over(g, a, X) = X[argmax([a(g, x) for x in X])]
_acquire(g, a, p::Points, _) = _argmax_over(g, a, p.X)
_acquire(g, a, b::Box, ::Val{:grid}) = _argmax_over(g, a, grid_points(b))

function _acquire(g, a, b::Box, m::SobolPolish)
    seq = Sobol.SobolSeq(b.lb, b.ub)
    raw = [Sobol.next!(seq) for _ in 1:m.n_raw]
    vals = [a(g, x) for x in raw]                          # evaluate once; reuse for ranking + seeding
    perm = sortperm(vals; rev=true)
    starts = raw[perm[1:min(m.n_restarts, length(raw))]]
    best = raw[perm[1]]; bestv = vals[perm[1]]
    for x0 in starts
        prob = OptimizationProblem(
            OptimizationFunction((x, _) -> -a(g, x), m.ad),
            collect(x0);
            lb=b.lb, ub=b.ub,
        )
        sol = solve(prob, Fminbox(LBFGS()))
        v = -sol.objective
        if v > bestv
            best, bestv = sol.u, v
        end
    end
    return best
end
