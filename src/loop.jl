"""
    ActiveLearner{TX, TY}

Mutable state for an active-learning run: the current GP, the acquisition, and the
accumulated query history.

# Fields
  - `gp`: the current [`AbstractGPModel`](@ref), replaced on each observation
  - `acq`: the acquisition function (reassignable, e.g. to wrap it with
    [`LocalPenalization`](@ref) over the live history)
  - `Xs`, `Ys`: accumulated query inputs and observed values, in query order
  - `acq_vals`: the acquisition score at each queried point
  - `rng`: the random-number generator threaded through `acquire`/`run!`

# Constructor

    ActiveLearner(gp, acq; rng=Random.default_rng())

Start a learner from an (optionally unconditioned) GP and an acquisition. Both `gp` and
`acq` are abstractly typed so they can be replaced in place during a run. `TX` and `TY`
are inferred from `gp` (inputs are `Vector{Float64}`; values are `Float64` for `ExactGP`
and `Bool` for `LaplaceGP`).
"""
mutable struct ActiveLearner{TX, TY}
    gp::AbstractGPModel              # abstract on purpose: concrete type changes on first update
    acq::AcquisitionFunction         # abstract on purpose: changes when wrapped by LocalPenalization
    Xs::Vector{TX}
    Ys::Vector{TY}
    acq_vals::Vector{Float64}
    rng::AbstractRNG
end

# Observation element type implied by the GP kind.
_obs_eltype(::ExactGP) = Float64
_obs_eltype(::LaplaceGP) = Bool

# Typed escape hatch: caller fixes the input/value element types.
ActiveLearner{TX, TY}(gp, acq; rng = Random.default_rng()) where {TX, TY} =
    ActiveLearner{TX, TY}(gp, acq, TX[], TY[], Float64[], rng)

# Convenience: infer Vector{Float64} inputs and the value type from the GP.
function ActiveLearner(gp, acq; rng = Random.default_rng())
    gp isa ExactGP && gp.d > 1 && throw(
        ArgumentError(
            "ActiveLearner supports single-output GPs only (got d=$(gp.d)); the loop's " *
                "acquire/fit also guard d>1. Multi-output acquisitions are deferred."
        )
    )
    return ActiveLearner{Vector{Float64}, _obs_eltype(gp)}(gp, acq; rng = rng)
end

"""The current posterior GP held by the learner."""
posterior_gp(al::ActiveLearner) = al.gp
"""Copy of the inputs queried so far, in order."""
queried_points(al::ActiveLearner) = copy(al.Xs)
"""Copies of all queried inputs and their observed values, as `(Xs, Ys)`."""
all_data(al::ActiveLearner) = (copy(al.Xs), copy(al.Ys))

# A batch is a non-empty vector whose elements are themselves vectors (multiple inputs).
_isbatch(X) = X isa AbstractVector && !isempty(X) && first(X) isa AbstractVector

"""
    observe!(al::ActiveLearner, X, Y) -> al

Record observation(s) `(X, Y)` and condition the GP on them. Accepts either a single
input/value or a batch (a vector of inputs with a matching vector of values).
"""
function observe!(al::ActiveLearner, X, Y)
    X_batch = _isbatch(X) ? X : [X]
    Y_batch = Y isa AbstractVector ? Y : [Y]
    _validate_obs(X_batch, Y_batch)
    append!(al.Xs, X_batch); append!(al.Ys, Y_batch)
    al.gp = Magpie.update(al.gp, X_batch, Y_batch)
    return al
end

"""
    fit!(al::ActiveLearner; restarts=1) -> al

Refit the GP's hyperparameters in place (see [`fit`](@ref)).
"""
fit!(al::ActiveLearner; restarts::Int = 1) = (al.gp = Magpie.fit(al.gp; restarts = restarts); al)

"""
    acquire(al::ActiveLearner; over, maximizer=default_for(over), q=1)

Pick the next query point by maximizing the (resampled) acquisition over `over`.
Batch acquisition (`q > 1`) is not yet implemented.
"""
function acquire(al::ActiveLearner; over, maximizer = default_for(over), q::Int = 1)
    q == 1 || error("batch acquisition (q>1) not yet implemented; use q=1")
    if over isa Box && !isempty(al.Xs)
        length(over.lb) == _inputdim(first(al.Xs)) ||
            throw(ArgumentError("domain dimension $(length(over.lb)) ≠ GP input dimension $(_inputdim(first(al.Xs)))"))
    end
    return acquire(al.gp, resample(al.acq, al.rng); over = over, maximizer = maximizer)
end

"""
    run!(al::ActiveLearner, oracle; budget, over, stop=al->false, refit_every=0) -> al

Run the active-learning loop for up to `budget` rounds. Each round resamples the
acquisition, picks the next point over `over`, queries `oracle` for its value, and
conditions the GP on it. Stops early when `stop(al)` is true; refits hyperparameters
every `refit_every` rounds (`0` disables refitting).
"""
function run!(al::ActiveLearner, oracle; budget::Int, over, stop = al -> false, refit_every::Int = 0)
    for t in 1:budget
        stop(al) && break
        acq = resample(al.acq, al.rng)
        x = acquire(al.gp, acq; over = over)
        push!(al.acq_vals, acq(al.gp, x))
        observe!(al, x, oracle(x))
        refit_every > 0 && t % refit_every == 0 && fit!(al)
    end
    return al
end
