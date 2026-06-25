# Example Authoring Style Guide

How to write the worked examples in `examples/`. These are Literate.jl tutorials that render to the
documentation site **and** run in CI as anti-rot tests. The goal is examples that look like *problems
a user would actually bring*, not feature demos.

## Philosophy: problem-driven, not feature-driven

Frame each example as a real task: *"I have an expensive black-box function; use Magpie to answer a
question about it in few evaluations."* Lead with the problem and why it's hard (cost of an
evaluation, no second endpoint, an anisotropic boundary), then build the approach. A reader should
recognize the problem before they meet the API.

- Use realistic, recognizable settings (a potential energy surface, a feasibility window, a real DEM)
  with concrete motivation for *why few evaluations matter*.
- Closed-form or bundled ground truth is fine — keep examples self-contained (no external fetches).
- One clear question per example. Don't stitch two stories together (the old Müller–Brown did both
  "enumerate all critical points" and "find the transition state" — split that).

## Tenor: narrative is structural, prose is plain

This is the rule we care about most.

- **The narrative lives in the structure** — the section arc carries the story:
  *problem → why it's hard → how we model it → build the approach → results/diagnostics.* The reader
  follows a coherent sequence; that *is* the narrative.
- **The prose stays plain and expository.** Write like a knowledgeable colleague explaining a real
  problem: *"In chemistry, locating a transition state is expensive because each energy evaluation is
  a DFT calculation. We can model this kind of problem with a GP and…"*
- **Collegial, not academic; plain, not flowery.** No dramatization ("imagine you are on a quest"),
  no purple metaphors, no hype adjectives. If a sentence is carrying narrative weight with adjectives,
  rewrite it so the *structure* carries it instead. Keep sentences economical.

## Honesty: don't overclaim

A claim in the prose must be backed by what the code shows.

- If you say a kernel is anisotropic (ARD), **show the fitted lengthscales** and that they actually
  differ (e.g. `ℓ = [0.158, 0.095]` — a real ~1.7× anisotropy). A claim with near-equal lengthscales
  is hollow; soften the prose or pick a better example.
- If a technique is **not** sensible for the problem, say so plainly and don't use it for show. (ARD
  on a *symmetric* classification boundary gains nothing — we kept that example isotropic and said
  why.)
- State results in calibrated terms ("a few dozen evaluations", not a precise number you didn't
  measure). Show a real diagnostic, not a vibe.

## What to show

Match the analyses to what a practitioner in that domain expects, scaled to the example's role
(headline = deep; supporting = light):

- **A correctness diagnostic**, not just a pretty plot — e.g. the located saddle's Hessian has exactly
  one negative eigenvalue (index-1); the recovered ARD lengthscales; level-set sign-recovery.
- **A fair baseline** where it makes the point (random/grid/uncertainty-only). Keep settings symmetric
  across arms — if the method refits, the baseline refits too, or the comparison is a confound.
- Standard pictures for the task: GP posterior mean ± std; the acquisition surface and where it
  queries; queries concentrating on the boundary; convergence/error vs number of evaluations; a
  learning curve vs queries. Don't include every plot — include the ones that make *this* example's
  point.

## Kernels

Use kernel composition (ARD SE, sums/products, custom kernels) **where it is genuinely motivated and
fittable** — anisotropic axes, per-subspace structure. Prefer letting `fit` recover hyperparameters
over hand-setting them, and show the recovered values. Defining a custom kernel is a fair teaching
moment when the problem needs one (e.g. the r²-Taylor Matérn for anisotropic derivative GPs).

## Coherence across the set

Examples should **complement, not duplicate.** After one example owns "find the transition state",
another can own "recover all the critical points" — keep the division of labor clean and the
cross-references accurate. When you change one example's story, fix any sibling that refers to it.

## Attribution

Credit borrowed techniques/data with their license (e.g. the r²-Taylor kernel pattern from
CovarianceFunctions.jl, MIT; the `volcano` DEM). A short inline note is enough.

## Mechanics

- **Format:** Literate.jl source in `examples/<name>.jl`. Prose in `#` comments; test-only lines end
  in `#src` (kept when executed, stripped from the rendered page). Figures via **Plots.jl**;
  `ENV["GKSwstype"] = "100"` for headless rendering.
- **Register it** in `docs/make.jl`'s `EXAMPLES` list (title + filename).
- **Anti-rot:** every example must run standalone — `julia --project=docs examples/<name>.jl` exits 0,
  with `#src` `@assert`/`@test` lines that **genuinely gate** the result (a wrong run fails them; they
  are not trivially true).
- Format with Runic before committing; confirm `julia --project=docs docs/make.jl` builds.

## Checklist

- [ ] Frames a realistic problem; one clear question.
- [ ] Narrative carried by section structure; prose plain and economical (no dramatization).
- [ ] Every prose claim backed by what the code shows; no overclaiming; honest about what isn't
      sensible.
- [ ] A real correctness diagnostic and (where it helps) a fair, symmetric baseline.
- [ ] Kernel composition used only where motivated and fittable; recovered values shown.
- [ ] Complements the other examples; cross-references accurate.
- [ ] Borrowed techniques/data attributed.
- [ ] Runs standalone (exit 0) with non-vacuous `#src` assertions; registered in `docs/make.jl`; docs
      build green.
