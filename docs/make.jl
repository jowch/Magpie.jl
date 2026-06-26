ENV["GKSwstype"] = "100"  # GR renders headlessly (no display) in CI / doc build
using Documenter, Literate, Magpie

const EXDIR = joinpath(@__DIR__, "..", "examples")
const GENDIR = joinpath(@__DIR__, "src", "generated")

# Example pages (Literate source → generated page); add new examples here.
const EXAMPLES = Tuple{String, String}[]

push!(EXAMPLES, ("Level-set recovery (Straddle)", "levelset_straddle.jl"))
push!(EXAMPLES, ("Decision boundary (BinaryBALD)", "bald_classification.jl"))
push!(EXAMPLES, ("Müller–Brown: a transition state from one known minimum", "muller_brown.jl"))
push!(EXAMPLES, ("Volcano terrain: Morse critical points of a DEM", "volcano_terrain.jl"))
push!(EXAMPLES, ("GP-UDE: Lotka-Volterra", "gp_ude_lotka_volterra.jl"))
push!(EXAMPLES, ("GP-UDE: Van der Pol oscillator", "gp_ude_vanderpol.jl"))
push!(EXAMPLES, ("GP-UDE: FitzHugh-Nagumo (UDE decomp)", "gp_ude_fitzhugh_nagumo.jl"))
push!(EXAMPLES, ("GP-UDE: identifiability (ridge + off-data divergence)", "gp_ude_identifiability.jl"))
# NOTE: gp_ude_scale_forcing.jl is a BY-HAND demo (multi-trajectory sampled SVGP training) — it ran
# ~78 min in the docs `@example` build, so it is excluded here (same rationale as its removal from the
# anti-rot CI matrix). The file stays runnable locally; it is just not rendered/executed in CI docs.

isdir(GENDIR) && rm(GENDIR; recursive = true)
mkpath(GENDIR)
# The GP-UDE examples are each a real through-solver training (minutes apiece) and are already
# executed + asserted in the `docs-examples` anti-rot CI jobs. Rendering them as STATIC code blocks
# here (rather than executed `@example` blocks) keeps the docs build fast — only the cheap Capability-A
# examples execute — while still publishing their narrative + code. Re-executing them in the docs build
# would re-run every training sequentially (~35+ min) for no extra verification.
for (_, src) in EXAMPLES
    if startswith(basename(src), "gp_ude_")
        # static ```julia blocks — rendered, not executed
        Literate.markdown(joinpath(EXDIR, src), GENDIR; documenter = true, codefence = "```julia" => "```")
    else
        # Capability-A examples: Literate's default executed `@example` blocks (cheap)
        Literate.markdown(joinpath(EXDIR, src), GENDIR; documenter = true)
    end
end

makedocs(
    sitename = "Magpie.jl",
    modules = [Magpie],
    authors = "Jonathan Chen <jonathanwchen@pm.me> and contributors",
    pages = [
        "Home" => "index.md",
        "Examples" => [title => "generated/$(first(splitext(src))).md" for (title, src) in EXAMPLES],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    warnonly = [:missing_docs],
)

deploydocs(; repo = "github.com/jowch/Magpie.jl.git", push_preview = true)
