ENV["GKSwstype"] = "100"  # GR renders headlessly (no display) in CI / doc build
using Documenter, Literate, Magpie

const EXDIR = joinpath(@__DIR__, "..", "examples")
const GENDIR = joinpath(@__DIR__, "src", "generated")

# Example pages (Literate source → generated page); add new examples here.
const EXAMPLES = Tuple{String, String}[]

push!(EXAMPLES, ("Level-set recovery (Straddle)", "levelset_straddle.jl"))
push!(EXAMPLES, ("Decision boundary (BinaryBALD)", "bald_classification.jl"))
push!(EXAMPLES, ("Critical-point survey (derivative GP)", "critical_point_survey.jl"))

isdir(GENDIR) && rm(GENDIR; recursive = true)
mkpath(GENDIR)
for (_, src) in EXAMPLES
    Literate.markdown(joinpath(EXDIR, src), GENDIR; documenter = true)  # markdown only at v0.1
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
