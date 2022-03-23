abstract type Integrators end
struct Leapfrog <: Integrators end
struct OMF4 <: Integrators end

# struct leapfrog <: Integrators
#     AD_tape::Union{Nothing, ReverseDiff.CompiledTape} 
# end

# function leapfrog(; AD_tape = nothing)
#     leapfrog(AD_tape)
# end
