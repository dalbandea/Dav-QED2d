module QED2d

import CUDA

struct LattParm
    iL::Tuple{Int64,Int64}
    beta::Float64
end

struct KernelParm
    threads::Tuple{Int64,Int64}
    blocks::Tuple{Int64,Int64}
end
export LattParm, KernelParm

include("QED2dAction.jl")
export plaquette!, qtop!, updt_fld!

include("QED2dHMC.jl")
export HMC!, Hamiltonian, Action, Plaquette, Qtop, OMF4!

include("QED2dFields.jl")
export unfold_fld!

end # module
