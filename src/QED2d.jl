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
export unfold_fld!, unfold_reflect!

include("QED2dDirac.jl")
export gamm5Dw, gamm5Dw_sqr, gamm5Dw_sqr_msq, tr_dQwdU

include("QED2dSolver.jl")
export CG

end # module
