module QED2d

import CUDA, DelimitedFiles, Elliptic, Elliptic.Jacobi, Random

struct LattParm
    iL::Tuple{Int64,Int64}
    beta::Float64
end

struct KernelParm
    threads::Tuple{Int64,Int64}
    blocks::Tuple{Int64,Int64}
end

struct RHMCParm
	r_b::Float64
	n::Int64
	eps::Float64
	A::Float64
	rho::Vector{Float64}
	mu::Vector{Float64}
    nu::Vector{Float64}
    delta::Float64
    reweighting_N::Int64
    reweighting_Taylor::Int64
end
export LattParm, KernelParm, RHMCParm

include("integrators/integrators.jl")
export Integrators, Leapfrog, OMF4

include("QED2dAction.jl")
export plaquette!, qtop!, updt_fld!, krnl_force!

include("QED2dHMC.jl")
export HMC!, Hamiltonian, Action, Plaquette, Qtop, OMF4!, leapgrog!, update_momenta!

include("QED2dFields.jl")
export unfold_fld!, unfold_reflect!

include("QED2dDirac.jl")
export gamm5Dw, gamm5Dw_sqr, gamm5Dw_sqr_msq, tr_dQwdU, gamm5Dw_sqr_musq, gamm5Dw_sqr_sqr_musq, gamm5Dw_sqr_sqr, gamm5, Dw

include("QED2dSolver.jl")
export CG

include("QED2dLoadSave.jl")
export save_gauge, load_gauge

include("QED2dRHMC.jl")
export k, v, a_r, c_r, mu, nu, rho_mu, P, d, A, delta, error, get_rhmc_params, power_method, reweighting_factor, R, MultiCG, generate_pseudofermion!

include("QED2dUtils.jl")
export plwv

end # module
