using CUDA, Logging, StructArrays, Random, DelimitedFiles
using Revise
using Pkg
Pkg.activate(".")
using QED2d

prm  = LattParm((50,50), 5.0)
kprm = KernelParm((50, 1), (1,50))
am0 = 0.1

n_rhmc = 6
eps_rhmc = 0.02
mu_rhmc = Array{Float64}(undef, n_rhmc)
rho_rhmc = Array{Float64}(undef, n_rhmc)
A_rhmc = A(n_rhmc,eps_rhmc)

for j in 1:n_rhmc
	mu_rhmc[j] = mu(j,n_rhmc,eps_rhmc)
	rho_rhmc[j] = rho_mu(j,1,n_rhmc,n_rhmc,eps_rhmc)
end

rprm = RHMCParm(n_rhmc, eps_rhmc, A_rhmc, rho_rhmc, mu_rhmc)

X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
println(CUDA.dot(X,X))
F = CUDA.zeros(ComplexF64,prm.iL[1], prm.iL[2], 2)
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)

CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F, U, X, am0, prm)

CG(X, U, F, am0, 100000, 0.000000000000000000000001, gamm5Dw_sqr_msq, prm, kprm)
println(CUDA.dot(X,F))

MultiCG(X, U, F, am0, 100000, 0.0000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
# for i in 1:1000
#        @time HMC!(U, 0.0, 0.0005, 20, Vector{Int64}(), 10000, prm, kprm, qzero=false)
#            println("   Plaquette: ", Plaquette(U, prm, kprm))
#            println("   Qtop:      ", Qtop(U, prm, kprm))
# end
