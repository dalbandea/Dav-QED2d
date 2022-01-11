using CUDA, Logging, StructArrays, Random, DelimitedFiles, Elliptic, Elliptic.Jacobi, LinearAlgebra
using Revise
using Pkg
Pkg.activate(".")
using QED2d

# Lattice and Zolotarev parameters
lsize = 20          # lattice size
lbeta = 5.00        # beta

am0 = 0.2           # bare mass
n_rhmc = 5          # number of Zolotarev monomial pairs
read_from = 0


global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

# U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
if read_from != 0
	gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(read_from)"
	load_gauge(U, gauge_file, prm)
end


lambda_min, lambda_max = power_method(U, am0)   # Apply power method to extract
                                                # maximum and minimum
                                                # eigenvalues of D^†D

# Generate Zolotarev parameters
r_a_rhmc = 80 |> real |> x->round(x)-1 |> sqrt  
r_b_rhmc = 220 |> real |> x->round(x)+1 |> sqrt         # eps_rhmc is defined
                                                        # such that r_a and r_b
                                                        # are the sqrt of
                                                        # minimum and maximum
                                                        # eigenvalues of D^†D.
eps_rhmc = ( r_a_rhmc/r_b_rhmc )^2
mu_rhmc = Array{Float64}(undef, n_rhmc)
nu_rhmc = Array{Float64}(undef, n_rhmc)
rho_rhmc = Array{Float64}(undef, n_rhmc)
A_rhmc = A(n_rhmc,eps_rhmc)

for j in 1:n_rhmc
    mu_rhmc[j] = mu(j,n_rhmc,eps_rhmc, r_b_rhmc)
    nu_rhmc[j] = nu(j,n_rhmc,eps_rhmc, r_b_rhmc)
    rho_rhmc[j] = rho_mu(j,1,n_rhmc,n_rhmc,eps_rhmc, r_b_rhmc)
end
rprm = RHMCParm(r_b_rhmc, n_rhmc, eps_rhmc, A_rhmc, rho_rhmc, mu_rhmc, nu_rhmc)


acc = Vector{Int64}()
plaqs = Vector{Float64}()
qtops = Vector{Float64}()


nsteps  = 100
epsilon = 1.0/nsteps
CGmaxiter = 10000
CGtol = 1e-16

@time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false)

for i in 1:100
    @time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false)
	plaq = Plaquette(U, prm, kprm)
    push!(plaqs, plaq)
	qtop = Qtop(U, prm, kprm)
    push!(qtops, qtop)
    println("Last plaquette: $plaq")
    println("Last Q: $qtop")
end
