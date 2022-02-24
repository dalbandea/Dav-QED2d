# using CUDA, Logging, StructArrays, Random, DelimitedFiles, Elliptic, Elliptic.Jacobi, LinearAlgebra
using Revise
using Pkg
Pkg.activate(".")
using QED2d
using CUDA, Logging, StructArrays, Random, DelimitedFiles, LinearAlgebra

# Lattice and Zolotarev parameters
lsize = 20          # lattice size
lbeta = 2.00        # beta
am0 = [0.6, 0.6]         # bare mass
read_from = 0

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

# U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
if read_from != 0
	gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(read_from)"
	load_gauge(U, gauge_file, prm)
end


lambda_min, lambda_max = power_method(U, am0[1], prm, kprm, iter=10000)    # Apply power method
                                                            # to extract maximum
                                                            # and minimum
                                                            # eigenvalues of
                                                            # D^†D

# Generate Zolotarev parameters
n_rhmc = 3                                              # number of Zolotarev
                                                        # monomial pairs
r_a_rhmc = 0.45 |> real |> x->x*1.0 |> sqrt  
r_b_rhmc = 22.0 |> real |> x->x*1.0 |> sqrt             # eps_rhmc is defined
                                                        # such that r_a and r_b
                                                        # are the sqrt of
                                                        # minimum and maximum
                                                        # eigenvalues of D^†D.
rprm = get_rhmc_params([n_rhmc, n_rhmc], [r_a_rhmc, r_a_rhmc], [r_b_rhmc, r_b_rhmc])

acc = Vector{Int64}()
reweight = Vector{Float64}()
plaqs = Vector{Float64}()
qtops = Vector{Float64}()


nsteps  = 15
epsilon = 1.0/nsteps
CGmaxiter = 10000
CGtol = 1e-16


@time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false)

for i in 1:20
    @time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false)
    Plaquette(U, prm, kprm) |> plaq_U -> push!(plaqs, plaq_U)
	Qtop(U, prm, kprm)      |> qtop_U -> push!(qtops, qtop_U)
    println("Last plaquette: $(plaqs[end])")
    println("Last Q: $(qtops[end])")
    # add reweighting factor
    reweighting_factor(U, am0, CGmaxiter, CGtol, prm, kprm, rprm) |> x -> push!(reweight, x)
    lambda_min, lambda_max = power_method(U, am0[1], prm, kprm, iter=5000)
    println("λ_min = $lambda_min, \nλ_max = $lambda_max")
end
