# using CUDA, Logging, StructArrays, Random, DelimitedFiles, Elliptic, Elliptic.Jacobi, LinearAlgebra
using CUDA, Logging, StructArrays, Random, DelimitedFiles, LinearAlgebra
import Elliptic, Elliptic.Jacobi
using Revise
using Pkg
Pkg.activate(".")
using QED2d

# Lattice and Zolotarev parameters
lsize = 20          # lattice size
lbeta = 5.00        # beta
am0 = 0.2           # bare mass
read_from = 0


global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

# U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
if read_from != 0
	gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(read_from)"
	load_gauge(U, gauge_file, prm)
end


lambda_min, lambda_max = power_method(U, am0, prm, kprm)    # Apply power method
                                                            # to extract maximum
                                                            # and minimum
                                                            # eigenvalues of
                                                            # D^†D

# Generate Zolotarev parameters
n_rhmc = 5                                              # number of Zolotarev
                                                        # monomial pairs
r_a_rhmc = 0.06 |> real |> x->x*1.0 |> sqrt  
r_b_rhmc = 20.0 |> real |> x->x*1.0 |> sqrt             # eps_rhmc is defined
                                                        # such that r_a and r_b
                                                        # are the sqrt of
                                                        # minimum and maximum
                                                        # eigenvalues of D^†D.
rprm = get_rhmc_params(n_rhmc, r_a_rhmc, r_b_rhmc)

acc = Vector{Int64}()
reweight = Vector{Float64}()
plaqs = Vector{Float64}()
qtops = Vector{Float64}()


nsteps  = 100
epsilon = 1.0/nsteps
CGmaxiter = 10000
CGtol = 1e-16

@time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false, so_as_guess=true)

for i in 1:5
    @time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false, so_as_guess=true)
    Plaquette(U, prm, kprm) |> plaq_U -> push!(plaqs, plaq_U)
	Qtop(U, prm, kprm)      |> qtop_U -> push!(qtops, qtop_U)
    # println("Last plaquette: $(plaqs[end])")
    # println("Last Q: $(qtops[end])")
    # add reweighting factor
    if(acc[end] == 0)
        push!(reweight, reweight[end])
    else
        reweighting_factor(U, am0, prm, kprm, rprm) |> x -> push!(reweight, x)
    end
    # lambda_min, lambda_max = power_method(U, am0, prm, kprm)
    # println("λ_min = $lambda_min, \nλ_max = $lambda_max")
end
