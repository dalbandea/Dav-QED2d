using QED2d
using CUDA
using Random

# Lattice and Zolotarev parameters
lsize = 20          # lattice size
lbeta = 5.00        # beta
am0 = 10.0          # bare mass
n_rhmc = 5          # number of Zolotarev monomial pairs

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)

lambda_min, lambda_max = power_method(U, am0, prm, kprm)

# Generate Zolotarev parameters
r_a_rhmc = lambda_min |> real |> x->0.8*x |> sqrt  
r_b_rhmc = lambda_max |> real |> x->1.2*x |> sqrt
rprm = get_rhmc_params(n_rhmc, r_a_rhmc, r_b_rhmc)

CGmaxiter = 10000
CGtol = 1e-25

# Generate random pseudofermionic field
X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
F = CUDA.zeros(ComplexF64,prm.iL[1], prm.iL[2], 2)

# Apply X - D†DR²X test (see Luscher rhmc ec. 4.3)
R(F, U, X, am0, CGmaxiter, CGtol, gamm5Dw_sqr_musq, rprm, prm, kprm)
tmp = copy(F)
R(F, U, tmp, am0, CGmaxiter, CGtol, gamm5Dw_sqr_musq, rprm, prm, kprm)
X_f = similar(X)
CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(X_f, U, F, am0, prm)
tmp2 = copy(X_f)
CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(X_f, U, tmp2, am0, prm)
deviation = X - X_f
delta_rhmc = delta(n_rhmc, rprm.eps) |> (x -> x*(2+x)) # maximum error
delta_X = sqrt(CUDA.dot(deviation, deviation))/sqrt(CUDA.dot(X,X))
# Se debe cumplir que delta_X ≤ delta_rhmc.
@test real(delta_X) < delta_rhmc
