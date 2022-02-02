using QED2d
using CUDA
using Random

# Lattice and Zolotarev parameters
lsize = 20          # lattice size
lbeta = 5.00        # beta
am0 = 0.2          # bare mass
n_rhmc = 5          # number of Zolotarev monomial pairs

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)

lambda_min, lambda_max = power_method(U, am0, prm, kprm, iter=10000)

# Generate Zolotarev parameters
r_a_rhmc = lambda_min*0.99 |> real |> sqrt  
r_b_rhmc = lambda_max*1.01 |> real |> sqrt
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



#####################################
#  Statistical fluctuations of Z^p  #
#####################################

# Lattice and Zolotarev parameters
n_rhmc = 10          # number of Zolotarev monomial pairs

CGtol = 1e-25
CGmaxiter = 100000

# Generate Zolotarev parameters
rprm = get_rhmc_params(n_rhmc, r_a_rhmc, r_b_rhmc)

# Check (X, ZᵖX) ≤ 2N(2δ)ᵖ for p = 1,...,10. Luscher eq. (4.5), (4.6)
tmp .= X
ZpX  = similar(X)
for i in 1:10
    QED2d.LuscherZ(ZpX, U, tmp, am0, CGmaxiter, CGtol, rprm, prm, kprm)  
    Ztrace = CUDA.dot(X, ZpX) |> real
    @test abs(Ztrace) < 2 * prm.iL[1] * prm.iL[2] * (2 * rprm.delta)^i
    tmp .= ZpX
end



#####################################
#  Statistical fluctuations of W_N  #
#####################################

V = prm.iL[1] * prm.iL[2]
W_1 = 1.0 # reweighting factor W_N with N=1

for i in 1:10
    # Lattice and Zolotarev parameters
    global n_rhmc = i          # number of Zolotarev monomial pairs
    # Generate Zolotarev parameters
    global rprm = get_rhmc_params(n_rhmc, r_a_rhmc, r_b_rhmc)
    global W_1 = reweighting_factor(U, am0, CGmaxiter, CGtol, prm, kprm, rprm)
    @test exp(-2 * V * rprm.delta) < W_1 < exp(2 * V * rprm.delta)
end
