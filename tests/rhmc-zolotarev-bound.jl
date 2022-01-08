using CUDA, Logging, StructArrays, Random, DelimitedFiles, Elliptic, Elliptic.Jacobi, LinearAlgebra
using Revise
using Pkg
Pkg.activate(".")
using QED2d

"""
    power_method(U, am0)

Given a gauge field `U` and a bare quark mass `am0`, return the maximum and
minimum eigenvalues of D^†D.

# Examples
```jldocs
lambda_min, lambda_max = power_method(U, am0)
```
"""
function power_method(U, am0)

    b = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2) # initial random fermionic field

    shift = 2     # this shift helps convergence

    # Apply recursively b = Ab.
    # To help convergence, apply instead b = (A + shift) b = Ab + shift b.
    # Then λ_max will be ⟨b|A|b⟩/⟨b|b⟩ - shift.
    for i in 1:1000
        b_aux = copy(b)
        gamm5Dw_sqr(b_aux, U, b, am0, prm, kprm)
        b_aux .= b_aux .+ shift*b
        b = b_aux/CUDA.dot(b_aux,b_aux)
    end
    bnext = copy(b)
    gamm5Dw_sqr(bnext, U, b, am0, prm, kprm)
    bnext .= bnext .+ shift*b
    lambda_max = CUDA.dot(b,bnext)/CUDA.dot(b,b) - shift

    # Apply recursively b = (A-λ_max I) b = Ab - λ_max b
    # Then λ_min will be ⟨b|A|b⟩/⟨b|b⟩ + λ_max
    for i in 1:1000
        b_last = copy(b)
        gamm5Dw_sqr(b, U, b, am0, prm, kprm)
        b .= b .- lambda_max*b_last
        b = b/CUDA.dot(b,b)
    end
    bnext = copy(b)
    gamm5Dw_sqr(bnext, U, b, am0, prm, kprm)
    bnext .= bnext .- lambda_max*b
    lambda_min = CUDA.dot(b,bnext)/CUDA.dot(b,b) + lambda_max

    return lambda_min, lambda_max

end

# Lattice and Zolotarev parameters
lsize = 20          # lattice size
lbeta = 6.05        # beta
am0 = 10.0          # bare mass
n_rhmc = 2          # number of Zolotarev monomial pairs

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
# U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)


lambda_min, lambda_max = power_method(U, am0)   # Apply power method to extract
                                                # maximum and minimum
                                                # eigenvalues of D^†D

# Generate Zolotarev parameters
r_a_rhmc = lambda_min |> real |> x->round(x)-1 |> sqrt  
r_b_rhmc = lambda_max |> real |> x->round(x)+1 |> sqrt  # eps_rhmc is defined
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


# Generate random pseudofermionic field
X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
# X = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2) |> (x -> (x + 2*x*im)/sqrt(2))
F = CUDA.zeros(ComplexF64,prm.iL[1], prm.iL[2], 2)


# Apply X - D†DR²X test (see Luscher rhmc ec. 4.3)
MultiCG(F, U, X, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
tmp = copy(F)
MultiCG(F, U, tmp, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
X_f = similar(X)
CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(X_f, U, F, am0, prm)
tmp2 = copy(X_f)
CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(X_f, U, tmp2, am0, prm)
deviation = X - X_f
delta_rhmc = delta(n_rhmc, eps_rhmc) |> (x -> x*(2+x)) # maximum error
delta_X = sqrt(CUDA.dot(deviation, deviation))/sqrt(CUDA.dot(X,X))
# Se debe cumplir que delta_X ≤ delta_rhmc.
real(delta_X) < delta_rhmc ? println("Test passed")  : println("Test not passed") 



############## END OF Z BOUND. BEGIN OF REWEIGHTING FACTOR W ##################


# Returns ZᵖX_in, with Z = D†DR² - I
function LuscherZp(Z, p::Int64, U, X_in, am0, rprm::RHMCParm, prm::LattParm, kprm::KernelParm)

    function LuscherZ(Z, U, X_in, am0, rprm::RHMCParm, prm::LattParm, kprm::KernelParm)
        # Z = D†DR² X_in
        MultiCG(Z, U, X_in, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
        tmp = copy(Z)
        MultiCG(Z, U, tmp, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
        tmp .= Z
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(Z, U, tmp, am0, prm)
        tmp .= Z
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(Z, U, tmp, am0, prm)
        # Z = Z - X_in = (D†DR² - I) X_in
        Z .= Z .- X_in
        return nothing
    end

    tmp = copy(X_in)
    # At the end, Z = (D†DR² - I)ᵖ X_in
    for i in 1:p
        LuscherZ(Z, U, tmp, am0, rprm, prm, kprm)
        tmp .= Z
    end

    return nothing

end

ZpX = copy(X)
power = 3
LuscherZp(ZpX,power,U,X,am0,rprm,prm,kprm)  # this returns ZᵖX in the first argument
XZpX_code = CUDA.dot(X,ZpX) |> real         # It must hold that (X,ZᵖX) ≤ 2N(2δ)ᵖ
XZpX_teo = 2*prm.iL[1]*prm.iL[2]*(2*delta_rhmc)^power
XZpX_code < XZpX_teo ? println("Test passed") : println("Test not passed")



# Compute reweighting factor W_N with N=1, eq. (4.1)

res = 0.0
Tfactor = 1/2   # Taylor factor of expansion (1+Z)^(-1/2)
ZpX = copy(X)
for i in 1:5
    LuscherZp(ZpX,i,U,X,am0,rprm,prm,kprm)  
    res += Tfactor * CUDA.dot(X,ZpX)
    Tfactor *= (-1)*(2*i+1)/2/factorial(i+1)*factorial(i)
end
W1 = exp(res)   # if 2Nδ≤0.01, W₁ is expected to deviate from 1 at most by 1%


############## SCRIPT CHECKED UNTIL HERE ##################



############# BEGIN NOTES ##############

# The test could be generalized so that `power_method` accepts any operator.
# This test will fail when I change the function MultiCG to other thing.

############# END NOTES  ###############






################# GARBAGE FROM HERE TO THE END #######################


# # Power method para autovalor máximo y mínimo

# b = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)

# shift = 2 # this shift helps convergence

# for i in 1:1000
#     b_aux = copy(b)
#     gamm5Dw_sqr(b_aux, U, b, am0, prm, kprm)
#     # gamm5Dw_sqr_sqr(b_aux, U, b, am0, prm, kprm)
#     # CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(b_aux, U, b, am0, prm)
#     # Dw(b_aux, U, b, am0, prm, kprm)
#     # MultiCG(b_aux, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
#     b_aux .= b_aux .+ shift*b
#     global b = b_aux/CUDA.dot(b_aux,b_aux)
#     # println("Segon: $(b[1,1,1])")
# end
# bnext = copy(b)
# gamm5Dw_sqr(bnext, U, b, am0, prm, kprm)
# # gamm5Dw_sqr_sqr(bnext, U, b, am0, prm, kprm)
# # CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(bnext, U, b, am0, prm)
# # Dw(bnext, U, b, am0, prm, kprm)
# # MultiCG(bnext, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
# bnext .= bnext .+ shift*b
# lambda_max = CUDA.dot(b,bnext)/CUDA.dot(b,b) - shift

# # lambda_max = CUDA.dot(b,bnext)/CUDA.dot(b,b)

# for i in 1:1000
#     b_last = copy(b)
#     gamm5Dw_sqr(b, U, b, am0, prm, kprm)
#     # gamm5Dw_sqr_sqr(b, U, b, am0, prm, kprm)
#     # CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(b, U, b, am0, prm)
#     # Dw(b, U, b, am0, prm, kprm)
#     # MultiCG(b, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
#     b .= b .- lambda_max*b_last
#     global b = b/CUDA.dot(b,b)
# end
# bnext = copy(b)
# gamm5Dw_sqr(bnext, U, b, am0, prm, kprm)
# # gamm5Dw_sqr_sqr(bnext, U, b, am0, prm, kprm)
# # CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(bnext, U, b, am0, prm)
# # Dw(bnext, U, b, am0, prm, kprm)
# # MultiCG(bnext, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
# bnext .= bnext .- lambda_max*b
# lambda_min = CUDA.dot(b,bnext)/CUDA.dot(b,b) + lambda_max
