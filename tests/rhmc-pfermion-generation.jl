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
n_rhmc = 5          # number of Zolotarev monomial pairs

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
    nu_rhmc[j] = nu(j,n_rhmc,eps_rhmc, r_b_rhmc)
    mu_rhmc[j] = mu(j,n_rhmc,eps_rhmc, r_b_rhmc)
    rho_rhmc[j] = rho_mu(j,1,n_rhmc,n_rhmc,eps_rhmc, r_b_rhmc)
end
rprm = RHMCParm(r_b_rhmc, n_rhmc, eps_rhmc, A_rhmc, rho_rhmc, mu_rhmc, nu_rhmc)

# Vector of n_rhmc pseudofermion fields
F = Array{CuArray}(undef, n_rhmc)

# Check that X†X = ξ†ϕᵢ for each pseudofermion ϕᵢ
prod_in = 0.0       # X†X
prod_out = 0.0      # ξ†ϕᵢ

for i in 1:n_rhmc
    # Generate random pseudofermionic field
    X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
    prod_in += CUDA.dot(X,X)            # Store action of pseudofermion i, X†X
    # Initialize pseudofermion i
    F[i] = CUDA.zeros(ComplexF64,prm.iL[1], prm.iL[2], 2)

    # ϕᵢ = (γD+iμ)X
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[i], U, X, am0, prm)
    F[i] .= F[i] .+ im*rprm.mu[i] .* X

    # ϕᵢ = (D†D+ν²)ϕᵢ
    aux_F = copy(F[i])
    CG(F[i], U, aux_F, am0, rprm.nu[i], 100000, 0.00000000000000000001, gamm5Dw_sqr_musq, prm, kprm)

    # ϕᵢ = (γD-iν)ϕᵢ
    aux_F .= F[i]
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[i], U, aux_F, am0, prm)
    F[i] .= F[i] .- im*rprm.nu[i] .* aux_F

    # ξ = (D†D+μ²)⁻¹(D†D+ν²)ϕᵢ
    xi = copy(F[i])
    tmp = copy(F[i])
    gamm5Dw_sqr_musq(xi, tmp, U, F[i], am0, rprm.nu[i], prm, kprm)
    tmp .= xi
    CG(xi, U, tmp, am0, rprm.mu[i], 100000, 0.00000000000000000001, gamm5Dw_sqr_musq, prm, kprm)
    prod_out += CUDA.dot(xi, F[i])      # Store action of pseudofermion i, ξ†ϕᵢ
end
