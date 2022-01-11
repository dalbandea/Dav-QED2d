using CUDA, Logging, StructArrays, Random, DelimitedFiles, Elliptic, Elliptic.Jacobi, LinearAlgebra
using Revise
using Pkg
Pkg.activate(".")
using QED2d
# CUDA.allowscalar(false)
CUDA.allowscalar(true)

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
lsize = 10          # lattice size
lbeta = 6.05        # beta
am0 = 10.0          # bare mass
n_rhmc = 2          # number of Zolotarev monomial pairs

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

file = "statistics.txt"
read_from = 93

print("Allocating gauge field...")
# U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
println(" DONE")

Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

if read_from == 0
	println("Starting new run")
	# @time HMC!(U, am0, eps, ns, acc, CGmaxiter, CGtol, prm, kprm, qzero=false)
	# println("   Plaquette: ", Plaquette(U, prm, kprm))
	# println("   Qtop:      ", Qtop(U, prm, kprm))
	# write(open(file, "w"), "$(Plaquette(U, prm, kprm)) $(Qtop(U, prm, kprm))\n")
else
	gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(read_from)"
	load_gauge(U, gauge_file, prm)
end


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


# Point to modify
link_x = 2  # position x of link
link_y = 5 # position y of link
link_dir = 1 # direction of link

for i in 1:n_rhmc
    # Compute initial action
    X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2).*(im))/sqrt(2)
    # Sini = CUDA.dot(X,X) + Action(U, prm, kprm)
    Sini = CUDA.dot(X,X)
    println("Initial action: ", Sini)

    # Get pseudofermion field
    F[i] = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
    ### ϕᵢ = (γD+iμ)X
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[i], U, X, am0, prm)
    end
    F[i] .= F[i] .+ im*rprm.mu[i] .* X
    ### ϕᵢ = (D†D+ν²)⁻¹ϕᵢ
    aux_F = copy(F[i])
    CG(F[i], U, aux_F, am0, rprm.nu[i], 100000, 0.00000000000000000001, gamm5Dw_sqr_musq, prm, kprm)
    ### ϕᵢ = (γD-iν)ϕᵢ
    aux_F .= F[i]
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[i], U, aux_F, am0, prm)
    end
    F[i] .= F[i] .- im*rprm.nu[i] .* aux_F

    # Compute the analytical force
    g5DX = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
    frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
    frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
    frc = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
    Frc = similar(frc)
    CGmaxiter = 10000
    CGtol = 0.00000000000000001

    CG(X, U, F[i], am0, rprm.mu[i], CGmaxiter, CGtol, gamm5Dw_sqr_musq, prm, kprm)
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
    end
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
    end
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
    end

    # Frc .= frc1 .+ frc2 .+ frc
    Frc .= frc*(rprm.nu[i]^2-rprm.mu[i]^2)
    Frc_i = Frc[link_x, link_y, link_dir]

    # Duplicate configuration U and modify one link
    Deps = 0.000001

    U2 = similar(U)
    U2 .= U
    U2[link_x, link_y, link_dir] = Complex(CUDA.cos(Deps), CUDA.sin(Deps))*U2[link_x, link_y, link_dir]

    # Inversion of Dirac operator
    ## ξ = (D†D+μ²)⁻¹(D†D+ν²)ϕᵢ
    xi = copy(F[i])
    tmp = copy(F[i])
    gamm5Dw_sqr_musq(xi, tmp, U2, F[i], am0, rprm.nu[i], prm, kprm)
    tmp .= xi
    CG(xi, U2, tmp, am0, rprm.mu[i], 100000, 0.00000000000000000001, gamm5Dw_sqr_musq, prm, kprm)

    # Compute final action
    # Sfin = CUDA.dot(xi,F[i]) + Action(U2, prm, kprm)
    Sfin = CUDA.dot(xi,F[i])

    # Compute numerical force
    F_num = (Sfin - Sini)/Deps

    println(Frc_i+F_num)
end


println("Numerical: ", F_num)
println("Analytic: ", Frc_i)





