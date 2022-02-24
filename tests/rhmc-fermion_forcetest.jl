using CUDA, Logging, StructArrays, Random, DelimitedFiles, Elliptic, Elliptic.Jacobi, LinearAlgebra
using Revise
using Pkg
Pkg.activate(".")
using QED2d
# CUDA.allowscalar(false)
CUDA.allowscalar(true)

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


lambda_min, lambda_max = power_method(U, am0, prm, kprm)    # Apply power method
                                                            # to extract maximum
                                                            # and minimum
                                                            # eigenvalues of
                                                            # D^†D

# Generate Zolotarev parameters
n_rhmc = 5                                              # number of Zolotarev
                                                        # monomial pairs
r_a_rhmc = lambda_min |> real |> x->x*1.0 |> sqrt  
r_b_rhmc = lambda_max |> real |> x->x*1.0 |> sqrt       # eps_rhmc is defined
                                                        # such that r_a and r_b
                                                        # are the sqrt of
                                                        # minimum and maximum
                                                        # eigenvalues of D^†D.
rprm = get_rhmc_params(n_rhmc, r_a_rhmc, r_b_rhmc)

# Vector of n_rhmc pseudofermion fields
F = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)


# Point to modify
link_x = 2  # position x of link
link_y = 5 # position y of link
link_dir = 1 # direction of link

# Compute initial action
X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2).*(im))/sqrt(2)
# Sini = CUDA.dot(X,X) + Action(U, prm, kprm)
Sini = CUDA.dot(X,X)
println("Initial action: ", Sini)

# Get pseudofermion field
CGmaxiter=10000
CGtol=1e-16
generate_pseudofermion!(F, U, X, am0, CGmaxiter, CGtol, prm, kprm, rprm)

# Compute the analytical force
g5DX = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
frc = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
frc_i = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
Frc = similar(frc)

# Fill fermion force in frc
for i in 1:rprm.n
    CG(X, U, F, am0, rprm.mu[i], CGmaxiter, CGtol, gamm5Dw_sqr_musq, prm, kprm)
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
    end
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc_i, U, X, g5DX, prm)
    end
    frc .= frc .+ rprm.rho[i]*frc_i
end

# Fill gauge force in frc1 and frc2
CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
end

# Frc .= frc1 .+ frc2 .+ frc
Frc .= frc
Frc_i = Frc[link_x, link_y, link_dir]

# Duplicate configuration U and modify one link
Deps = 0.000001

U2 = similar(U)
U2 .= U
U2[link_x, link_y, link_dir] = Complex(CUDA.cos(Deps), CUDA.sin(Deps))*U2[link_x, link_y, link_dir]

# Inversion of Dirac operator
# ξⱼ = ∏(D†D+μ²)⁻¹(D†D+ν²)ϕⱼ = ( 1+∑ρᵢ/(DD†+μ²) ) ϕⱼ =
# MultiCG(ξⱼ,U,ϕⱼ,...)
xi = copy(F)
MultiCG(xi, U2, F, am0, CGmaxiter, CGtol, gamm5Dw_sqr_musq, rprm, prm, kprm)

# Compute final action
# Sfin = CUDA.dot(xi,F[i]) + Action(U2, prm, kprm)
Sfin = CUDA.dot(xi,F)

# Compute numerical force
F_num = (Sfin - Sini)/Deps

println(Frc_i+F_num)


println("Numerical: ", F_num)
println("Analytic: ", Frc_i)
