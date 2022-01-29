using QED2d
using CUDA
using Random

CUDA.allowscalar(true)

# Lattice and Zolotarev parameters
lsize = 10          # lattice size
lbeta = 6.05        # beta
am0 = 10.0          # bare mass
n_rhmc = 2          # number of Zolotarev monomial pairs

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

print("Allocating gauge field...")
U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
println(" DONE")

# Apply power method to extract maximum and minimum eigenvalues of D^†D
lambda_min, lambda_max = power_method(U, am0, prm, kprm)    

# Generate Zolotarev parameters.
# eps_rhmc is defined such that r_a and r_b are the sqrt of minimum and maximum
# eigenvalues of D^†D.
n_rhmc = 5
r_a_rhmc = lambda_min |> real |> x->x*1.0 |> sqrt  
r_b_rhmc = lambda_max |> real |> x->x*1.0 |> sqrt
rprm = get_rhmc_params(n_rhmc, r_a_rhmc, r_b_rhmc)

# Vector of n_rhmc pseudofermion fields
F = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)

# Point to modify
link_x = 2          # position x of link
link_y = 5          # position y of link
link_dir = 1        # direction of link

# Compute initial action
X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2).*(im))/sqrt(2)
Sini = CUDA.dot(X,X)

# Get pseudofermion field
CGmaxiter=10000
CGtol=1e-16
generate_pseudofermion!(F, U, X, am0, CGmaxiter, CGtol, prm, kprm, rprm)

# Compute the analytical force
g5DX = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
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

Frc .= frc
Frc_i = Frc[link_x, link_y, link_dir]

# Duplicate configuration U and modify one link
Deps = 0.000001

U2 = similar(U)
U2 .= U
U2[link_x, link_y, link_dir] = Complex(cos(Deps), sin(Deps))*U2[link_x, link_y, link_dir]

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

@test Frc_i+F_num |> abs < 1e-5
