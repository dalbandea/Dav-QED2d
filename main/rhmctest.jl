using CUDA, Logging, StructArrays, Random, DelimitedFiles, Elliptic, Elliptic.Jacobi, LinearAlgebra
using Revise
using Pkg
Pkg.activate(".")
using QED2d

prm  = LattParm((20,20), 6.05)
kprm = KernelParm((20, 1), (1,20))
am0 = 10.0
read_from = 0

if read_from != 0
	gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(read_from)"
	load_gauge(U, gauge_file, prm)
end

r_a_rhmc = 98 |> sqrt
r_b_rhmc = 201 |> sqrt
n_rhmc = 3
eps_rhmc = ( r_a_rhmc/r_b_rhmc )^2
mu_rhmc = Array{Float64}(undef, n_rhmc)
rho_rhmc = Array{Float64}(undef, n_rhmc)
A_rhmc = A(n_rhmc,eps_rhmc)

for j in 1:n_rhmc
	mu_rhmc[j] = mu(j,n_rhmc,eps_rhmc, r_b_rhmc)
	rho_rhmc[j] = rho_mu(j,1,n_rhmc,n_rhmc,eps_rhmc, r_b_rhmc)
end

rprm = RHMCParm(r_b_rhmc, n_rhmc, eps_rhmc, A_rhmc, rho_rhmc, mu_rhmc)

X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
# X = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2) |> (x -> (x + 2*x*im)/sqrt(2))
println(CUDA.dot(X,X))
F = CUDA.zeros(ComplexF64,prm.iL[1], prm.iL[2], 2)

U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
# U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)


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


# Se debe cumplir que delta_X ≤ delta_rhmc. Comprobar que X_f no haya cambiado
# de signo con respecto a X para hacer este check


# Power method para autovalor máximo y mínimo

b = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)

shift = 15 # this shift helps convergence

for i in 1:1000
	b_aux = copy(b)
	gamm5Dw_sqr(b_aux, U, b, am0, prm, kprm)
	# gamm5Dw_sqr_sqr(b_aux, U, b, am0, prm, kprm)
	# CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(b_aux, U, b, am0, prm)
	# Dw(b_aux, U, b, am0, prm, kprm)
	# MultiCG(b_aux, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
	b_aux .= b_aux .+ shift*b
	global b = b_aux/CUDA.dot(b_aux,b_aux)
	# println("Segon: $(b[1,1,1])")
end
bnext = copy(b)
gamm5Dw_sqr(bnext, U, b, am0, prm, kprm)
# gamm5Dw_sqr_sqr(bnext, U, b, am0, prm, kprm)
# CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(bnext, U, b, am0, prm)
# Dw(bnext, U, b, am0, prm, kprm)
# MultiCG(bnext, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
bnext .= bnext .+ shift*b
lambda_max = CUDA.dot(b,bnext)/CUDA.dot(b,b) - shift

# lambda_max = CUDA.dot(b,bnext)/CUDA.dot(b,b)

for i in 1:1000
	b_last = copy(b)
	gamm5Dw_sqr(b, U, b, am0, prm, kprm)
	# gamm5Dw_sqr_sqr(b, U, b, am0, prm, kprm)
	# CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(b, U, b, am0, prm)
	# Dw(b, U, b, am0, prm, kprm)
	# MultiCG(b, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
	b .= b .- lambda_max*b_last
	global b = b/CUDA.dot(b,b)
end
bnext = copy(b)
gamm5Dw_sqr(bnext, U, b, am0, prm, kprm)
# gamm5Dw_sqr_sqr(bnext, U, b, am0, prm, kprm)
# CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(bnext, U, b, am0, prm)
# Dw(bnext, U, b, am0, prm, kprm)
# MultiCG(bnext, U, b, am0, 100000, 0.0000000000000000000001, gamm5Dw_sqr_musq, rprm, prm, kprm)
bnext .= bnext .- lambda_max*b
lambda_min = CUDA.dot(b,bnext)/CUDA.dot(b,b) + lambda_max


# Plane wave eigenvalues

b = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)

CUDA.@cuda threads=kprm.threads blocks=kprm.blocks plwv(b, (2*pi*0/20, 2*pi*5/20), prm)

# CUDA.@cuda threads=kprm.threads blocks=kprm.blocks plwv(b, (0.0, 0.0), prm)

bnext = copy(b)
Dw(bnext, U, b, am0, prm, kprm)
lambda = CUDA.dot(b,bnext)/CUDA.dot(b,b)
resres = bnext/lambda - b
