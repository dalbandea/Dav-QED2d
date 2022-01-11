using CUDA, Logging, StructArrays, Random, DelimitedFiles
using Revise
using Pkg
Pkg.activate(".")
using QED2d
# CUDA.allowscalar(false)
CUDA.allowscalar(true)

prm  = LattParm((10,10), 6.05)
kprm = KernelParm((10, 1), (1,10))
am0 = 10.0
file = "statistics.txt"
read_from = 93


print("Allocating gauge field...")
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
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

# Point to modify
link_x = 8  # position x of link
link_y = 4 # position y of link
link_dir = 2 # direction of link

# Compute initial action
X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2).*(im))/sqrt(2)
Sini = CUDA.dot(X,X) + Action(U, prm, kprm)
println("Initial action: ", Action(U, prm, kprm))

# Get pseudofermion field
F = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
CUDA.@sync begin
	CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F, U, X, am0, prm)
end

# Compute the analytical force
g5DX = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
frc = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
Frc = similar(frc)
CGmaxiter = 10000
CGtol = 0.00000000000000001

CG(X, U, F, am0, CGmaxiter, CGtol, gamm5Dw_sqr_msq, prm, kprm)
CUDA.@sync begin
	CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
end
CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
end
CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
end

CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
end

CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU_dav(frc, U, X, g5DX, prm)
end

Frc .= frc1 .+ frc2 .+ frc
Frc_i = Frc[link_x, link_y, link_dir]

# Duplicate configuration U and modify one link
Deps = 0.000001

U2 = similar(U)
U2 .= U
U2[link_x, link_y, link_dir] = Complex(CUDA.cos(Deps), CUDA.sin(Deps))*U2[link_x, link_y, link_dir]

# Inversion of Dirac operator
eta = similar(X)
CG(eta, U2, F, am0, CGmaxiter, CGtol, gamm5Dw_sqr_msq, prm, kprm)

# Compute final action
Sfin = CUDA.dot(eta,F) + Action(U2, prm, kprm)

# Compute numerical force
F_num = (Sfin - Sini)/Deps


println("Numerical: ", F_num)
println("Analytic: ", Frc_i)




