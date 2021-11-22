using CUDA, Logging, StructArrays, Random, DelimitedFiles
using Revise
using Pkg
Pkg.activate(".")
using QED2d
# CUDA.allowscalar(false)
CUDA.allowscalar(true)

prm  = LattParm((50,50), 6.05)
kprm = KernelParm((50, 1), (1,50))
am0 = 0.01
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
link_x = 20  # position x of link
link_y = 10 # position y of link
link_dir = 2 # direction of link

# Compute initial action
X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2).*(im))/sqrt(2)
# X_h = zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
# X2 = DelimitedFiles.readdlm("/home/david/test/schwinger/X.txt")[:,2]
# X1_re = DelimitedFiles.readdlm("/home/david/test/schwinger/X.txt")[begin:2:end,1]
# X1_re = reshape( X1_re, (prm.iL[1], prm.iL[2]) )
# X1_im = DelimitedFiles.readdlm("/home/david/test/schwinger/X.txt")[begin:2:end,2]
# X1_im = reshape( X1_im, (prm.iL[1], prm.iL[2]) )
# X2_re = DelimitedFiles.readdlm("/home/david/test/schwinger/X.txt")[2:2:end,1]
# X2_re = reshape( X2_re, (prm.iL[1], prm.iL[2]) )
# X2_im = DelimitedFiles.readdlm("/home/david/test/schwinger/X.txt")[2:2:end,2]
# X2_im = reshape( X2_im, (prm.iL[1], prm.iL[2]) )
# X_h[:,:,1] = X1_re .+ X1_im*im
# X_h[:,:,2] = X2_re .+ X2_im*im
# CUDA.copyto!(X, X_h)
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

Frc .= frc1 .+ frc2 .+ frc
Frc_i = Frc[link_x, link_y, link_dir]

# Duplicate configuration U and modify one link
Deps = 0.0001

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




