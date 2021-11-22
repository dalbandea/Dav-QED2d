using CUDA, Logging, StructArrays, Random, DelimitedFiles
using Revise
using Pkg
Pkg.activate(".")
using QED2d
CUDA.allowscalar(false)

prm  = LattParm((50,50), 6.05)
kprm = KernelParm((50, 1), (1,50))
am0 = 0.01
file = "statistics.txt"
read_from = 193


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


# Compute initial action
Sini = Action(U, prm, kprm)

# Duplicate configuration U and modify one link
Deps = 0.0000001

U2 = similar(U)
U2 .= U
U2_h = ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
CUDA.copyto!(U2_h, U2)
U2_h[1,10,2] = Complex(cos(Deps), sin(Deps))*U2_h[1,10,2]
CUDA.copyto!(U2, U2_h)

# Compute final action
Sfin = Action(U2, prm, kprm)

# Compute numerical force
F_num = (Sfin - Sini)/Deps

# Compute the analytical force
frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
F = similar(U)
F_h = zeros(Float64, prm.iL[1], prm.iL[2], 2)

CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
end

F .= frc1 .+ frc2

# Force at link i
CUDA.copyto!(F_h, F)
F_i = F_h[1,10,2]

println("Numerical: ", F_num)
println("Analytic: ", F_i)
