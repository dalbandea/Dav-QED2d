using CUDA, Logging, StructArrays, Random, DelimitedFiles
using Revise
using Pkg
Pkg.activate(".")
using QED2d

prm  = LattParm((10,10), 6.05)
kprm = KernelParm((10, 1), (1,10))
am0 = 10.0
file = "statistics.txt"
read_from = 0


print("Allocating gauge field...")
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
println("DONE")

Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

acc = Vector{Int64}()
ns  = 100
eps = 1.0/ns
CGmaxiter = 10000
CGtol = 1e-16

if read_from == 0
	@time HMC!(U, am0, eps, ns, acc, CGmaxiter, CGtol, prm, kprm, qzero=false)
	# println("   Plaquette: ", Plaquette(U, prm, kprm))
	# println("   Qtop:      ", Qtop(U, prm, kprm))
	write(open(file, "w"), "$(Plaquette(U, prm, kprm)) $(Qtop(U, prm, kprm))\n")
else
	gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(read_from)"
	load_gauge(U, gauge_file, prm)
end

for i in (read_from+1):1000
       # @time HMC!(U, am0, eps, ns, acc, CGmaxiter, prm, kprm, qzero=false)
       @time HMC!(U, am0, eps, ns, acc, CGmaxiter, CGtol, prm, kprm, qzero=false)
        # println("   Plaquette: ", Plaquette(U, prm, kprm))
        # println("   Qtop:      ", Qtop(U, prm, kprm))
	plaq = Plaquette(U, prm, kprm)
	qtop = Qtop(U, prm, kprm)
	write(open(file, "a"), "$(plaq) $(qtop)\n")
	local gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(i)"
	save_gauge(U, gauge_file, prm)
end
