using CUDA, Logging, StructArrays, Random, DelimitedFiles
using Revise
using Pkg
Pkg.activate(".")
using QED2d

prm  = LattParm((40,40), 5.0)
kprm = KernelParm((40, 1), (1,40))
am0 = 10.0
file = "statistics.txt"
read_from = 0


print("Allocating gauge field...")
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
println("DONE")

Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

acc = Vector{Int64}()
plaqs = Vector{Float64}()
nsteps  = 100
epsilon = 1.0/nsteps
CGmaxiter = 10000
CGtol = 1e-16

if read_from == 0
	@time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, qzero=false)
	# println("   Plaquette: ", Plaquette(U, prm, kprm))
	# println("   Qtop:      ", Qtop(U, prm, kprm))
	write(open(file, "w"), "$(Plaquette(U, prm, kprm)) $(Qtop(U, prm, kprm))\n")
else
	gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(read_from)"
	load_gauge(U, gauge_file, prm)
end

for i in (read_from+1):10
       # @time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, prm, kprm, qzero=false)
       @time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, qzero=false)
        # println("   Plaquette: ", Plaquette(U, prm, kprm))
        # println("   Qtop:      ", Qtop(U, prm, kprm))
	plaq = Plaquette(U, prm, kprm)
    push!(plaqs, plaq)
    println("Last plaquette: $plaq")
	# qtop = Qtop(U, prm, kprm)
	# write(open(file, "a"), "$(plaq) $(qtop)\n")
	# local gauge_file = "configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(i)"
	# save_gauge(U, gauge_file, prm)
end
