using CUDA, Logging, StructArrays, Random

io = open("log.txt", "w+")
global_logger(SimpleLogger(io))

import Pkg
# Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/QED2d")
# Pkg.activate("/home/david/src/alberto-qed2d.jl/")
Pkg.activate("/home/david/git/dalbandea/phd/codes/1-Schwinger-model/Dav-QED2d")
using QED2d

prm  = LattParm((90,90), 1.0)
kprm = KernelParm((90, 1), (1,90))

print("Allocating gauge field...")
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)


#a = ones(Float64, prm.iL[1], prm.iL[2], 2)
#b = zeros(Float64, prm.iL[1], prm.iL[2], 2)
#Ucpu = StructArray{ComplexF64}((a, b))
#U = replace_storage(CuArray, Ucpu)

Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)


println(" DONE")
print("Allocating momentum field...")
mom = CUDA.zeros(Float64,  prm.iL[1], prm.iL[2], 2)
println(" DONE")

acc = Vector{Int64}()
eps = 0.1
ns  = 10
@time HMC!(U, eps, ns, acc, prm, kprm, qzero=false)
for i in 1:10
    @time HMC!(U, eps, ns, acc, prm, kprm, qzero=false)
    println("   Plaquette: ", Plaquette(U, prm, kprm))
    println("   Qtop:      ", Qtop(U, prm, kprm))
    println("Pacc = $(count(acc.==1)/length(acc))")
end
