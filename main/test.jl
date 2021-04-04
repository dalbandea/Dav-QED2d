using CUDA, Logging, StructArrays, Random

io = open("log.txt", "w+")
global_logger(SimpleLogger(io))

import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/QED2d")
using QED2d

prm  = LattParm((512,512), 5.25)
kprm = KernelParm((512, 1), (1,512))

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
eps = 0.025
ns  = 20
@time HMC!(U, eps, ns, acc, prm, kprm, qzero=true)
for i in 1:1000
    @time HMC!(U, eps, ns, acc, prm, kprm, qzero=true)
    println("   Plaquette: ", Plaquette(U, prm, kprm))
    println("   Qtop:      ", Qtop(U, prm, kprm))
    println("Pacc = $(count(acc.==1)/length(acc))")
end
