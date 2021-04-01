using CUDA, Logging

io = open("log.txt", "w+")
global_logger(SimpleLogger(io))

import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/QED2d")
using QED2d

prm  = LattParm((256,256), 11.25)
kprm = KernelParm((256, 1), (1,256))

print("Allocating gauge field...")
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
println(" DONE")
print("Allocating momentum field...")
mom = CUDA.zeros(Float64,  prm.iL[1], prm.iL[2], 2)
println(" DONE")

eps = 0.05
ns  = 20
@time HMC!(mom, U, eps, ns, prm, kprm)
for i in 1:1000
    @time HMC!(mom, U, eps, ns, prm, kprm)
    println("   Plaquette: ", Plaquette(U, prm, kprm))
    println("   Qtop:      ", Qtop(U, prm, kprm))
end
