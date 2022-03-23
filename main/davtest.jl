using CUDA, Logging, StructArrays, Random, DelimitedFiles
using Revise
using Pkg
Pkg.activate(".")
using QED2d

# Lattice and Zolotarev parameters
lsize = 20          # lattice size
lbeta = 2.00        # beta
am0 = 0.6         # bare mass

prm  = LattParm((lsize,lsize), lbeta)
kprm = KernelParm((lsize, 1), (1,lsize))

U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)

acc = Vector{Int64}()
nsteps  = 10
epsilon = 1.0/nsteps
CGmaxiter = 10000
CGtol = 1e-16

@time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, integrator = Leapfrog())

# for i in 1:1000
#        @time HMC!(U, 0.0, 0.0005, 20, Vector{Int64}(), 10000, prm, kprm, qzero=false)
#            println("   Plaquette: ", Plaquette(U, prm, kprm))
#            println("   Qtop:      ", Qtop(U, prm, kprm))
# end

