using CUDA, Logging, StructArrays, Random, DelimitedFiles, LinearAlgebra
using Revise
using Pkg
Pkg.activate(".")
using QED2d

# Lattice parameters
lsize = 20          # lattice size
lbeta = 6.05        # beta
am0 = 10.0          # bare mass

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

# Identity gauge field
U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)

# Initialize fermion field
b = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)

# Construct plane wave eigenvalue solution with k₁,₂ = 2πn/L
k_1 = 2*pi*5/20
k_2 = 2*pi*5/20
CUDA.@cuda threads=kprm.threads blocks=kprm.blocks plwv(b, (k_1, k_2), prm)

# Determine if planewave is eigenvector of D
bnext = copy(b)
Dw(bnext, U, b, am0, prm, kprm)             # apply b_next = Db
lambda = CUDA.dot(b,bnext)/CUDA.dot(b,b)    # check eigenvalue λ of b
resres = bnext/lambda - b                   # shouled be 0 if b is eigenvector

# Check with analytic results. λ should be λ₁ or λ₂ depending on the structure
# defined in the function plwv
lambda_1 = am0 + 2*( sin(k_1/2)^2 + sin(k_2/2)^2 ) + im*sqrt( sin(k_1)^2 + sin(k_2)^2 )
lambda_2 = am0 + 2*( sin(k_1/2)^2 + sin(k_2/2)^2 ) - im*sqrt( sin(k_1)^2 + sin(k_2)^2 )

