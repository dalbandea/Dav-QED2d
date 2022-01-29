using QED2d
using CUDA
using Random

# Lattice and Zolotarev parameters
lsize = 10          # lattice size
lbeta = 6.05        # beta
am0 = [10.0]        # bare mass
n_rhmc = 2          # number of Zolotarev monomial pairs

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

print("Allocating gauge field...")
U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
println(" DONE")

# Apply power method to extract maximum and minimum eigenvalues of D^†D
lambda_min, lambda_max = power_method(U, am0[1], prm, kprm)    

# Generate Zolotarev parameters.
# eps_rhmc is defined such that r_a and r_b are the sqrt of minimum and maximum
# eigenvalues of D^†D.
n_rhmc = 5
r_a_rhmc = lambda_min |> real |> x->x*1.0 |> sqrt  
r_b_rhmc = lambda_max |> real |> x->x*1.0 |> sqrt
rprm = get_rhmc_params([n_rhmc], [r_a_rhmc], [r_b_rhmc])

# Initial action
Sini = Action(U, prm, kprm)

# Get pseudofermion field
acc = Vector{Int64}()
nsteps  = 100
epsilon = 1.0/nsteps
CGmaxiter = 10000
CGtol = 1e-25
@time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false, reversibility_test = true)

# Final action after reversibility test
Sfin =  Action(U, prm, kprm)

deltadeltaS = Sfin - Sini

println("Δ(ΔS) = $deltadeltaS")

@test abs(deltadeltaS) < 1e-10
