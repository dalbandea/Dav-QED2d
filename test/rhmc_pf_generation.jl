using QED2d
using CUDA
using Random


# Lattice and Zolotarev parameters
lsize = 10          # lattice size
lbeta = 6.05        # beta
am0 = [0.02, 1.2]   # bare mass
n_rhmc = [1, 1]     # number of Zolotarev monomial pairs

global prm  = LattParm((lsize,lsize), lbeta)
global kprm = KernelParm((lsize, 1), (1,lsize))

Random.seed!(CURAND.default_rng(), 1234)
Random.seed!(1234)

print("Allocating gauge field...")
U = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
println(" DONE")

# Generate Zolotarev parameters.
# eps_rhmc is defined such that r_a and r_b are the sqrt of minimum and maximum
# eigenvalues of D^†D.
r_a_rhmc = 0.1 |> sqrt  
r_b_rhmc = 26  |> sqrt
rprm = get_rhmc_params(n_rhmc, [r_a_rhmc, r_a_rhmc], [r_b_rhmc, r_b_rhmc])

CGmaxiter = 10000
tol = 1e-25

# Initialize array of pseudofermion fields and X
N_fermions = length(am0)  
F = Array{CUDA.CuArray}(undef, N_fermions)
X = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)

global hini = 0.0
for j in 1:N_fermions
    # Generate random Xⱼ
    X .= (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
    global hini += CUDA.mapreduce(x -> abs2(x), +, X)
    
    # Initialize pseudofermion field ϕⱼ and obtain it from X
    F[j] = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
    generate_pseudofermion!(F[j], U, X, am0[j], CGmaxiter, tol, prm, kprm, rprm[j])
end

global hfin = 0.0
for j in 1:N_fermions
        # ξⱼ = ∏(D†D+μ²)⁻¹(D†D+ν²)ϕⱼ = ( 1+∑ρᵢ/(DD†+μ²) ) ϕⱼ =
        # MultiCG(ξⱼ,U,ϕⱼ,...) but reuse ξⱼ in X
        MultiCG(X, U, F[j], am0[j], CGmaxiter, tol, gamm5Dw_sqr_musq, rprm[j], prm, kprm)
        global hfin += CUDA.dot(X,F[j]) |> real
end

println("ΔH = $(hfin - hini)")
@test abs(hfin - hini) < 1e-10
