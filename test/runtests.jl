using Test

@testset "RHMC" begin
    println("RHMC pseudofermion force test")
    include("rhmc_pf_forcetest.jl") 

    println("RHMC pseudofermion generation test")
    include("rhmc_pf_generation.jl")

    println("RHMC reversibility test")
    include("rhmc_hmc_reversibility.jl")

    println("RHMC Zolotarev bound test")
    include("rhmc_zolovarev_bound.jl")
end


