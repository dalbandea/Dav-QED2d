using Test

@testset verbose = true "RHMC" begin
    @testset "Pseudofermion generation" begin
        include("rhmc_pf_generation.jl")
    end

    @testset "Pseudofermion force" begin
        include("rhmc_pf_forcetest.jl") 
    end

    @testset "HMC reversibility" begin
        include("rhmc_hmc_reversibility.jl")
    end

    @testset "Zolotarev bound" begin
        include("rhmc_zolovarev_bound.jl")
    end
end

