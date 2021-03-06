# leapfrog for Nf=Nf non-degenerate fermions (RHMC) with chronological time
# ordering (to reuse solution sfrom last integration step)
function leapfrog!(mom, U, X, F, g5DX, am0::Array{Float64}, eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm::Array{RHMCParm}; so_as_guess::Bool=false)

    iter = 0

	# First half-step for momenta
	iter += update_momenta!(mom, U, X, F, g5DX, am0, 0.5*eps, ns, CGmaxiter, tol, prm, kprm, rprm)

	# ns-1 steps
	for i in 1:(ns-1) 
		# Update gauge links
		CUDA.@sync begin
		    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps)         
		end
		#Update momenta
		iter += update_momenta!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm, kprm, rprm, so_as_guess=so_as_guess)
	end
	# Last update for gauge links
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps)
    end

	# Last half-step for momenta
	iter += update_momenta!(mom, U, X, F, g5DX, am0, 0.5*eps, ns, CGmaxiter, tol, prm, kprm, rprm)

	return nothing
end

# update_momenta for Nf=Nf non-degenerate fermions (RHMC) with chronological
# time ordering (to reuse solutions from last integration step)
function update_momenta!(mom, U, X, F, g5DX, am0, eps, ns, maxiter, tol, prm::LattParm, kprm::KernelParm, rprm::Array{RHMCParm}; so_as_guess::Bool=false)
	# frc1 and frc2 will be for the gauge part of the force, 
    # frc_i will hold the force of the pseudofermion ϕᵢ of fermion j
	# frc = ∑ⱼ ∑ᵢ frc_i
	frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc_i = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)

    iter = 0

    for j in 1:length(am0)
        for i in 1:rprm[j].n
            # X = (D†D+μ²)⁻¹ϕᵢ
            iter += CG(X[j][i], U, F[j][i], am0[j], rprm[j].mu[i], maxiter, tol, gamm5Dw_sqr_musq, prm, kprm, so_as_guess=so_as_guess)

            # Apply γD to X
            CUDA.@sync begin
                CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X[j][i], am0[j], prm)
            end
            
            # Get fermion part of the force in frc_i
            CUDA.@sync begin
                CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc_i, U, X[j][i], g5DX, prm)
            end
            frc .= frc .+ (rprm[j].nu[i]^2-rprm[j].mu[i]^2)*frc_i
        end
    end

	# Get gauge part of the force in frc1 and frc2
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
    end

	# Final force is frc1+frc2+frc
    mom .= mom .+ (eps).*(frc1.+frc2.+frc)

	return iter
end


# leapfrog for Nf=1 fermions (deprecated)
function leapfrog!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm::RHMCParm)

    iter = 0    # number of CG iterations

	# First half-step for momenta
	iter += update_momenta!(mom, U, X, F, g5DX, am0, 0.5*eps, ns, CGmaxiter, tol, prm, kprm, rprm)

	# ns-1 steps
	for i in 1:(ns-1) 
		# Update gauge links
		CUDA.@sync begin
		    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps)         
		end
		#Update momenta
		iter += update_momenta!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm, kprm, rprm)
	end
	# Last update for gauge links
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps)         
        end

	# Last half-step for momenta
	iter += update_momenta!(mom, U, X, F, g5DX, am0, 0.5*eps, ns, CGmaxiter, tol, prm, kprm, rprm)


    println("Iterations: $iter")

	return nothing
end

# update_momenta for Nf=1 fermions (deprecated)
function update_momenta!(mom, U, X, F, g5DX, am0, eps, ns, maxiter, tol, prm::LattParm, kprm::KernelParm, rprm::RHMCParm)
	# frc1 and frc2 will be for the gauge part of the force, 
    # frc_i will hold the force of the pseudofermion ϕᵢ
	# frc = ∑ frc_i
	frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc_i = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)

    iter = 0    # number of CG iterations

    for i in 1:rprm.n
        # X = (D†D+μ²)⁻¹ϕᵢ
        iter += CG(X, U, F[i], am0, rprm.mu[i], maxiter, tol, gamm5Dw_sqr_musq, prm, kprm, so_as_guess)

        # Apply γD to X
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
        end
        
        # Get fermion part of the force in frc_i
        CUDA.@sync begin
                CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc_i, U, X, g5DX, prm)
        end
        frc .= frc .+ (rprm.nu[i]^2-rprm.mu[i]^2)*frc_i
    end

	# Get gauge part of the force in frc1 and frc2
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
    end

	# Final force is frc1+frc2+frc
    mom .= mom .+ (eps).*(frc1.+frc2.+frc)

	return iter
end

#  HMC step for Nf=1 fermion (RHMC) 
#  (deprecated, use HMC function for Nf non-degenerate fermions instead)
function HMC!(U, am0, eps, ns, acc, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm::RHMCParm; qzero = false)

    Ucp = similar(U)
    Ucp .= U
    
    # Generate random momenta
    mom = CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)

    # Initialize initial Hamiltonian
    hini = Hamiltonian(mom, U, prm, kprm) |> real

    # Initialize array of pseudofermion fields, X and γDX
    F = Array{CUDA.CuArray}(undef, rprm.n)
    X = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
    g5DX = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)

    # Fill array of n_rhmc pseudofermion fields and complete initial Hamiltonian
    # S_pf = ∑ ϕᵢ†(D†D + μᵢ²)⁻¹ (D†D + νᵢ²)ϕᵢ = ∑ X† X
    # ϕᵢ = (γD + iν)⁻¹(γD + iμ) Xᵢ  if Xᵢ is random normal. X is reused
    for i in 1:rprm.n
        # Generate random Xᵢ
        X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
        hini += CUDA.dot(X,X) |> real
        
        # Initialize pseudofermion field ϕᵢ and obtain it from Xᵢ
        F[i] = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
        # ϕᵢ = (γD+iμ)X
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[i], U, X, am0, prm)
        end
        F[i] .= F[i] .+ im*rprm.mu[i] .* X
        # ϕᵢ = (D†D+ν²)⁻¹ϕᵢ
        aux_F = copy(F[i])
        CG(F[i], U, aux_F, am0, rprm.nu[i], CGmaxiter, tol, gamm5Dw_sqr_musq, prm, kprm)
        # ϕᵢ = (γD-iν)ϕᵢ
        aux_F .= F[i]
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[i], U, aux_F, am0, prm)
        end
        F[i] .= F[i] .- im*rprm.nu[i] .* aux_F
    end

    # OMF4!(mom, U, X, F, g5DX, am0,  eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm::RHMCParm)
    leapfrog!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm)
    
    # Initialize final Hamiltonian and complete it with pseudofermion
    # contributions
    hfin = Hamiltonian(mom, U, prm, kprm) |> real
    for i in 1:rprm.n
        # ξᵢ = (D†D+μ²)⁻¹(D†D+ν²)ϕᵢ but reuse ξᵢ in X
        X = copy(F[i])
        tmp = copy(F[i])
        gamm5Dw_sqr_musq(X, tmp, U, F[i], am0, rprm.nu[i], prm, kprm)
        tmp .= X
        CG(X, U, tmp, am0, rprm.mu[i], CGmaxiter, tol, gamm5Dw_sqr_musq, prm, kprm)
        hfin += CUDA.dot(X,F[i]) |> real
    end

    if qzero
        if abs(Qtop(U, prm, kprm)) > 0.1
            pacc = 0.0
        else
            pacc = exp(-(hfin-hini))
        end
    else
        pacc = exp(-(hfin-hini))
    end
    
    if (pacc < 1.0)
        r = rand()
        if (pacc > r) 
            push!(acc, 1)
        else
            U .= Ucp
            push!(acc, 0)
        end
    else
        push!(acc, 1)
    end

    if (acc[end] == 0)
        @info("    REJECT: Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]; Pacc = $(count(acc.==1)/length(acc))")
    else
        @info("    ACCEPT: Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]; Pacc = $(count(acc.==1)/length(acc))")
    end
    
    return nothing
end

# HMC step for Nf=Nf non-degenerate (RHMC)
function HMC!(U, am0::Array{Float64}, eps, ns, acc, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm::Array{RHMCParm}; qzero = false, so_as_guess=false)

    Ucp = similar(U)
    Ucp .= U
    
    # Generate random momenta
    mom = CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)

    # Initialize initial Hamiltonian
    hini = Hamiltonian(mom, U, prm, kprm) |> real

    # Initialize array of pseudofermion fields
    F = Array{Array{CUDA.CuArray}}(undef, length(am0))
    for j in 1:length(am0)
        F[j] = Array{CUDA.CuArray}(undef, rprm[j].n)
    end
    # Initialize array of X.
    # If so_as_guess=true, X occupies more memory, but faster in CG;
    # If false, all elements point to the same object: save memory, but slower
    # in CG
    if(so_as_guess == true) 
        X = Array{Array{CUDA.CuArray}}(undef, length(am0))
        for j in 1:length(am0)
            X[j] = Array{CUDA.CuArray}(undef, rprm[j].n)
            for i in 1:rprm[j].n
                X[j][i] = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
            end
        end
    else
        X_reused = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
        X = Array{Array{CUDA.CuArray}}(undef, length(am0))
        for j in 1:length(am0)
            X[j] = Array{CUDA.CuArray}(undef, rprm[j].n)
            for i in 1:rprm[j].n
                X[j][i] = X_reused  # the two objects are the same
            end
        end
    end
    # Initialize g5DX
    g5DX = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)

    # Fill array of pseudofermion fields for each fermion
    for j in 1:length(am0)
        # Fill array of n_rhmc pseudofermion fields of fermion j and complete
        # initial Hamiltonian 
        # S_pf = ∑ ϕᵢ†(D†D + μᵢ²)⁻¹ (D†D + νᵢ²)ϕᵢ = ∑ X† X
        # ϕᵢ = (γD + iν)⁻¹(γD + iμ) Xᵢ  if Xᵢ is random normal. X is reused
        for i in 1:rprm[j].n
            # Generate random Xᵢ
            X[j][i] .= (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
            hini += CUDA.dot(X[j][i],X[j][i]) |> real
            
            # Initialize pseudofermion field ϕᵢ and obtain it from Xᵢ
            F[j][i] = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
            # ϕᵢ = (γD+iμ)X
            CUDA.@sync begin
                CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[j][i], U, X[j][i], am0[j], prm)
            end
            F[j][i] .= F[j][i] .+ im*rprm[j].mu[i] .* X[j][i]
            # ϕᵢ = (D†D+ν²)⁻¹ϕᵢ
            aux_F = copy(F[j][i])
            CG(F[j][i], U, aux_F, am0[j], rprm[j].nu[i], CGmaxiter, tol, gamm5Dw_sqr_musq, prm, kprm)
            # ϕᵢ = (γD-iν)ϕᵢ
            aux_F .= F[j][i]
            CUDA.@sync begin
                CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F[j][i], U, aux_F, am0[j], prm)
            end
            F[j][i] .= F[j][i] .- im*rprm[j].nu[i] .* aux_F
        end
    end

    # OMF4!(mom, U, X, F, g5DX, am0,  eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm::RHMCParm)
    leapfrog!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm, rprm, so_as_guess=so_as_guess)
    
    # Initialize final Hamiltonian and complete it with pseudofermion
    # contributions
    hfin = Hamiltonian(mom, U, prm, kprm) |> real
    for j in 1:length(am0)
        for i in 1:rprm[j].n
            # ξ = (D†D+μ²)⁻¹(D†D+ν²)ϕᵢ but reuse ξ in X
            X = copy(F[j][i])
            tmp = copy(F[j][i])
            gamm5Dw_sqr_musq(X, tmp, U, F[j][i], am0[j], rprm[j].nu[i], prm, kprm)
            tmp .= X
            CG(X, U, tmp, am0[j], rprm[j].mu[i], CGmaxiter, tol, gamm5Dw_sqr_musq, prm, kprm)
            hfin += CUDA.dot(X,F[j][i]) |> real
        end
    end

    if qzero
        if abs(Qtop(U, prm, kprm)) > 0.1
            pacc = 0.0
        else
            pacc = exp(-(hfin-hini))
        end
    else
        pacc = exp(-(hfin-hini))
    end
    
    if (pacc < 1.0)
        r = rand()
        if (pacc > r) 
            push!(acc, 1)
        else
            U .= Ucp
            push!(acc, 0)
        end
    else
        push!(acc, 1)
    end

    if (acc[end] == 0)
        @info("    REJECT: Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]; Pacc = $(count(acc.==1)/length(acc))")
    else
        @info("    ACCEPT: Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]; Pacc = $(count(acc.==1)/length(acc))")
    end
    
    return nothing
end
