
function HMC!(U, eps, ns, acc, prm::LattParm, kprm::KernelParm; qzero = false)

    Ucp = similar(U)
    Ucp .= U
    
    mom = CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)
    hini = Hamiltonian(mom, U, prm, kprm)

    OMF4!(mom, U, eps, ns, prm::LattParm, kprm::KernelParm)
    
    hfin = Hamiltonian(mom, U, prm, kprm)
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
        @info("    ACCEPT:  Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]; Pacc = $(count(acc.==1)/length(acc))")
    end
    
    return nothing
end


function HMC!(U, am0, eps, ns, acc, CGmaxiter, tol, prm::LattParm, kprm::KernelParm; qzero = false)

    Ucp = similar(U)
    Ucp .= U
    
    mom = CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)
    X = (CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2) .+ CUDA.randn(Float64, prm.iL[1], prm.iL[2], 2)im)/sqrt(2)
    F = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)
    g5DX = CUDA.zeros(ComplexF64, prm.iL[1], prm.iL[2], 2)

    hini = real(Hamiltonian(mom, U, prm, kprm) + CUDA.dot(X,X))

    CUDA.@sync begin
	    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(F, U, X, am0, prm)
    end

    # OMF4!(mom, U, X, F, g5DX, am0,  eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm)
    leapfrog!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm)
    
    
    hfin = real(Hamiltonian(mom, U, prm, kprm) + CUDA.dot(X,F))

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


function Hamiltonian(mom, U, prm::LattParm, kprm::KernelParm)

    hini = CUDA.mapreduce(x -> x^2, +, mom)/2.0 + Action(U, prm, kprm)

    return hini
end


function Action(U, prm::LattParm, kprm::KernelParm)

    acum = CUDA.CuArray{Float64}(undef, prm.iL[1], prm.iL[2])
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks plaquette!(acum, U, prm)
    end
    S = prm.beta * ( prm.iL[1]*prm.iL[2] - CUDA.reduce(+, acum) )

    return S
end

function Plaquette(U, prm::LattParm, kprm::KernelParm)

    acum = CUDA.CuArray{Float64}(undef, prm.iL[1], prm.iL[2])
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks plaquette!(acum, U, prm)
    end
    S = CUDA.reduce(+, acum) / (prm.iL[1]*prm.iL[2])

    return S
end

function Qtop(U, prm::LattParm, kprm::KernelParm)

    acum = CUDA.CuArray{Float64}(undef, prm.iL[1], prm.iL[2])
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks qtop!(acum, U, prm)
    end
    S = CUDA.reduce(+, acum) / (2.0*pi)

    return S
end

function OMF4!(mom, U, eps, ns, prm::LattParm, kprm::KernelParm)

    r1::Float64 =  0.08398315262876693
    r2::Float64 =  0.2539785108410595
    r3::Float64 =  0.6822365335719091
    r4::Float64 = -0.03230286765269967
    r5::Float64 =  0.5-r1-r3
    r6::Float64 =  1.0-2.0*(r2+r4)

    frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
    frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)

    for i in 1:ns
        # STEP 1
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r1*eps).*(frc1.+frc2)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r2)         
        end

        # STEP 2
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r3*eps).*(frc1.+frc2)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r4)         
        end

        # STEP 3
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r5*eps).*(frc1.+frc2)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r6)         
        end

        # STEP 4
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r5*eps).*(frc1.+frc2)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r4)         
        end

        # STEP 5
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r3*eps).*(frc1.+frc2)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r2)         
        end

        # STEP 6
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r1*eps).*(frc1.+frc2)
    end

    return nothing
end

function OMF4!(mom, U, X, F, g5DX, am0, eps, ns, maxiter, tol, prm::LattParm, kprm::KernelParm)

    r1::Float64 =  0.08398315262876693
    r2::Float64 =  0.2539785108410595
    r3::Float64 =  0.6822365335719091
    r4::Float64 = -0.03230286765269967
    r5::Float64 =  0.5-r1-r3
    r6::Float64 =  1.0-2.0*(r2+r4)

    frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
    frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
    frc = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)

    for i in 1:ns
        # STEP 1
	CG(X, U, F, am0, maxiter, tol, gamm5Dw_sqr_msq, prm, kprm)

	CUDA.@sync begin
		CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
	end
	CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r1*eps).*(frc1.+frc2.+frc)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r2)         
        end

        # STEP 2
	CG(X, U, F, am0, maxiter, tol, gamm5Dw_sqr_msq, prm, kprm)

	CUDA.@sync begin
		CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
	end
	CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r3*eps).*(frc1.+frc2.+frc)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r4)         
        end

        # STEP 3
	CG(X, U, F, am0, maxiter, tol, gamm5Dw_sqr_msq, prm, kprm)

	CUDA.@sync begin
		CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
	end
	CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r5*eps).*(frc1.+frc2.+frc)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r6)         
        end

        # STEP 4
	CG(X, U, F, am0, maxiter, tol, gamm5Dw_sqr_msq, prm, kprm)

	CUDA.@sync begin
		CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
	end
	CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r5*eps).*(frc1.+frc2.+frc)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r4)         
        end

        # STEP 5
	CG(X, U, F, am0, maxiter, tol, gamm5Dw_sqr_msq, prm, kprm)

	CUDA.@sync begin
		CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
	end
	CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r3*eps).*(frc1.+frc2.+frc)
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps*r2)         
        end

        # STEP 6
	CG(X, U, F, am0, maxiter, tol, gamm5Dw_sqr_msq, prm, kprm)

	CUDA.@sync begin
		CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
	end
	CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end
        mom .= mom .+ (r1*eps).*(frc1.+frc2.+frc)
    end

    return nothing
end


function leapfrog!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm::LattParm, kprm::KernelParm)

	# First half-step for momenta
	update_momenta!(mom, U, X, F, g5DX, am0, 0.5*eps, ns, CGmaxiter, tol, prm, kprm)

	# ns-1 steps
	for i in 1:(ns-1) 
		# Update gauge links
		CUDA.@sync begin
		    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps)         
		end
		#Update momenta
		update_momenta!(mom, U, X, F, g5DX, am0, eps, ns, CGmaxiter, tol, prm, kprm)
	end
	# Last update for gauge links
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks updt_fld!(U, mom, eps)         
        end

	# Last half-step for momenta
	update_momenta!(mom, U, X, F, g5DX, am0, 0.5*eps, ns, CGmaxiter, tol, prm, kprm)

	return nothing
end

function update_momenta!(mom, U, X, F, g5DX, am0, eps, ns, maxiter, tol, prm::LattParm, kprm::KernelParm)
	# frc1 and frc2 will be for the gauge part of the force, frc for the
	# fermion part
	frc1 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc2 = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)
	frc = CUDA.zeros(Float64, prm.iL[1], prm.iL[2], 2)

	# Solve DX = F for X
	iter = CG(X, U, F, am0, maxiter, tol, gamm5Dw_sqr_msq, prm, kprm)

	# Apply gamm5D to X
	CUDA.@sync begin
		CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(g5DX, U, X, am0, prm)
	end
	
	# Get fermion part of the force in frc
	CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks tr_dQwdU(frc, U, X, g5DX, prm)
        end

	# Get gauge part of the force in frc1 and frc2
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_force!(frc1, frc2, U, prm)
        end

	# Final force is frc1+frc2+frc
        mom .= mom .+ (eps).*(frc1.+frc2.+frc)

	return nothing
end
