

function CG(so, U, si, am0, maxiter, eps, A, prm::LattParm, kprm::KernelParm)

    r  = similar(si)
    p  = similar(si)
    Ap = similar(si)
    tmp = similar(si)
    
    so .= complex(0.0,0.0)
    r  .= si
    p  .= si
    norm = CUDA.mapreduce(x -> abs2(x), +, si)
    err = 0.0
    
    tol = eps * norm
	    # println( tol)
	    iterations = 0
    for i in 1:maxiter
        A(Ap, tmp, U, p, am0, prm, kprm)
        prod  = CUDA.dot(p, Ap)
        alpha = norm/prod

        so .= so .+ alpha .*  p
        r  .= r  .- alpha .* Ap

        err = CUDA.mapreduce(x -> abs2(x), +, r)
        
        if err < tol
		iterations=i
            break
        end

        beta = err/norm
        p .= r .+ beta .* p
        
        norm = err;
    end

    if err > tol
	    println(err)
	    error("CG not converged after $maxiter iterationss")
    end
    
    return iterations
end


function CG(so, U, si, am0, mu_mass, maxiter, eps, A, prm::LattParm, kprm::KernelParm; so_as_guess::Bool=false)

    r  = similar(si)
    p  = similar(si)
    Ap = similar(si)
    tmp = similar(si)
    
    tol = eps * CUDA.mapreduce(x -> abs2(x), +, si)

    if(so_as_guess == false)
        so .= complex(0.0,0.0)

        r  .= si
        p  .= si
        norm = CUDA.mapreduce(x -> abs2(x), +, si)
    else
        A(Ap, tmp, U, so, am0, mu_mass, prm, kprm)
        r  .= si - Ap
        p  .= r
        norm = CUDA.mapreduce(x -> abs2(x), +, r)
        if norm < tol
            return 0
        end
    end

    err = 0.0
    
    # println( tol)
    iterations = 0
    for i in 1:maxiter
        A(Ap, tmp, U, p, am0, mu_mass, prm, kprm)
        prod  = CUDA.dot(p, Ap)
        alpha = norm/prod

        so .= so .+ alpha .*  p
        r  .= r  .- alpha .* Ap

        err = CUDA.mapreduce(x -> abs2(x), +, r)
        
        if err < tol
		iterations=i
            break
        end

        beta = err/norm
        p .= r .+ beta .* p
        
        norm = err;
    end

    if err > tol
	    println(err)
	    Base.error("CG not converged after $maxiter iterationss")
    end
    
    return iterations
end

function MultiCG(so, U, si, am0, maxiter, eps, A, rprm::RHMCParm, prm::LattParm, kprm::KernelParm)
	aux = similar(so)
	so .= si  # this accounts for identity in partial fraction descomposition
	for j in 1:rprm.n
		CG(aux, U, si, am0, rprm.mu[j], maxiter, eps, A, prm, kprm)
		so .= so .+ rprm.rho[j]*aux
	end
	so .= rprm.r_b^(-1) * rprm.A * so
end
