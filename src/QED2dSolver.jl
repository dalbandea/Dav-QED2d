

function CG(so, U, si, am0, maxiter, eps, A, prm::LattParm, kprm::KernelParm)

    r  = similar(si)
    p  = similar(si)
    Ap = similar(si)
    tmp = similar(si)
    
    so .= complex(0.0,0.0)
    r  .= si
    p  .= si
    norm = CUDA.mapreduce(x -> abs2(x), +, si)
    
    tol = eps * normsq
    for i in 1:maxiter
        CUDA.@sync begin
            CUDA.@cuda threads=kprm.threads blocks=kprm.blocks A(Ap, tmp, U, p, am0, prm, kprm)
        end
        prod  = CUBLAS.dot(p, q2p)
        alpha = norm/prod

        so .= so .+ alpha .*  p
        r  .= r  .- alpha .* Ap

        err = CUDA.mapreduce(x -> abs2(x), +, r)
        
        if err < tol
            break
        end

        beta = err/norm
        p .= r .+ beta .* p
        
        norm = err;
    end

    error("CG not converged after $maxiter iterations")
    
    return i
end
