
function unfold_fld!(Uin, prm::LattParm, kprm::KernelParm)

    Uout = CUDA.ones(ComplexF64, 2*prm.iL[1], 2*prm.iL[2], 2)    
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks krnl_unfold!(Uout, Uin, prm.iL[1], prm.iL[2])
    end

    return Uout
end

function krnl_unfold!(Uout, Uin, l1old, l2old)

    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    for id in 1:2
        Uout[i1,i2,id]             = Uin[i1,i2,id]
        Uout[i1+l1old,i2,id]       = Uin[i1,i2,id]
        Uout[i1,i2+l2old,id]       = Uin[i1,i2,id]
        Uout[i1+l1old,i2+l2old,id] = Uin[i1,i2,id]
    end
        
    return nothing
end
