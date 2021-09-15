
function gamm5Dw(so, U, si, am0, prm::LattParm)

    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    iu1 = mod(i1, prm.iL[1]) + 1
    iu2 = mod(i2, prm.iL[2]) + 1

    id1 = mod1(i1-1, prm.iL[1])
    id2 = mod1(i2-1, prm.iL[2])

    A = 0.5 * ( U[i1,i2,1] * (si[iu1,i2,1] - si[iu1,i2,2]) +
                conj(U[id1,i2,1])*(si[id1,i2,1] + si[id1,i2,2]) )
    B = 0.5 * ( U[i1,i2,2] * (si[i1,iu2,1] +
                              complex(-imag(si[i1,iu2,2]),real(si[i1,iu2,2]))) +
                conj(U[i1,id2,2])*(si[i1,id2,1] +
                                    complex(imag(si[i1,id2,2]),-real(si[i1,id2,2]))) )
    A2 = 0.5 * ( U[i1,i2,1] * (si[iu1,i2,1] - si[iu1,i2,2]) -
                conj(U[id1,i2,1])*(si[id1,i2,1] + si[id1,i2,2]) )
    B2 = 0.5 * ( U[i1,i2,2] * (-si[i1,iu2,2] +
                              complex(-imag(si[i1,iu2,1]),real(si[i1,iu2,1]))) -
                conj(U[i1,id2,2])*(si[i1,id2,2] +
                                    complex(-imag(si[i1,id2,1]),real(si[i1,id2,1]))) )

    
    so[i1,i2,1] =  (2.0+am0) * si[i1,i2,1] - A - B
    # so[i1,i2,2] = -(2.0+am0) * si[i1,i2,2] - A + complex(-imag(B),real(B))
    so[i1,i2,2] = -(2.0+am0) * si[i1,i2,2] - A2 -B2
    

    return nothing
end

function gamm5Dw_sqr(so, U, si, am0::Float64, prm::LattParm, kprm::KernelParm)

    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(so, U, si, am0, prm)
    end
    si .= so
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(so, U, si, am0, prm)
    end
    
    return nothing
end

function gamm5Dw_sqr_msq(so, tmp, U, si, am0::Float64, prm::LattParm, kprm::KernelParm)

    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(so, U, si, am0, prm)
    end
    tmp .= so
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks gamm5Dw(so, U, tmp, am0, prm)
    end
    # so .= so .+ (am0^2)
    
    return nothing
end

function tr_dQwdU(frc, U, X, g5DwX, prm::LattParm)

    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    iu1 = mod(i1, prm.iL[1]) + 1
    iu2 = mod(i2, prm.iL[2]) + 1

    z1 = conj(U[i1,i2,1])*( conj(X[iu1,i2,1])*(g5DwX[i1,i2,1] + g5DwX[i1,i2,2])  -
                           conj(X[iu1,i2,2])*(g5DwX[i1,i2,1] + g5DwX[i1,i2,2]) ) -
     U[i1,i2,1]*( conj(X[i1,i2,1])*(g5DwX[iu1,i2,1] - g5DwX[iu1,i2,2]) +
                   conj(X[i1,i2,2])*(g5DwX[iu1,i2,1] - g5DwX[iu1,i2,2]) )




    z2 = conj(U[i1,i2,2])*( conj(X[i1,iu2,1])*(g5DwX[i1,i2,1] - (1im)*g5DwX[i1,i2,2])  -
                           conj(X[i1,iu2,2])*((1im)*g5DwX[i1,i2,1] + g5DwX[i1,i2,2])) -
     U[i1,i2,2]*( conj(X[i1,i2,1])*(g5DwX[i1,iu2,1] + (1im)*g5DwX[i1,iu2,2]) +
                   conj(X[i1,i2,2])*((1im)*g5DwX[i1,iu2,1] - g5DwX[i1,iu2,2]) )
   
                   
    
    frc[i1,i2,1] = real((1im)*z1)
    frc[i1,i2,2] = real((1im)*z2)

    return nothing
end
