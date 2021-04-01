
function plaquette!(plx, U, prm::LattParm)

    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    iu1 = mod(i1, prm.iL[1]) + 1
    iu2 = mod(i2, prm.iL[2]) + 1
    
    plx[i1,i2] = real(U[i1,i2,1] *
                      U[iu1,i2,2] *
                      conj(U[i1,iu2,1] *
                           U[i1,i2,2]))
    
    return nothing
end

function qtop!(plx, U, prm::LattParm)

    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    iu1 = mod(i1, prm.iL[1]) + 1
    iu2 = mod(i2, prm.iL[2]) + 1
    
    plx[i1,i2] = CUDA.angle(U[i1,i2,1] *
                            U[iu1,i2,2] *
                            conj(U[i1,iu2,1] *
                                 U[i1,i2,2]))
    
    return nothing
end

function krnl_force!(frc1, frc2, U, prm::LattParm)
    
    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    iu1 = mod(i1, prm.iL[1]) + 1
    iu2 = mod(i2, prm.iL[2]) + 1

    v = prm.beta*imag(U[i1,i2,1] *
                      U[iu1,i2,2] *
                      conj(U[i1,iu2,1] *
                           U[i1,i2,2]))
    
    frc1[i1,i2,1]  = -v 
    frc1[i1,i2,2]  =  v 
    frc2[iu1,i2,2] = -v 
    frc2[i1,iu2,1] =  v 

    return nothing
end

function updt_fld!(U, mom, eps)

    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    for id in 1:2
        U[i1,i2,id] = complex(CUDA.cos(eps*mom[i1,i2,id]), CUDA.sin(eps*mom[i1,i2,id])) * U[i1,i2,id]
    end
    
    return nothing
end
    
