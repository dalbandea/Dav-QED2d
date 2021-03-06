# Generates a fermionic planewave
function plwv(so, k::Tuple{Float64, Float64}, prm::LattParm)

    i1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    i2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

    so[i1,i2,1] = im*sqrt(-sin(k[1])^2-sin(k[2])^2 |> complex )/(sin(k[1])+im*sin(k[2]))*exp(im*( k[1]*(i1-1) + k[2]*(i2-1) ) )
    so[i1,i2,2] = exp(im*( k[1]*(i1-1) + k[2]*(i2-1) ) )

    return nothing
end
