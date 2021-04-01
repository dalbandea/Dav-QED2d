using CUDA, Logging, TerminalLoggers, BDIO, ArgParse, Statistics

import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/QED2d")
using QED2d

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--beta"
        help = "beta for the simulation"
        required = true
        arg_type = Float64
        "--size"
        help = "Target lattice size"
        required = true
        nargs = 2
    end

    return parse_args(s)
end

function set_kernel_parameters(l1, l2)

    if (l1 < 1024)
        th1 = l1
        bl1 = 1
    else
        th1 = 1024
        bl1 = div(l1, 1024)
    end
    th2 = 1
    bl2 = l2

    return KernelParm((th1, th2), (bl1, bl2))
end

function err_avg(dfoo)
    ll1  = div(size(dfoo,1), 16)
    ll2  = div(size(dfoo,2), 16)
    blks = zeros(Float64, 16, 16)
    for b1 in 1:16
        for b2 in 1:16
            for i1 in 1:ll1, i2 in 1:ll2
                blks[b1,b2] += dfoo[i1+(b1-1)*ll1, i2+(b2-1)*ll2]
            end
        end
    end
    V = ll1 * ll2
    blks .= blks ./ V
    
    return mean(blks), std(blks)/16
end


io = open("log.txt", "w+")
global_logger(TerminalLogger(io))
BDIO_set_user("alberto")
BDIO_set_host("artemisa")

parsed_args = parse_commandline()
bt = parsed_args["beta"]
sz = parsed_args["size"]

lsize = Vector{Int64}(undef, 2)
lsize[1] = tryparse(Int64, sz[1])
lsize[2] = tryparse(Int64, sz[2])
if (lsize[1] < 128) || (lsize[2] < 128)
    @error "Lattice is too small for a master field"
end

eps = 0.025
ns  = 20
nt  = 2000
@info("SIMULATION AT BETA = $bt")
@info("  MD PARAMETER eps = $eps")
@info("  TRAJECTORY tau   = $(eps*ns)")
@info("  TRAJECTORIES / folding = $nt")

global l1  = 16
global l2  = 16
U   = CUDA.ones(ComplexF64, l1, l2, 2)    
while true
    global prm  = LattParm((l1, l2), bt)
    global kprm = set_kernel_parameters(prm.iL[1], prm.iL[2])

    @info(" ## DOUBLING Lattice size: $(prm.iL[1]) x $(prm.iL[2])")
    for i in 1:nt
        @info(" # Trajectory $i / $nt [$(prm.iL[1]) x $(prm.iL[2])]")
        HMC!(U, eps, ns, prm, kprm)
        @info("   Plaquette: ", Plaquette(U, prm, kprm))
        @info("   Qtop:      ", Qtop(U, prm, kprm))
    end

    global U = unfold_fld!(U, prm, kprm)

    global l1 = 2*prm.iL[1]
    global l2 = 2*prm.iL[2]
    if l1 > lsize[1]
        break
    end
end

fb = BDIO_open("data_$(prm.iL[1])x$(prm.iL[2]).bdio", "w",
               "Data from Master field simulation in QED [2d]")

BDIO_start_record!(fb, BDIO_BIN_F64LE, 2, true)
dfoo = Vector{Float64}(undef, 3)
dfoo[1] = prm.iL[1]
dfoo[2] = prm.iL[2]
dfoo[3] = bt
BDIO_write!(fb, dfoo)
BDIO_write_hash!(fb)

acum = CUDA.CuArray{Float64}(undef, prm.iL[1], prm.iL[2])
CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks plaquette!(acum, U, prm)
end

dfoo = Array(acum)
BDIO_start_record!(fb, BDIO_BIN_F64LE, 3, true)
for i in 1:prm.iL[1]
    BDIO_write!(fb, dfoo[i,:])
end
BDIO_write_hash!(fb)
pavg, perr = err_avg(dfoo)
@info("### Plaquette and error: $pavg +/- $perr")
            
CUDA.@sync begin
    CUDA.@cuda threads=kprm.threads blocks=kprm.blocks qtop!(acum, U, prm)
end

dfoo = Array(acum)
BDIO_start_record!(fb, BDIO_BIN_F64LE, 4, true)
for i in 1:prm.iL[1]
    BDIO_write!(fb, dfoo[i,:])
end
BDIO_write_hash!(fb)

BDIO_close!(fb)
