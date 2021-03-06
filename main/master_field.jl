using CUDA, Logging, TerminalLoggers, BDIO, ArgParse, Statistics, JSON, Random

import Pkg
Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/QED2d")
using QED2d

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-i"
        help = "input file"
        required = true
        arg_type = String
    end

    return parse_args(s)
end

function set_kernel_parameters(l1, l2)

    if (l1*l2 < 1024)
        th1 = l1
        th2 = l2
        return KernelParm((th1, th2), (1, 1))
    end
    
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

function read_options(fname)

    io = open(fname, "r")
    s = JSON.parse(io)
    close(io)

    BDIO_set_user(s["Run"]["user"])
    BDIO_set_host(s["Run"]["host"])
    nmsm = s["Run"]["measurements"]
    
    io = open(s["Run"]["name"]*".log", "w+")
    global_logger(TerminalLogger(io))

    Random.seed!(CURAND.default_rng(), 1234)
    Random.seed!(1234)

    eps   = s["HMC"]["eps"]
    ns    = s["HMC"]["ns"]
    nt    = s["HMC"]["ntraj"]
    qzero = s["HMC"]["Qzero"]
    
    lsize = Vector{Int64}(undef, 2)
    lsize[1] = s["Lattice"]["size"][1]
    lsize[2] = s["Lattice"]["size"][2]
    if (lsize[1] < 128) || (lsize[2] < 128)
        @error "Lattice is too small for a master field"
    end
    bt = s["Lattice"]["beta"]
    
    fb = BDIO_open(s["Run"]["name"]*".bdio", "w",
                   "Data from Master field simulation in QED [2d]; beta = $bt; $(lsize[1]) x $(lsize[2])")
    BDIO_start_record!(fb, BDIO_ASC_GENERIC, 1, true)
    BDIO_write!(fb, read(fname))
    BDIO_write_hash!(fb)
    
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 2, true)
    dfoo = Vector{Float64}(undef, 4)
    dfoo[1] = lsize[1]
    dfoo[2] = lsize[2]
    dfoo[3] = bt
    dfoo[4] = nmsm
    BDIO_write!(fb, dfoo)
    BDIO_write_hash!(fb)
    
    return eps, ns, nt, lsize, bt, nmsm, qzero, fb
end

function save_msm(fb, U, prm, kprm)
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
    @info(" MEASUREMENT $i of $nmsm")
    @info("   Plaquette and error: $pavg +/- $perr")
    
    CUDA.@sync begin
        CUDA.@cuda threads=kprm.threads blocks=kprm.blocks qtop!(acum, U, prm)
    end
    
    dfoo = Array(acum)
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 4, true)
    for i in 1:prm.iL[1]
        BDIO_write!(fb, dfoo[i,:])
    end
    BDIO_write_hash!(fb)

end

parsed_args = parse_commandline()
infile = parsed_args["i"]

eps, ns, nt, lsize, bt, nmsm, qzero, fb = read_options(infile)

@info("MASTER FIELD SIMULATION AT BETA = $bt. TARGET SIZE: $(lsize[1]) x $(lsize[2])")
@info("  MD PARAMETER eps = $eps")
@info("  TRAJECTORY tau   = $(eps*ns)")
@info("  MDU's / folding  = $(eps*ns*nt)")

acc = Vector{Int64}()
global l1  = 16
global l2  = 16
U   = CUDA.ones(ComplexF64, l1, l2, 2)    
while true
    global prm  = LattParm((l1, l2), bt)
    global kprm = set_kernel_parameters(prm.iL[1], prm.iL[2])

    @info(" ## DOUBLING Lattice size: $(prm.iL[1]) x $(prm.iL[2])")
    for i in 1:nt
        @info(" # Trajectory $i / $nt [$(prm.iL[1]) x $(prm.iL[2])]")
        HMC!(U, eps, ns, acc, prm, kprm, qzero=qzero)
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


for j in 1:nmsm    
    save_msm(fb, U, prm, kprm)    
    
    for i in 1:nt
        @info(" # [measurement $j / $nmsm] Trajectory $i / $nt [$(prm.iL[1]) x $(prm.iL[2])]")
        HMC!(U, eps, ns, acc, prm, kprm, qzero=qzero)
        @info("   Plaquette: ", Plaquette(U, prm, kprm))
        @info("   Qtop:      ", Qtop(U, prm, kprm))
    end
end
    
save_msm(fb, U, prm, kprm)
BDIO_close!(fb)
