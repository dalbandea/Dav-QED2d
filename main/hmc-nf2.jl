using Pkg
Pkg.activate(".")
using QED2d
using CUDA, Logging, Random, DelimitedFiles, BDIO, JSON, ArgParse, Statistics

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "-i"
        help = "input file"
        required = true
        arg_type = String
        "--savedir"
        help = "path to directory to save configurations"
        required = false
        arg_type = String
        default = "configs/"
        "--cont"
        help = "continue from configuration number"
        required = false
        arg_type = Int
        default = 0
    end

    return parse_args(s)
end

function read_options(fname)

    io = open(fname, "r")
    s = JSON.parse(io)
    close(io)

    BDIO_set_user(s["Run"]["user"])
    BDIO_set_host(s["Run"]["host"])
    seed = s["Run"]["seed"]
    
    # io = open(s["Run"]["name"]*".log", "w+")
    # global_logger(TerminalLogger(io))

    Random.seed!(CURAND.default_rng(), seed)
    Random.seed!(seed)

    tau             = s["HMC"]["tau"]
    nsteps          = s["HMC"]["ns"]
    nthermalize     = s["HMC"]["nthm"]
    ntraj           = s["HMC"]["ntraj"]
    qzero           = s["HMC"]["Qzero"]
    am0             = s["HMC"]["mass"]

    lsize       = Vector{Int64}(undef, 2)
    lsize[1]    = s["Lattice"]["size"][1]
    lsize[2]    = s["Lattice"]["size"][2]
    beta        = s["Lattice"]["beta"]
    
    fb = BDIO_open(s["Run"]["name"]*".bdio", "w",
                   "Data from Master field simulation in QED [2d]; beta = $beta;
                   $(lsize[1]) x $(lsize[2]); seed = $seed")
    BDIO_start_record!(fb, BDIO_ASC_GENERIC, 1, true)
    BDIO_write!(fb, read(fname))
    BDIO_write_hash!(fb)
    
    BDIO_start_record!(fb, BDIO_BIN_F64LE, 2, true)
    dfoo = Vector{Float64}(undef, 4)
    dfoo[1] = lsize[1]
    dfoo[2] = lsize[2]
    dfoo[3] = beta
    dfoo[4] = nthermalize
    BDIO_write!(fb, dfoo)
    BDIO_write_hash!(fb)
    
    return tau, nsteps, nthermalize, ntraj, lsize, beta, am0, qzero, fb
end

parsed_args = parse_commandline()
infile          = parsed_args["i"]
savedir         = parsed_args["savedir"]
continue_from   = parsed_args["cont"]

tau, nsteps, nthermalize, ntraj, lsize, beta, am0, qzero, fb = read_options(infile)

BDIO_close!(fb)

global prm  = LattParm((lsize[1],lsize[2]), beta)
global kprm = KernelParm((lsize[1], 1), (1,lsize[2]))

U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
if continue_from != 0
    load_gauge(U, savedir*"configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(continue_from)", prm)
end

acc = Vector{Int64}()
plaqs = Vector{Float64}()
qtops = Vector{Float64}()

epsilon = tau/nsteps
CGmaxiter = 10000
CGtol = 1e-16

file_stat = savedir*"statistics-hmc-nf2.txt"
io_stat = open(file_stat, "a")
write(io_stat, "step,accepted,")
write(io_stat, "plaquette,top_charge,")
write(io_stat, "acc_rate\n")
close(io_stat)

for i in (1 + continue_from):(ntraj + continue_from)
    @time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, qzero=false)
    Plaquette(U, prm, kprm) |> plaq_U -> push!(plaqs, plaq_U)
	Qtop(U, prm, kprm)      |> qtop_U -> push!(qtops, qtop_U)
    # add reweighting factor

    global io_stat = open(file_stat, "a")
    write(io_stat, "$i,$(acc[end]),")
    write(io_stat, "$(plaqs[end]),$(qtops[end]),")
    write(io_stat, "$(mean(acc))\n")
    close(io_stat)

    if i % 100 == 0
        gauge_file = savedir*"configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0)_n$(i)"
        save_gauge(U, gauge_file, prm)
    end
end

