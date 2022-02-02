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
        help = "path to directory to save configurations and logs"
        required = true
        arg_type = String
        # default = "configs/"
        "--cont"
        help = "continue from configuration number; 0 if new run"
        required = true
        arg_type = Int
        # default = 0
    end

    return parse_args(s)
end

function read_options(fname)

    io = open(fname, "r")
    s = JSON.parse(io)
    close(io)

    BDIO_set_user(s["Run"]["user"])
    BDIO_set_host(s["Run"]["host"])
    
    # io = open(s["Run"]["name"]*".log", "w+")
    # global_logger(TerminalLogger(io))

    if s["Run"]["seed"] != 0
        seed = s["Run"]["seed"]
    else
        seed = rand(Int64) |> abs
    end
    Random.seed!(CURAND.default_rng(), seed)
    Random.seed!(seed)


    tau             = s["HMC"]["tau"]
    nsteps          = s["HMC"]["ns"]
    nthermalize     = s["HMC"]["nthm"]
    ntraj           = s["HMC"]["ntraj"]
    qzero           = s["HMC"]["Qzero"]

    am0             = s["RHMC"]["masses"]
    am0             = convert(Array{Float64},am0)
    n_rhmc          = s["RHMC"]["n_rhmc"]
    r_a_rhmc        = s["RHMC"]["r_a_g5Dw_sqr"] .|> sqrt
    r_b_rhmc        = s["RHMC"]["r_b_g5Dw_sqr"] .|> sqrt
    
    lsize       = Vector{Int64}(undef, 2)
    lsize[1]    = s["Lattice"]["size"][1]
    lsize[2]    = s["Lattice"]["size"][2]
    beta        = s["Lattice"]["beta"]
    
    # fb = BDIO_open(s["Run"]["name"]*".bdio", "w",
    #                "Data from Master field simulation in QED [2d]; beta = $beta;
    #                $(lsize[1]) x $(lsize[2]); seed = $seed")
    # BDIO_start_record!(fb, BDIO_ASC_GENERIC, 1, true)
    # BDIO_write!(fb, read(fname))
    # BDIO_write_hash!(fb)
    
    # BDIO_start_record!(fb, BDIO_BIN_F64LE, 2, true)
    # dfoo = Vector{Float64}(undef, 4)
    # dfoo[1] = lsize[1]
    # dfoo[2] = lsize[2]
    # dfoo[3] = beta
    # dfoo[4] = nthermalize
    # BDIO_write!(fb, dfoo)
    # BDIO_write_hash!(fb)
    
    # return tau, nsteps, nthermalize, ntraj, lsize, beta, am0, n_rhmc, r_a_rhmc, r_b_rhmc, qzero, fb
    return tau, nsteps, nthermalize, ntraj, lsize, beta, am0, n_rhmc, r_a_rhmc, r_b_rhmc, qzero, seed
end

parsed_args = parse_commandline()
infile          = parsed_args["i"]
savedir         = parsed_args["savedir"]
continue_from   = parsed_args["cont"]

tau, nsteps, nthermalize, ntraj, lsize, beta, am0, n_rhmc, r_a_rhmc, r_b_rhmc, qzero, seed = read_options(infile)

# BDIO_close!(fb)

global prm  = LattParm((lsize[1],lsize[2]), beta)
global kprm = KernelParm((lsize[1], 1), (1,lsize[2]))

U = CUDA.ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
if continue_from != 0
    load_gauge(U, savedir*"configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0[1])_n$(continue_from)", prm)
end

rprm = get_rhmc_params(n_rhmc, r_a_rhmc, r_b_rhmc)

acc = Vector{Int64}()
reweight = Vector{Float64}()
plaqs = Vector{Float64}()
qtops = Vector{Float64}()

epsilon = tau/nsteps
CGmaxiter = 10000
CGtol = 1e-16

file_stat   = "statistics.txt"
file_log    = "log.txt"
if continue_from == 0
    mkpath(savedir*"configs/")

    io_stat = open(savedir*file_stat, "w")
    write(io_stat, "step,accepted,")
    write(io_stat, "plaquette,top_charge,reweight_factor,")
    write(io_stat, "lambda_min,lambda_max,acc_rate\n")
    close(io_stat)

    io_log = open(savedir*file_log, "w")
    write(io_log, "Start RHMC run in QED2d with beta = $beta; $(lsize[1]) x
          $(lsize[2]); masses = $am0; n_rhmc = $n_rhmc, r_a = $r_a_rhmc, r_b =
          $r_b_rhmc; ntraj = $ntraj; seed = $seed\n")
    close(io_log)
else
    io_log = open(savedir*file_log, "a")
    write(io_log, "Continue RHMC run from n=$continue_from with beta = $beta;
          $(lsize[1]) x $(lsize[2]); masses = $am0; n_rhmc = $n_rhmc, r_a =
          $r_a_rhmc, r_b = $r_b_rhmc; ntraj = $ntraj; seed = $seed\n")
    close(io_log)
end

for i in (1 + continue_from):(ntraj + continue_from)
    @time HMC!(U, am0, epsilon, nsteps, acc, CGmaxiter, CGtol, prm, kprm, rprm, qzero=false)
    Plaquette(U, prm, kprm) |> plaq_U -> push!(plaqs, plaq_U)
	Qtop(U, prm, kprm)      |> qtop_U -> push!(qtops, qtop_U)
    # println("Last plaquette: $(plaqs[end])")
    # println("Last Q: $(qtops[end])")
    # add reweighting factor
    if acc[end] == 0
        push!(reweight, reweight[end])
    else
        reweighting_factor(U, am0, CGmaxiter, CGtol, prm, kprm, rprm) |> x -> push!(reweight, x)
    end
    lambda_min, lambda_max = power_method(U, am0[1], prm, kprm, iter=5000) .|> real
    # println("λ_min = $lambda_min, \nλ_max = $lambda_max")
    
    global io_stat = open(savedir*file_stat, "a")
    write(io_stat, "$i,$(acc[end]),")
    write(io_stat, "$(plaqs[end]),$(qtops[end]),$(reweight[end]),")
    write(io_stat, "$lambda_min,$lambda_max,$(mean(acc))\n")
    close(io_stat)

    if i % 10 == 0
        gauge_file = savedir*"configs/config_$(prm.iL[1])_$(prm.iL[2])_b$(prm.beta)_m$(am0[1])_n$(i)"
        save_gauge(U, gauge_file, prm)
    end
end
