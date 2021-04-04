# Compact QED in 2d

This is a GPU code for simulating compact QED in 2d using the
HMC.

## `MC.jl`

This is a regular HMC simulation. The code has to be run with the
option `-i <input_file>`. An exaple input file is
```
{
    "Run": {
	"user": "alberto",
	"host": "artemisa",
	"name": "MC_bt11.25_128x128", 
	"seed": 1234
    },
    "Lattice": {
	"size": [128, 128],
	"beta": 11.25
    },
    "HMC": {
	"ns": 10,
	"eps": 0.15,
    "nthm":  2000,
    "ntraj": 2000,
	"Qzero": false
    }
}
```
where the parameters are:
- user: Who is doing the run
- host: in which machine
- name: Name of the run 
- seed: The seed for the RNG.
- size: Size of the lattice
- beta: beta
- ns: number of integration steps per trajectory
- eps: step size in the integration of the eom.
- nthm: Number of trajectories for thermalization
- ntraj: Number of mesaurement trajectories
- Qzero: Reject all updates that make Q different from zero. (for
  simulations at fized topology).

Measurements for the plaquette and the topological charge are stored in
a `BDIO` file. A log of the run is also generated.

An example input file can be found in the `extra` directory.

## `master_field.jl`

This is a code that aims at thermalizing a single/few configurations
and storing the measurements of the plaquette and topological charge
density at each point of space. This is a achieved by a series of
thermalization + unfolding os lattices. 

An example input file is
```
{
    "Run": {
	"user": "alberto",
	"host": "artemisa",
	"name": "test_run", 
	"seed": 1234,
	"measurements": 4
    },
    "Lattice": {
	"size": [8192, 8192],
	"beta": 11.25
    },
    "HMC": {
	"ns": 20,
	"eps": 0.025,
	"ntraj": 2000,
	"Qzero": false
    }
}
```
where the parameters are:
- user: Who is doing the run
- host: in which machine
- name: Name of the run 
- seed: The seed for the RNG.
- measurements: How many independent measurements to make
- size: Size of the lattice
- beta: beta
- ns: number of integration steps per trajectory
- eps: step size in the integration of the eom.
- nthm: Number of trajectories for thermalization
- ntraj: Number of mesaurement trajectories
- Qzero: Reject all updates that make Q different from zero. (for
  simulations at fized topology).

An example input file can be found in the `extra` directory.

# Sendind jobs in artemisa

A singularity image is available in
`/lustre/ific.uv.es/ml/ific051/s.images/julia`. Example script files
to submit jobs are available in `extras`.
