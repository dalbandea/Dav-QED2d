#!/bin/sh

echo ">>>> NVIDIA INFO"
set -x #echo on
nvidia-smi

/usr/bin/singularity run --nv --writable /lhome/ific/a/alramos/s.images/julia/ /lhome/ific/a/alramos/s.images/julia/workspace/QED2d/main/MC.jl -i MC.in

