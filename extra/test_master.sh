#!/bin/sh

echo ">>>> NVIDIA INFO"
set -x #echo on
nvidia-smi

/usr/bin/singularity run --nv --writable /lhome/ific/a/alramos/s.images/julia/ /lhome/ific/a/alramos/s.images/julia/workspace/QED2d/main/master_field.jl -i master_field.in

