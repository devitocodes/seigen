#!/bin/bash

cd $FIREDRAKE_MAIN_DIR
source setenv.env
cd $SEIGEN_DIR

mkdir -p output

export OMP_NUM_THREADS=1
export SLOPE_BACKEND=SEQUENTIAL

OPTS="--output 10000 --time-max 0.05 --no-tofile --coffee-opt O3"
TILE_OPTS="--fusion-mode only_tile --coloring default"

# Copy the mesh to a local TMP directory
pbsdsh cp $WORK/meshes/wave_elastic/domain$h.msh $TMPDIR
MESHES=$TMPDIR

LOGGER=$TMPDIR"/logger_"$nodename"_cache_populator.txt"
rm -f $LOGGER
touch $LOGGER

export TSFC_CACHE=$TMPDIR/tsfc-cache
export PYOP2_CACHE=$TMPDIR/pyop2-cache

### System-specific setup - BEGIN ###

# Recognized systems: [Erebus (0), CX1-Ivy (1), CX1-Haswell (2), CX2-Westmere (3), CX2-SandyBridge (4), CX2-Haswell (5), CX2-Broadwell (6)]

function erebus_setup {
    MPICMD="mpirun -np 4 --bind-to-core -x FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TSFC_CACHE -x PYOP2_CACHE_DIR=$PYOP2_CACHE -x NODENAME=$nodename"
    export PYOP2_BACKEND_COMPILER=intel
}

function cx1_setup {
    MPICMD="mpiexec -env FIREDRAKE_TSFC_KERNEL_CACHE_DIR $TSFC_CACHE -env PYOP2_CACHE_DIR $PYOP2_CACHE -env NODENAME $nodename"
    export PYOP2_BACKEND_COMPILER=intel
    module load intel-suite/2016.3
    module load mpi/intel-5.1.1.109
    module load mpi4py/1.3.1
}

function cx2_setup {
    export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TSFC_CACHE
    export PYOP2_CACHE_DIR=$PYOP2_CACHE
    MPICMD="mpiexec"
    module load gcc
    module load mpi
    export PYOP2_BACKEND_COMPILER=gnu
    export MPICC_CC=gcc
    export MPICXX_CXX=g++
    export MPIF08_F08=gfortran
    export MPIF90_F90=gfortran
    export PETSC_CONFIGURE_OPTIONS=--download-fblaslapack
    export PATH=$HOME/.local/bin:$PATH
}

if [ "$nodename" -eq 0 ]; then
    export nodename="erebus-sandyb"
    export PYOP2_SIMD_ISA=avx
    erebus_setup
elif [ "$nodename" -eq 1 ]; then
    export nodename="cx1-ivyb"
    export PYOP2_SIMD_ISA=avx
    cx1_setup
elif [ "$nodename" -eq 2 ]; then
    export nodename="cx1-haswell"
    export PYOP2_SIMD_ISA=avx
    cx1_setup
elif [ "$nodename" -eq 3 ]; then
    export nodename="cx2-westmere"
    export PYOP2_SIMD_ISA=sse
    cx2_setup
elif [ "$nodename" -eq 4 ]; then
    export nodename="cx2-sandyb"
    export PYOP2_SIMD_ISA=avx
    cx2_setup
elif [ "$nodename" -eq 5 ]; then
    export nodename="cx2-haswell"
    export PYOP2_SIMD_ISA=avx
    cx2_setup
elif [ "$nodename" -eq 6 ]; then
    export nodename="cx2-broadwell"
    export PYOP2_SIMD_ISA=avx
    cx2_setup
else
    echo "Unrecognized nodename: $nodename"
    echo "Run as: nodename=integer h=float poly=integer executor.sh"
    exit
fi

### System-specific setup - END ###

MPICMD="$MPICMD python explosive_source.py $OPTS"

declare -a polys=($poly)

declare -a opts_em1=("--glb-maps")
declare -a opts_em2=("--glb-maps")
declare -a opts_em3=("--glb-maps")

declare -a part_all=("chunk")

declare -a mesh_p1=("--mesh-size (50.0,25.0) --mesh-spacing $h")
declare -a mesh_p2=("--mesh-size (30.0,15.0) --mesh-spacing $h")
declare -a mesh_p3=("--mesh-size (30.0,15.0) --mesh-spacing $h")
declare -a mesh_p4=("--mesh-size (30.0,15.0) --mesh-spacing $h")

declare -a mesh_default=("--mesh-file $MESHES/domain1.0.msh --mesh-spacing 1.0")

declare -a em_all=(2 3)

# Populate the local cache
for poly in ${polys[@]}
do
    output_file="output/populator_p"$poly"_h"$h"_"$nodename".txt"
    rm -f $output_file
    touch $output_file
    echo "Populate polynomial order "$poly >> $LOGGER
    mesh_p="mesh_p$poly[@]"
    meshes=( "${!mesh_p}" )
    for mesh in "${meshes[@]}"
    do
        echo "    Populate "$mesh >> $LOGGER
        echo "        Populate Untiled ..." >> $LOGGER
        $MPICMD --poly-order $poly $mesh --num-unroll 0 1>> $output_file 2>> $output_file
        $MPICMD --poly-order $poly $mesh_default --num-unroll 0 1>> $output_file 2>> $output_file  # Create the expression kernels
        for p in ${part_all[@]}
        do
            for em in ${em_all[@]}
            do
                opts="opts_em$em[@]"
                opts_em=( "${!opts}" )
                for opt in "${opts_em[@]}"
                do
                    ts=100000
                    echo "        Populate Tiled (pm="$p", ts="$ts", em="$em") ..." >> $LOGGER
                    $MPICMD --poly-order $poly $mesh --num-unroll 1 --tile-size $ts --part-mode $p --explicit-mode $em $TILE_OPTS $opt 1>> $output_file 2>> $output_file
                    $MPICMD --poly-order $poly $mesh_default --num-unroll 1 --tile-size $ts --part-mode $p --explicit-mode $em $TILE_OPTS $opt 1>> $output_file 2>> $output_file
                done
            done
        done
    done
done

rm $LOGGER

# Copy the local cache to the shared file system
mkdir -p $FIREDRAKE_MAIN_DIR/firedrake-cache/pyop2-cache
mkdir -p $FIREDRAKE_MAIN_DIR/firedrake-cache/tsfc-cache
cp -n $PYOP2_CACHE/* $FIREDRAKE_MAIN_DIR/firedrake-cache/pyop2-cache
cp -n $TSFC_CACHE/* $FIREDRAKE_MAIN_DIR/firedrake-cache/tsfc-cache
