#!/bin/bash

cd $FIREDRAKE_MAIN_DIR
source setenv.env
cd $SEIGEN_DIR

mkdir -p output

OPTS="-log_view --output 10000 --coffee-opt O3"
TILE_OPTS="--fusion-mode only_tile --coloring default"

LOG=""

# Copy the mesh to a local TMP directory
pbsdsh cp $WORK/meshes/wave_elastic/domain$h.msh $TMPDIR
MESHES=$TMPDIR

TSFC_CACHE=$FIREDRAKE_MAIN_DIR/firedrake-cache/tsfc-cache
PYOP2_CACHE=$FIREDRAKE_MAIN_DIR/firedrake-cache/pyop2-cache

export OMP_NUM_THREADS=1
export SLOPE_BACKEND=SEQUENTIAL

# The execution modes
declare -a em_all=(2 3)

# Extra options for each mode
declare -a opts_em1=("--glb-maps")
declare -a opts_em2=("--glb-maps")
declare -a opts_em3=("--glb-maps")

# Tile sizes for each poly order
declare -a ts_p1=(140 250 320 400)
declare -a ts_p2=(70 140 200 300)
declare -a ts_p3=(30 45 60 75)
declare -a ts_p4=(10 20 30 40)

# Partition modes for each poly order
declare -a partitionings=("chunk")

# Meshes
declare -a meshes=("--mesh-file $MESHES/domain$h.msh --mesh-spacing $h")

# The polynomial orders tested
if [ -z "$poly" ]; then
    echo "Warning: testing all polynomial orders [1, 2, 3, 4]"
    declare -a polys=(1 2 3 4)
else
    declare -a polys=($poly)
fi

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

# If only logging tiling stuff, tweak a few things to run only what is strictly necessary
if [ "$1" == "onlylog" ]; then
    declare -a polys=(2)
    declare -a mesh_p2=("--mesh-size (300.0,150.0,1.2)")
    declare -a part_p2=("chunk")
    LOG="--log --time_max 0.05"
    EXTRA_OUT="(Just logging!)"
    mkdir -p all-logs
fi

for poly in ${polys[@]}
do
    output_file="output/output_p"$poly"_h"$h"_"$nodename".txt"
    rm -f $output_file
    touch $output_file
    echo "Polynomial order "$poly
    for mesh in "${meshes[@]}"
    do
        echo "    Running "$mesh
        echo "        Untiled ..."$EXTRA_OUT
        $MPICMD --poly-order $poly $mesh --num-unroll 0 $LOG 1>> $output_file 2>> $output_file
        $MPICMD --poly-order $poly $mesh --num-unroll 0 $LOG 1>> $output_file 2>> $output_file
        $MPICMD --poly-order $poly $mesh --num-unroll 0 $LOG 1>> $output_file 2>> $output_file
        for p in "${partitionings[@]}"
        do
            for em in ${em_all[@]}
            do
                opts="opts_em$em[@]"
                opts_em=( "${!opts}" )
                for opt in "${opts_em[@]}"
                do
                    ts_p="ts_p$poly[*]"
                    for ts in ${!ts_p}
                    do
                        echo "        Tiled (pm="$p", ts="$ts", em="$em") ..."$EXTRA_OUT
                        $MPICMD --poly-order $poly $mesh --num-unroll 1 --tile-size $ts --part-mode $p --explicit-mode $em $TILE_OPTS $opt $LOG 1>> $output_file 2>> $output_file
                        if [ "$1" == "onlylog" ]; then
                            logdir=log_p"$poly"_em"$em"_part"$part"_ts"$ts"
                            mv log $logdir
                            mv $logdir all-logs
                        fi
                    done
                done
            done
        done
    done
done

# Copy output back to $WORK
pbsdsh cp -r $TMPDIR/output $WORK
