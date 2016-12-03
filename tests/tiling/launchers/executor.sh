#!/bin/bash

cd $FIREDRAKE_MAIN_DIR
source setenv.env
cd $SEIGEN_DIR

TSFC_CACHE=$FIREDRAKE_MAIN_DIR/firedrake-cache/tsfc-cache
PYOP2_CACHE=$FIREDRAKE_MAIN_DIR/firedrake-cache/pyop2-cache

### System-specific setup - BEGIN ###

# Recognized systems: [Erebus (0), CX1-Ivy (1), CX1-Haswell (2), CX2-Westmere (3), CX2-SandyBridge (4), CX2-Haswell (5), CX2-Broadwell (6)]

function erebus_setup {
    MPICMD="mpirun -np 4 --bind-to-core -x FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TSFC_CACHE -x PYOP2_CACHE_DIR=$PYOP2_CACHE -x NODENAME=$nodename"
}

function cx1_setup {
    WORKDIR=$WORK
    MPICMD="mpiexec -env FIREDRAKE_TSFC_KERNEL_CACHE_DIR $TSFC_CACHE -env PYOP2_CACHE_DIR $PYOP2_CACHE -env NODENAME $nodename"
    module load intel-suite/2016.3
    module load mpi/intel-5.1.1.109
    module load mpi4py/1.3.1
}

function cx2_setup {
    WORKDIR=$SCRATCH
    MPICMD="mpiexec"
    export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TSFC_CACHE
    export PYOP2_CACHE_DIR=$PYOP2_CACHE
    module load gcc
    module load intel-suite
    module load mpi
}

export PYOP2_BACKEND_COMPILER=intel

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

mkdir -p output

MESHES=$WORKDIR/meshes/wave_elastic/

export SLOPE_BACKEND=SEQUENTIAL

# The run modes (dry run, normal run)
declare -a runs=("--output 10000 --time-max 0.05 --no-tofile --coffee-opt O3" "-log_view --output 10000 --coffee-opt O3")

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

MPICMD="$MPICMD python explosive_source.py"

# If only logging tiling stuff, tweak a few things to run only what is strictly necessary
if [ "$1" == "onlylog" ]; then
    declare -a polys=(1)
    declare -a runs=("--output 100000 --time-max 0.05 --no-tofile --coffee-opt O3 --log")
    mkdir -p all-logs
fi

for run in "${runs[@]}"
do
    for poly in ${polys[@]}
    do
        output_file=$WORKDIR"/output_p"$poly"_h"$h"_"$nodename".txt"
        rm -f $output_file
        touch $output_file
        echo "Polynomial order "$poly
        for mesh in "${meshes[@]}"
        do
            PROBLEM="--poly-order $poly $mesh $run"
            echo "    Running "$mesh
            echo "        Untiled ..."
            $MPICMD $PROBLEM --num-unroll 0 1>> $output_file 2>> $output_file
            $MPICMD $PROBLEM --num-unroll 0 1>> $output_file 2>> $output_file
            $MPICMD $PROBLEM --num-unroll 0 1>> $output_file 2>> $output_file
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
                            echo "        Tiled (pm="$p", ts="$ts", em="$em") ..."
                            TILING="--tile-size $ts --part-mode $p --explicit-mode $em --fusion-mode only_tile --coloring default $opt"
                            $MPICMD $PROBLEM --num-unroll 1 $TILING 1>> $output_file 2>> $output_file
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
        mv $output_file "output/"
    done
done

# Copy output back to $WORKDIR
pbsdsh -- cp -r $TMPDIR/output $SCRATCH
