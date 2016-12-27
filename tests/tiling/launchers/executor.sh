#!/bin/bash

cd $FIREDRAKE_MAIN_DIR
source setenv.env
cd $SEIGEN_DIR

TSFC_CACHE=$FIREDRAKE_MAIN_DIR/firedrake-cache/tsfc-cache
PYOP2_CACHE=$FIREDRAKE_MAIN_DIR/firedrake-cache/pyop2-cache

### System-specific setup - BEGIN ###

# Recognized systems: [Erebus (0), CX1-Ivy (1), CX1-Haswell (2), CX2-Westmere (3), CX2-SandyBridge (4), CX2-Haswell (5), CX2-Broadwell (6)]

function erebus_setup {
    MPIOPTS="-np 4 --bind-to-core -x FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TSFC_CACHE -x PYOP2_CACHE_DIR=$PYOP2_CACHE -x NODENAME=$1"
    MPICMD="mpirun -np 4 $MPIOPTS"
    MPICMD_SN=$MPICMD
}

function cx1_setup {
    WORKDIR=$WORK
    MPIOPTS="-env FIREDRAKE_TSFC_KERNEL_CACHE_DIR $TSFC_CACHE -env PYOP2_CACHE_DIR $PYOP2_CACHE -env NODENAME $1"
    MPICMD="mpiexec $MPIOPTS"
    MPICMD_SN="mpirun -genv I_MPI_DEVICE rdssm -genv I_MPI_DEBUG 0 -genv I_MPI_FALLBACK_DEVICE 0 -genv DAPL_MAX_CM_RETRIES 500 -genv DAPL_MAX_CM_RESPONSE_TIME 300 -genv I_MPI_DAPL_CONNECTION_TIMEOUT 300 -genv I_MPI_PIN yes -genv I_MPI_PIN_MODE lib -genv I_MPI_PIN_PROCESSOR_LIST allcores:map=bunch $MPIOPTS -n 20"
    module load intel-suite/2016.3
    module load mpi/intel-5.1.1.109
    module load mpi4py/1.3.1
    export CC="mpicc"
    export MPICC="mpicc"
    export CXX="mpicc"
    export MPICXX="mpicc"
    export FC="ifort"
    export MPIFC="ifort"
    export F77="ifort"
    export MPIF77="ifort"
    export F90="ifort"
    export MPIF90="ifort"
}

function cx2_setup {
    WORKDIR=$SCRATCH
    MPICMD="mpiexec"
    MPICMD_SN="mpiexec -n $2"
    export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TSFC_CACHE
    export PYOP2_CACHE_DIR=$PYOP2_CACHE
    export NODENAME=$1
    module load gcc
    module load intel-suite
    module load mpi
}

export PYOP2_BACKEND_COMPILER=intel
export PYOP2_SIMD_ISA=avx

if [ "$nodename" -eq 0 ]; then
    erebus_setup "erebus-sandyb"
elif [ "$nodename" -eq 1 ]; then
    cx1_setup "cx1-ivyb"
elif [ "$nodename" -eq 2 ]; then
    cx1_setup "cx1-haswell"
elif [ "$nodename" -eq 3 ]; then
    export PYOP2_SIMD_ISA=sse
    cx2_setup "cx2-westmere" 12
elif [ "$nodename" -eq 4 ]; then
    cx2_setup "cx2-sandyb" 16
elif [ "$nodename" -eq 5 ]; then
    cx2_setup "cx2-haswell" 24
elif [ "$nodename" -eq 6 ]; then
    cx2_setup "cx2-broadwell" 24
else
    echo "Unrecognized nodename: $nodename"
    echo "Run as: nodename=integer h=float poly=integer executor.sh"
    exit
fi

### System-specific setup - END ###

mkdir -p output

MESHES=$WORKDIR/meshes/wave_elastic/

export SLOPE_BACKEND=SEQUENTIAL

MPICMD="$MPICMD python explosive_source.py"
MPICMD_SN="$MPICMD_SN python explosive_source.py"

# Three runs: [populate cache, populate cache, normal run]
declare -a runs=("$MPICMD_SN --output 100000 --time-max 0.05 --no-tofile --coffee-opt O3 --mesh-size (50.0,25.0) --mesh-spacing $h"
                 "$MPICMD_SN --output 100000 --time-max 0.05 --no-tofile --coffee-opt O3 --mesh-file $MESHES/domain1.0.msh --mesh-spacing 1.0"
                 "$MPICMD -log_view --output 100000 --coffee-opt O3 --mesh-file $MESHES/domain$h.msh --mesh-spacing $h")

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

# The polynomial orders tested
if [ -z "$poly" ]; then
    echo "Warning: testing all polynomial orders [1, 2, 3, 4]"
    declare -a polys=(1 2 3 4)
else
    declare -a polys=($poly)
fi

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
        output_file=$WORKDIR"/output_p"$poly"_h"$h"_"$NODENAME".txt"
        rm -f $output_file
        touch $output_file
        RUN="$run --poly-order $poly"
        echo "    Running "$RUN

        echo "        Untiled ..."
        $RUN --num-unroll 0 1>> $output_file 2>> $output_file
        $RUN --num-unroll 0 1>> $output_file 2>> $output_file
        $RUN --num-unroll 0 1>> $output_file 2>> $output_file

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
                        $RUN --num-unroll 1 $TILING 1>> $output_file 2>> $output_file
                        if [ "$1" == "onlylog" ]; then
                            logdir=log_p"$poly"_em"$em"_part"$part"_ts"$ts"
                            mv log $logdir
                            mv $logdir all-logs
                        fi
                    done
                done
            done
        done
        mv $output_file "output/"
    done
done

# Copy output back to $WORKDIR
mv $TMPDIR/output $SCRATCH
mv $SCRATCH/output $SCRATCH/output_p$poly
