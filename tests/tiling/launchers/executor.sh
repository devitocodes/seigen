#!/bin/bash

cd $HOME/NewFiredrake
source setenv.env
cd $SEIGEN_DIR

mkdir -p output

OPTS="-log_view --output 10000 --coffee-opt O3"
TILE_OPTS="--fusion-mode only_tile --coloring default"

LOG=""

MESHES=$WORK/meshes/wave_elastic

TSFC_CACHE=$HOME/firedrake-cache/tsfc-cache
PYOP2_CACHE=$HOME/firedrake-cache/pyop2-cache

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

# Recognized systems: [Erebus (0), CX1-Ivy (1), CX1-Haswell (2)]
if [ "$nodename" -eq 0 ]; then
    nodename="erebus-sandyb"
    MPICMD="mpirun -np 4 --bind-to-core -x FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TSFC_CACHE -x PYOP2_CACHE_DIR=$PYOP2_CACHE -x NODENAME=$nodename"
elif [ "$nodename" -eq 1 ]; then
    nodename="cx1-ivyb"
    MPICMD="mpiexec -env FIREDRAKE_TSFC_KERNEL_CACHE_DIR $TSFC_CACHE -env PYOP2_CACHE_DIR $PYOP2_CACHE -env NODENAME $nodename"
elif [ "$nodename" -eq 2 ]; then
    nodename="cx1-haswell"
    MPICMD="mpiexec -env FIREDRAKE_TSFC_KERNEL_CACHE_DIR $TSFC_CACHE -env PYOP2_CACHE_DIR $PYOP2_CACHE -env NODENAME $nodename"
else
    echo "Unrecognized nodename: $nodename"
    echo "Run as: nodename=integer h=float poly=integer launcher.sh"
    exit
fi

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

export OMP_NUM_THREADS=4
export KMP_AFFINITY=scatter
export SLOPE_BACKEND=OMP
echo "No OMP experiments set"
