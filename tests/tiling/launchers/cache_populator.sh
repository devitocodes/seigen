export OMP_NUM_THREADS=1
export SLOPE_BACKEND=SEQUENTIAL

OPTS="--output 10000 --time_max 0.01 --to-file False"
TILE_OPTS="--fusion-mode only_tile --coloring default"
TMPDIR=/tmp

LOGGER=$TMPDIR"/logger_"$nodename"_cache_populator.txt"
rm -f $LOGGER
touch $LOGGER

export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$TMPDIR/tsfc-cache
export PYOP2_CACHE_DIR=$TMPDIR/pyop2-cache

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

declare -a polys=($poly)

declare -a opts_em1=("")
declare -a opts_em2=("")
declare -a opts_em3=("")

declare -a part_all=("chunk")

declare -a mesh_p1=("--mesh-size (50.0,25.0) --mesh-spacing $h")
declare -a mesh_p2=("--mesh-size (30.0,15.0) --mesh-spacing $h")
declare -a mesh_p3=("--mesh-size (30.0,15.0) --mesh-spacing $h")
declare -a mesh_p4=("--mesh-size (30.0,15.0) --mesh-spacing $h")

declare -a mesh_default=("--mesh-size (300.0,150.0) --mesh-spacing 1.0")

declare -a em_all=(1 2 3)

# Populate the local cache
for poly in ${polys[@]}
do
    echo "Populate polynomial order "$poly >> $LOGGER
    mesh_p="mesh_p$poly[@]"
    meshes=( "${!mesh_p}" )
    for mesh in "${meshes[@]}"
    do
        echo "    Populate "$mesh >> $LOGGER
        echo "        Populate Untiled ..." >> $LOGGER
        $MPICMD --poly-order $poly $mesh --num-unroll 0
        $MPICMD --poly-order $poly $mesh_default --num-unroll 0  # Create the expression kernels
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
                    $MPICMD --poly-order $poly $mesh --num-unroll 1 --tile-size $ts --part-mode $p --explicit-mode $em $TILE_OPTS $opt
                    $MPICMD --poly-order $poly $mesh_default --num-unroll 1 --tile-size $ts --part-mode $p --explicit-mode $em $TILE_OPTS $opt
                done
            done
        done
    done
done

rm $LOGGER

# Copy the local cache to the shared file system
mkdir -p $HOME/firedrake-cache/pyop2-cache
mkdir -p $HOME/firedrake-cache/tsfc-cache
cp -n $PYOP2_CACHE_DIR/* $HOME/firedrake-cache/pyop2-cache
cp -n $FIREDRAKE_TSFC_KERNEL_CACHE_DIR/* $HOME/firedrake-cache/tsfc-cache
