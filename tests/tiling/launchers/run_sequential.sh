#!/bin/bash

echo "Running a simple sequential run with sparse tiling activated."
MPICMD="python explosive_source.py --output 100000 --coffee-opt O3 --poly-order 2 --mesh-file ./meshes/domain.msh --mesh-spacing 2.5 --time-max 0.05 --no-tofile"
W_TILING=$MPICMD" --num-unroll 1 --tile-size 150 --part-mode chunk --explicit-mode 2 --fusion-mode only_tile --coloring default --glb-maps"
$W_TILING
