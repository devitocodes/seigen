# Summary

This test case is used to perform a convergence analysis in order to verify the correctness of the model's implementation, and to demonstrate that high-order convergence can be attained with the solution method. A time and space-dependent solution involving the propagation of an eigenmode is considered here. Four different structured unit meshes of increasing resolution are used to resolve the solution (dx = 0.25, 0.125, 0.0625, 0.03125). For each mesh, piecewise-discontinuous polynomial basis functions of degree 1, 2, 3 and 4 (also known as P1, P2, P3 and P4 basis functions) are employed, yielding a total of 16 simulations. Both two- and three-dimensional versions of this test case exist in this directory.

To run the 2D and 3D simulations, respectively, run:

python eigenmode_2d.py
python eigenmode_3d.py

Other Python scripts exist for the purposes of performance analysis; these are described below.

# Performance analysis

## Strong scaling

On the cluster run:
```
for NP in 1 2 4 8 16; do
  for SIZE in 256; do
    for DEGREE in 4; do
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 explicit=True opt=4 T=2.0 degree=$DEGREE N=$SIZE
    done
  done
done
```
Plot locally with:
```
python eigenmode_plot.py strong -i <arch>/results -o <arch>/plots -p 1 2 4 8 16 --dim 2 -N 256 --degree 4 --opt 4
```

## Spatial degree comparison
On the cluster run:
```
for NP in 16; do
  for SIZE in 256; do
    for DEGREE in 1 2 3 4; do
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 explicit=False opt=2 T=2.0 degree=$DEGREE N=$SIZE
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 explicit=True opt=2 T=2.0 degree=$DEGREE N=$SIZE
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 explicit=True opt=3 T=2.0 degree=$DEGREE N=$SIZE
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 explicit=True opt=4 T=2.0 degree=$DEGREE N=$SIZE
    done
  done
done
```
