## Performance analysis

### Strong scaling

On the cluster run:
```
for NP in 1 2 4 8 16; do
  for SIZE in 256; do
    for DEGREE in 4; do
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=4 T=2.0 degree=$DEGREE N=$SIZE
    done
  done
done
```
Plot locally with:
```
python eigenmode_plot.py strong -i <arch>/results -o <arch>/plots -p 1 2 4 8 16 --dim 2 -N 256 --degree 4 --opt 4
```

### Spatial degree comparison
On the cluster run:
```
for NP in 16; do
  for SIZE in 256; do
    for DEGREE in 1 2 3 4; do
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=implicit opt=2 T=2.0 degree=$DEGREE N=$SIZE
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=2 T=2.0 degree=$DEGREE N=$SIZE
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=3 T=2.0 degree=$DEGREE N=$SIZE
      <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=4 T=2.0 degree=$DEGREE N=$SIZE
    done
  done
done
```

```
python eigenmode_plot.py comparison -i <arch>/results -o <arch>/plots -p 16 --dim 2 -N 256 --degree 1 2 3 4 --opt 2 3 4
```

### Error-cost plots
On the cluster run:
```
export NP=16
for SIZE in 32 64 128; do <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=4 T=5.0 degree=1 N=$SIZE dt=-1; done
for SIZE in 16 32 64; do <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=4 T=5.0 degree=2 N=$SIZE dt=-1; done
for SIZE in 8 16 32; do <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=4 T=5.0 degree=3 N=$SIZE dt=-1; done
for SIZE in 4 8 16; do <mpiexec> -n $NP python eigenmode_bench.py -b -l -s -- dim=2 solver=explicit opt=4 T=5.0 degree=4 N=$SIZE dt=-1; done
```

Then plot with:
```
python eigenmode_plot.py error -i <arch>/results -o <arch>/plots -p 16 --dim 2 -N 4 8 16 32 64 128 --degree 1 2 3 4 -T=5.0 --dt=-1.0 --opt 4
```

### Kernel-based roofline plots
First get stream results via PETSc:
```
cd $PETSC_DIR/src/benchmarks/streams
make stream NPMAX=<NP> MPI_BINDING="--bysocket --bind-to-socket"
```
