## Performance analysis

### Strong scaling

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

### Spatial degree comparison
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