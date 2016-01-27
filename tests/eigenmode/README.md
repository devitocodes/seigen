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
python eigenmode_plot.py comparison -i <arch>/results -o <arch>/plots -p 16 --dim 2 -N 2562 --degree 1 2 3 4 --opt 2 3 4
```
