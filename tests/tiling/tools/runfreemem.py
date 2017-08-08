import sys
import subprocess
from mpi4py import MPI

corespernode = int(sys.argv[1])
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
num_procs = MPI.COMM_WORLD.size

if rank % corespernode == 0:
    subprocess.call(["./tools/freemem"])
    print "[Python-land] Rank %d: DRAM freed." % rank
    subprocess.call(["numastat -m | grep MemFree"], shell=True)
