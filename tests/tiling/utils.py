import sys
import os
import argparse
import platform
import math

from pyop2.mpi import MPI
from pyop2.profiling import Timer


def parser():
    p = argparse.ArgumentParser(description='Run Seigen with loop tiling')
    # Tiling
    p.add_argument('-n', '--num-unroll', type=int, help='time loop unroll factor', default=1)
    p.add_argument('-z', '--explicit-mode', type=int, help='split chain as [(f, l, ts), ...]', default=0)
    p.add_argument('-t', '--tile-size', type=int, help='initial average tile size', default=5)
    p.add_argument('-e', '--fusion-mode', help='(soft, hard, tile, only_tile)', default='tile')
    p.add_argument('-p', '--part-mode', help='(chunk, metis)', default='chunk')
    p.add_argument('-x', '--extra-halo', type=int, help='add extra halo layer', default=0)
    p.add_argument('-v', '--verbose', help='print additional information', default=False)
    p.add_argument('-l', '--log', help='output inspector to a file', default=False)
    p.add_argument('-g', '--glb-maps', type=bool, help='use global maps (defaults to False)', default=False)
    p.add_argument('-r', '--prefetch', type=bool, help='use software prefetching', default=False)
    p.add_argument('-c', '--coloring', type=bool, help='(default, rand, omp)', default='default')
    # Correctness
    p.add_argument('-d', '--debug', help='debug mode (defaults to False)', default=False)
    p.add_argument('--profile', type=bool, help='enable Python-level profiling', default=False)
    p.add_argument('--check', type=bool, help='check the numerical results', default=False)
    # Simulation
    p.add_argument('-y', '--poly-order', type=int, help='the method\'s order in space', default=2)
    p.add_argument('-f', '--mesh-file', help='use a specific mesh file')
    p.add_argument('-h', '--mesh-spacing', type=float, help='set the mesh spacing', default=2.5)
    p.add_argument('-m', '--mesh-size', help='format: (Lx, Ly)', default=1)
    p.add_argument('-o', '--output', type=int, help='timesteps between two solution field writes', default=False)
    p.add_argument('-cn', '--courant-number', type=float, help='set the courant number', default=0.05)
    p.add_argument('--time-max', type=float, help='set the simulation duration', default=2.5)
    return p.parse_args()


def output_time(start, end, **kwargs):
    verbose = kwargs.get('verbose', False)
    tofile = kwargs.get('tofile', False)
    meshid = kwargs.get('meshid', 'default_mesh')
    nloops = kwargs.get('nloops', 0)
    tile_size = kwargs.get('tile_size', 0)
    partitioning = kwargs.get('partitioning', 'chunk')
    extra_halo = 'yes' if kwargs.get('extra_halo', False) else 'no'
    explicit_mode = kwargs.get('explicit_mode', None)
    glb_maps = 'yes' if kwargs.get('glb_maps', False) else 'no'
    poly_order = kwargs.get('poly_order', -1)
    domain = kwargs.get('domain', 'default_domain')
    coloring = kwargs.get('coloring', 'default')
    prefetch = 'yes' if kwargs.get('prefetch', False) else 'no'
    backend = os.environ.get("SLOPE_BACKEND", "SEQUENTIAL")

    # Where do I store the output ?
    # defaults to /firedrake/demos/tiling/...
    output_dir = ""
    if "FIREDRAKE_DIR" in os.environ:
        output_dir = os.path.join(os.environ["FIREDRAKE_DIR"], "demos", "tiling")

    # Find number of processes, and number of threads per process
    num_procs = MPI.comm.size
    num_threads = int(os.environ.get("OMP_NUM_THREADS", 1)) if backend == 'OMP' else 1

    # What execution mode is this?
    if num_procs == 1 and num_threads == 1:
        versions = ['sequential', 'openmp', 'mpi', 'mpi_openmp']
    elif num_procs == 1 and num_threads > 1:
        versions = ['openmp']
    elif num_procs > 1 and num_threads == 1:
        versions = ['mpi']
    else:
        versions = ['mpi_openmp']

    # Determine the total execution time (Python + kernel execution + MPI cost
    if MPI.comm.rank in range(1, num_procs):
        MPI.comm.isend([start, end], dest=0)
    elif MPI.comm.rank == 0:
        starts, ends = [0]*num_procs, [0]*num_procs
        starts[0], ends[0] = start, end
        for i in range(1, num_procs):
            starts[i], ends[i] = MPI.comm.recv(source=i)
        min_start, max_end = min(starts), max(ends)
        tot = round(max_end - min_start, 3)
        print "Time stepping: ", tot, "s"

    # Determine ACT - Average Compute Time, pure kernel execution -
    # and ACCT - Average Compute and Communication Time (ACS + MPI cost)
    avg = lambda v: (sum(v) / len(v)) if v else 0.0
    compute_time = Timer.get_timers().get('ParLoop kernel', 'VOID').total
    mpi_time = Timer.get_timers().get('ParLoop halo exchange end', 'VOID').total

    if MPI.comm.rank in range(1, num_procs):
        MPI.comm.isend([compute_time, mpi_time], dest=0)
    elif MPI.comm.rank == 0:
        compute_times, mpi_times = [0]*num_procs, [0]*num_procs
        compute_times[0], mpi_times[0] = compute_time, mpi_time
        for i in range(1, num_procs):
            compute_times[i], mpi_times[i] = MPI.comm.recv(source=i)
        ACT = round(avg(compute_times), 3)
        AMT = round(avg(mpi_times), 3)
        ACCT = ACT + AMT
        print "Average Compute Time: ", ACT, "s"
        print "Average Compute and Communication Time: ", ACCT, "s"

    # Determine if a multi-node execution
    is_multinode = False
    platformname = platform.node().split('.')[0]
    if MPI.comm.rank in range(1, num_procs):
        MPI.comm.isend(platformname, dest=0)
    elif MPI.comm.rank == 0:
        all_platform_names = [None]*num_procs
        all_platform_names[0] = platformname
        for i in range(1, num_procs):
            all_platform_names[i] = MPI.comm.recv(source=i)
        if any(i != platformname for i in all_platform_names):
            is_multinode = True
        if is_multinode:
            cluster_island = platformname.split('-')
            platformname = "%s_%s" % (cluster_island[0], cluster_island[1])

    # Adjust /tile_size/ and /version/ based on the problem that was actually run
    assert nloops >= 0
    if nloops == 0:
        tile_size = 0
        mode = "untiled"
    elif explicit_mode:
        mode = "fs%d" % explicit_mode
    else:
        mode = "loops%d" % nloops

    ### Print to file ###

    def fix(values):
        new_values = []
        for v in values:
            try:
                new_v = int(v)
            except ValueError:
                try:
                    new_v = float(v)
                except ValueError:
                    new_v = v.strip()
            if new_v != '':
                new_values.append(new_v)
        return tuple(new_values)

    if MPI.comm.rank == 0 and tofile:
        name = os.path.splitext(os.path.basename(sys.argv[0]))[0]  # Cut away the extension
        for version in versions:
            filename = os.path.join(output_dir, "times", name, "poly_%d" % poly_order, domain,
                                    meshid, version, platformname, "np%d_nt%d.txt" % (num_procs, num_threads))
            # Create directory and file (if not exist)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            if not os.path.exists(filename):
                open(filename, 'a').close()
            # Read the old content, add the new time value, order
            # everything based on <execution time, #loops tiled>, write
            # back to the file (overwriting existing content)
            with open(filename, "r+") as f:
                lines = [line.split('|') for line in f if line.strip()][2:]
                lines = [fix(i) for i in lines]
                lines += [(tot, ACT, ACCT, mode, tile_size, partitioning, extra_halo, glb_maps, coloring, prefetch)]
                lines.sort(key=lambda x: x[0])
                template = "| " + "%12s | " * 10
                prepend = template % ('time', 'ACT', 'ACCT', 'mode', 'tilesize',
                                      'partitioning', 'extrahalo', 'glbmaps', 'coloring', 'prefetch')
                lines = "\n".join([prepend, '-'*151] + [template % i for i in lines]) + "\n"
                f.seek(0)
                f.write(lines)
                f.truncate()

    if verbose:
        for i in range(num_procs):
            if MPI.comm.rank == i:
                tot_time = compute_time + mpi_time
                offC = (end - start) - tot_time
                print "Rank %d: compute=%.2fs, mpi=%.2fs -- tot=%.2fs (py=%.2fs, %.2f%%; mpi_oh=%.2f%%)" % \
                    (i, compute_time, mpi_time, tot_time, offC, (offC / (end - start))*100, (mpi_time / (end - start))*100)
                sys.stdout.flush()
            MPI.comm.barrier()


def calculate_sdepth(num_solves, num_unroll, extra_halo):
    """The sdepth is calculated through the following formula:

        sdepth = 1 if sequential else 1 + num_solves*num_unroll + extra_halo

    Where:

    :arg num_solves: number of solves per loop chain iteration
    :arg num_unroll: unroll factor for the loop chain
    :arg extra_halo: to expose the nonexec region to the tiling engine
    """
    if MPI.parallel and num_unroll > 0:
        return (int(math.ceil(num_solves/2.0)) or 1) + extra_halo
    else:
        return 1


class FusionSchemes(object):

    """
    The fusion schemes attempted in Seigen.

    The format of a fusion scheme is: ::

         (num_solves, [(first_loop_index, last_loop_index, tile_size_multiplier), ...])
    """

    modes = {
        1: (1, [(1, 3, 4), (8, 10, 4), (17, 19, 4)]),
        2: (1, [(1, 2, 4), (4, 6, 4), (8, 9, 4), (10, 12, 2),
                (13, 15, 4), (17, 18, 4), (20, 22, 4), (23, 25, 1)]),
        3: (1, [(1, 3, 4), (4, 7, 4), (8, 12, 2), (13, 16, 4), (17, 19, 4), (20, 25, 1)]),
        4: (2, [(1, 7, 1), (8, 16, 1), (17, 25, 1)]),
        5: (4, [(1, 12, 1), (13, 25, 1)]),
        6: (8, [(1, 25, 1)])
    }

    @staticmethod
    def get(mode, part_mode, tile_size):
        num_solves, mode = FusionSchemes.modes[mode]
        mode = [(i, j, tile_size*(k if part_mode == 'chunk' else 1)) for i, j, k in mode]
        return num_solves, mode
