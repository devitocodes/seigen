from eigenmode_2d import Eigenmode2DLF4
from eigenmode_3d import Eigenmode3DLF4
from firedrake import *
from pyop2.mpi import MPI
from firedrake.petsc import PETSc
from os import path, getcwd
from itertools import product
from collections import OrderedDict
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import numpy as np
from opescibench import Executor

try:
    from opescibench import Plotter, LinePlotter, BarchartPlotter, AxisScale
except:
    Plotter = None
    LinePlotter = None
    BarchartPlotter = None


parameters["pyop2_options"]["profiling"] = True
parameters["coffee"]["O2"] = False
parameters["seigen"] = {}
parameters["seigen"]["profiling"] = True

field_stages = {
    'all': ['uh1', 'stemp', 'uh2', 'u1', 'sh1', 'utemp', 'sh2', 's1'],
    'velocity': ['uh1', 'stemp', 'uh2', 'u1'],
    'stress': ['sh1', 'utemp', 'sh2', 's1']
}
petsc_events = ['MatMult', 'ParLoopExecute']

ploop_types = {'cell': 'form0_cell_integral_otherwise',
               'intfac': 'form0_interior_facet_integral_otherwise',
               'extfac': 'form0_exterior_facet_integral_otherwise'}


class EigenmodeExecutor(Executor):

    def run(self, dim, size, time, dt, degree, solver, opt, **kwargs):
        # If dt is supressed (<0) infer it based on Courant number 0.5
        if dt < 0:
            dt = 0.5 * (1.0 / size) / (2.0**(degree-1))

        # Set COFFEE optimisation flags
        parameters["coffee"]["O2"] = opt >= 2
        parameters["coffee"]["O3"] = opt >= 3
        parameters["coffee"]["O4"] = opt >= 4

        # Start PETSc performance logging
        PETSc.Log().begin()

        if dim == 2:
            self.eigen = Eigenmode2DLF4(size, degree, dt, solver=solver, output=False)
            self.u1, self.s1 = self.eigen.eigenmode2d(T=time)
        elif dim == 3:
            self.eigen = Eigenmode3DLF4(size, degree, dt, solver=solver, output=False)
            self.u1, self.s1 = self.eigen.eigenmode3d(T=time)

        # Dump PETSc performance log info to temporary file
        logfile = path.join(getcwd(), 'petsclog.py')
        vwr = PETSc.Viewer().createASCII(logfile)
        vwr.pushFormat(PETSc.Viewer().Format().ASCII_INFO_DETAIL)
        PETSc.Log().view(vwr)

        # Read performance results and register results
        if MPI.COMM_WORLD.rank == 0:
            petsclog = {}
            execfile(logfile, globals(), petsclog)
            for stage, stagelog in petsclog['Stages'].items():
                for event, eventlog in stagelog.items():
                    time = max([eventlog[p]['time'] for p in range(petsclog['numProcs'])])
                    if time > 0.0:
                        self.register(time, event='%s:%s' % (stage, event), measure='time')
                    flops = sum([eventlog[p]['flops'] for p in range(petsclog['numProcs'])])
                    if flops > 0.0:
                        self.register(flops, event='%s:%s' % (stage, event), measure='flops')

        # Store meta data
        num_dofs = MPI.COMM_WORLD.allreduce(self.eigen.elastic.S.dof_count, op=MPI.SUM)
        if MPI.COMM_WORLD.rank == 0:
            self.meta['dofs'] = num_dofs
            self.meta['spacing'] = 1. / size
            self.meta['kernel_profile'] = parameters['seigen']['profiling']

    def postprocess(self, **kwargs):
        """ Error estimates are compute intensive, so only run them once. """
        if self.no_error:
            return
        try:
            u_error, s_error = self.eigen.eigenmode_error(self.u1, self.s1)
            self.meta['velocity_error'] = u_error
            self.meta['stress_error'] = s_error
        except RuntimeError:
            print "WARNING: Couldn't establish error norm"
            self.meta['velocity_error'] = 'NaN'
            self.meta['stress_error'] = 'NaN'


if __name__ == '__main__':
    from opescibench import Benchmark
    op2.init(log_level='ERROR')
    description = ("Benchmarking script for Eigenmode example - modes:\n" +
                   "\tbench: Run benchmark with given parameters\n" +
                   "\tplot: Plot results from the benchmark\n")
    parser = ArgumentParser(description=description,
                            formatter_class=RawDescriptionHelpFormatter)
    # subparsers = parser.add_subparsers(dest='mode', help="Mode of operation")
    # parser_bench = subparsers.add_parser('bench', help='Perform benchmarking runs on target machine')
    # parser_plot = subparsers.add_parser('plot', help='Plot diagrams from stored results')

    parser.add_argument(dest="mode", nargs="?", default="bench",
                        choices=["bench", "plot"], help="Execution mode")
    parser.add_argument('-i', '--resultsdir', default='results',
                        help='Directory containing results')

    simulation = parser.add_argument_group("Simulation")
    simulation.add_argument('--dim', type=int, default=2, help='Problem dimension')
    simulation.add_argument('--size', type=int, nargs='+', default=[8],
                            help='Mesh size (number of edges per dimension)')
    simulation.add_argument('--time', type=float, nargs=1, default=5.0,
                            help='Total runtime in seconds')
    simulation.add_argument('--dt', type=float, nargs=1, default=-1.,
                            help='Timestep size in seconds; auto-derived if dt < 0.')
    simulation.add_argument('--degree', type=int, nargs='+', default=[1],
                            help='Degree of spatial discretisation')
    simulation.add_argument('--nprocs', type=int, nargs='+', default=[MPI.COMM_WORLD.size] or [1],
                            help='Number of parallel processes')
    simulation.add_argument('--solver', choices=('implicit', 'explicit'), nargs='+',
                            default=['explicit'], help='Solver method used')
    simulation.add_argument('--opt', type=int, nargs='+', default=[3],
                            help='Coffee optimisation level; default -O3')
    simulation.add_argument('--no_error', action='store_true', default=False,
                            help='Suppress error computation')

    # Additional arguments for plotting
    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument('--plottype', default='error',
                          choices=('error', 'strong', 'roofline', 'compare'),
                          help='Type of plot to generate: error-cost or roofline')
    plotting.add_argument('--field', default='all',
                          choices=('all', 'velocity', 'stress'),
                          help='Type of plot to generate: error-cost or roofline')
    plotting.add_argument('-o', '--plotdir', default='plots',
                          help='Directory to store generated plots')
    plotting.add_argument('--max-bw', metavar='max_bw', type=float,
                          help='Maximum memory bandwidth for roofline plots')
    plotting.add_argument('--max-flops', metavar='max_flops', type=float,
                          help='Maximum flop rate for roofline plots')
    plotting.add_argument('--kernel', type=str, nargs='+', default=['uh1'],
                          help='Name of kernels for roofline plots')

    args = parser.parse_args()
    params = vars(args).copy()
    del params["mode"]
    del params["plottype"]
    del params["resultsdir"]
    del params["plotdir"]
    del params["max_bw"]
    del params["max_flops"]
    del params["kernel"]
    del params["field"]
    del params["no_error"]

    bench = Benchmark(parameters=params, resultsdir=args.resultsdir,
                      name='EigenmodeBench')

    if args.mode == 'bench':
        # Run the model across the parameter sweep and save the result
        executor = EigenmodeExecutor()
        executor.no_error = args.no_error
        bench.execute(executor, warmups=1, repeats=3)
        bench.save()
    elif args.mode == 'plot':
        plotter = Plotter(plotdir=args.plotdir)
        bench.load()
        if not bench.loaded:
            warning("Could not load any results, nothing to plot. Exiting...")
            sys.exit(0)

        stages = field_stages[args.field]

        if args.plottype == 'strong':
            figname = 'SeigenStrong_%s.pdf' % args.field
            effname = 'SeigenEfficiency_%s.pdf' % args.field
            events = ['%s:summary' % s for s in stages]
            # Plot strong scalability
            with LinePlotter(figname=figname, plotdir=args.plotdir,
                             xlabel='Number of processors',
                             ylabel='Wall time (s)') as scaling:
                # Sweep over everything but 'nprocs'
                for key in bench.sweep(keys={'nprocs': params['nprocs'][0]}):
                    time = []
                    for p in params['nprocs']:
                        key['nprocs'] = p
                        time += [sum(bench.lookup(params=key, measure='time',
                                                  event=events).values()[0])]
                    scaling.add_line(params['nprocs'], time)

            # Plot parallel efficiency
            with LinePlotter(figname=effname, plotdir=args.plotdir,
                             xlabel='Number of processors',
                             ylabel='Parallel efficiency',
                             plot_type='semilogx', ytype=np.float32) as efficiency:
                # Sweep over everything but 'nprocs'
                for key in bench.sweep(keys={'nprocs': params['nprocs'][0]}):
                    time = []
                    for p in params['nprocs']:
                        key['nprocs'] = p
                        time += [sum(bench.lookup(params=key, measure='time',
                                                  event=events).values()[0])]
                    eff = [t / time[0] for t in time]
                    efficiency.add_line(params['nprocs'], eff)

        elif args.plottype == 'error':
            # Plot error-cost diagram
            assert args.field in ['velocity', 'stress']
            figname = 'SeigenError_%s.pdf' % args.field
            events = ['%s:summary' % s for s in stages]
            with LinePlotter(figname=figname, plotdir=args.plotdir,
                             xscale=AxisScale(scale='log', base=10., dtype=np.float32),
                             xlabel='Error in L2 norm',
                             yscale=AxisScale(scale='log', base=2., dtype=np.int32),
                             ylabel='Wall time (s)',
                             legend={'ncol': 4},
                             plot_type='loglog') as error_cost:
                # Sweep over degrees
                for d in params['degree']:
                    time = []
                    error = []
                    annotations = []
                    for key in bench.sweep(keys={'degree': d}):
                        ev_time = bench.lookup(params=key, measure='time',
                                               event=events).values()
                        try:
                            err = bench.lookup(params=key, category='meta',
                                               measure='%s_error' % args.field).values()
                        except:
                            # Skip this point if no error values are available
                            continue
                        if len(ev_time) > 0:
                            time += [sum(ev_time[0])]
                            error += err
                            annotations += ['dx=%0.3f' % (1. / key['size'])]
                    style = '-%s%s' % (error_cost.marker[d-1], error_cost.color[d-1])
                    label = r'P$_{%d}$DG' % d
                    error_cost.add_line(error, time, style=style, label=label,
                                        annotations=annotations)

        elif args.plottype == 'roofline':
            for kernel in args.kernel:
                for pltype, plname in ploop_types.items():
                    figname = 'SeigenRoofline_%s_%s.pdf' % (kernel, pltype)
                    flops_s = {}
                    op_int = {}
                    for deg, opt in product(args.degree, args.opt):
                        params = {'degree': deg, 'opt': opt}
                        event = '%s:%s' % (kernel, plname)
                        label = u'P%d$_{DG}$: %s' % (deg, 'Raw' if opt == 0 else 'Opt')
                        profile = bench.lookup(params=params, measure='kernel_profile',
                                               category='meta')[0][kernel]
                        if plname in profile:
                            time = bench.lookup(params=params, event=event, measure='time')
                            flops = bench.lookup(params=params, event=event, measure='flops')
                            assert(len(time) == 1 and len(flops) == 1)
                            flops_s[label] = (flops[0] / 1.e6 / time[0])
                            op_int[label] = profile[plname]['ai']
                    if len(flops_s) > 0:
                        plotter.plot_roofline(figname, flops_s, op_int)

        elif args.plottype == 'compare':
            # Barchart comparison of optimised vs. unoptimised runs on
            # different discretisations
            figname = 'SeigenCompare.pdf'
            events = ['%s:summary' % s for s in stages]
            with BarchartPlotter(figname=figname, plotdir=args.plotdir) as plot:
                for key, time in bench.lookup(params=params, event=events,
                                              measure='time').items():
                    param = dict(key)
                    plot.add_value(sum(time), grouplabel='DG-P%d' % param['degree'],
                                   label='Raw' if param['opt'] <= 2 else 'Opt',
                                   color='b' if param['opt'] <= 2 else 'r')
        else:
            raise NotImplementedErorr('Plot type %s not yet implemented' % plottype)
    else:
        RuntimeError("Must specify either 'bench' or 'plot' mode")
