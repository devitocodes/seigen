from eigenmode_2d import Eigenmode2DLF4
from eigenmode_3d import Eigenmode3DLF4
from firedrake import *
from pyop2.mpi import MPI
from firedrake.petsc import PETSc
from os import path, getcwd
from itertools import product
from collections import OrderedDict
from opescibench import Executor, Plotter
from argparse import ArgumentParser, RawDescriptionHelpFormatter

parameters["pyop2_options"]["profiling"] = True
parameters["coffee"]["O2"] = False
parameters["seigen"] = {}
parameters["seigen"]["profiling"] = True

petsc_stages = {'velocity solve': ['uh1', 'stemp', 'uh2', 'u1'],
                'stress solve': ['sh1', 'utemp', 'sh2', 's1']}
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
        self.meta['dofs'] = MPI.COMM_WORLD.allreduce(self.eigen.elastic.S.dof_count, op=MPI.SUM)
        self.meta['spacing'] = 1. / size
        self.meta['kernel_profile'] = parameters['seigen']['profiling']

    def postprocess(self, **kwargs):
        """ Error estimates are compute intensive, so only run them once. """
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
    simulation.add_argument('--solver', choices=('implicit', 'explicit'), nargs='+', default='explicit',
                            help='Coffee optimisation level; default -O3')
    simulation.add_argument('--opt', type=int, nargs='+', default=[3],
                            help='Coffee optimisation level; default -O3')

    # Additional arguments for plotting
    plotting = parser.add_argument_group("Plotting")
    plotting.add_argument('--plottype', default='error',
                          choices=('error', 'strong', 'roofline'),
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

    bench = Benchmark(parameters=params, name='EigenmodeBench')

    if args.mode == 'bench':
        # Run the model across the parameter sweep and save the result
        bench.execute(EigenmodeExecutor(), warmups=0, repeats=1)
        bench.save()
    elif args.mode == 'plot':
        plotter = Plotter(plotdir=args.plotdir)
        bench.load()
        if args.plottype == 'strong':
            for field, solver in product(['velocity', 'stress'], args.solver):
                figname = 'SeigenStrong_%s_%s.pdf' % (field, solver)
                time = bench.lookup(event='%s solve:summary' % field, measure='time')
                events = ['%s:summary' % s for s in petsc_stages['%s solve' % field]]
                if solver != 'implicit':
                    for ev in events:
                        time += bench.lookup(event=ev, measure='time')
                plotter.plot_strong_scaling(figname, args.nprocs, time)
                figname = 'SeigenEfficiency_%s_%s.pdf' % (field, solver)
                plotter.plot_efficiency(figname, args.nprocs, time)

        elif args.plottype == 'error':
            plotter.figsize = (8.1, 5.2)
            for field in ['velocity', 'stress']:
                figname = 'SeigenError_%s.pdf' % field
                time = OrderedDict()
                error = OrderedDict()
                styles = OrderedDict()
                events = ['%s:summary' % s for s in petsc_stages['%s solve' % field]]
                for deg, solver in product(args.degree, args.solver):
                    label = u'P%d$_{DG}$ %s' % (deg, solver)
                    params = {'degree': deg, 'solver': solver}
                    styles[label] = '%s%s%s' % ('-' if solver == 'explicit' else ':',
                                                plotter.marker[deg-1],
                                                plotter.colour[deg-1])
                    # Annoyingly, nested stage:summary data is not accumulative,
                    # so we need to sum across the relevant forms ourselves
                    time[label] = bench.lookup(event='%s solve:summary' % field,
                                               measure='time', params=params)
                    if solver != 'implicit':
                        for ev in events:
                            time[label] += bench.lookup(event=ev, measure='time',
                                                        params=params)
                    error[label] = bench.lookup(event=None, measure='%s_error' % field,
                                                params=params, category='meta')
                plotter.plot_error_cost(figname, error, time, styles,
                                        xlabel='%s error in L2 norm' % field.capitalize())

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
        else:
            raise NotImplementedErorr('Plot type %s not yet implemented' % plottype)
    else:
        RuntimeError("Must specify either 'bench' or 'plot' mode")
