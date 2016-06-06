from eigenmode_2d import Eigenmode2DLF4
from eigenmode_3d import Eigenmode3DLF4
from pybench import Benchmark
from firedrake import *
import mpi4py
from firedrake.petsc import PETSc
from os import path, getcwd
import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

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

class EigenmodeBench(Benchmark):
    warmups = 1
    repeats = 3

    method = 'eigenmode'
    benchmark = 'EigenmodeLF4'

    def eigenmode(self, dim=3, N=3, degree=1, dt=0.125, T=2.0,
                  solver='explicit', opt=2):
        self.series['np'] = op2.MPI.comm.size
        self.series['dim'] = dim
        self.series['size'] = N
        self.series['T'] = T
        self.series['solver'] = solver
        self.series['opt'] = opt
        self.series['degree'] = degree
        self.series['dt'] = dt

        # Start PETSc performance logging
        PETSc.Log().begin()

        # If dt is supressed (<0) Infer it based on Courant number
        if dt < 0:
            # Courant number of 0.5: (dx*C)/Vp
            dt = 0.5*(1.0/N)/(2.0**(degree-1))

        parameters["coffee"]["O3"] = opt >= 3
        parameters["coffee"]["O4"] = opt >= 4

        if dim == 2:
            eigen = Eigenmode2DLF4(N, degree, dt, solver=solver, output=False)
            u1, s1 = eigen.eigenmode2d(T=T)
        elif dim == 3:
            eigen = Eigenmode3DLF4(N, degree, dt, solver=solver, output=False)
            u1, s1 = eigen.eigenmode3d(T=T)

        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)

        # Dump PETSc performance log infor to file
        logfile = path.join(self.resultsdir, '%s_petsc.py' % self.name)
        vwr = PETSc.Viewer().createASCII(logfile)
        vwr.pushFormat(PETSc.Viewer().Format().ASCII_INFO_DETAIL)
        PETSc.Log().view(vwr)

        profile = path.join(self.resultsdir, '%s_seigen.json' % self.name)
        with open(profile, 'w') as f:
            json.dump(parameters['seigen']['profiling'], f, indent=4)

        self.meta['dofs'] = op2.MPI.comm.allreduce(eigen.elastic.S.dof_count, op=mpi4py.MPI.SUM)
        if 'u_error' not in self.meta:
            try:
                u_error, s_error = eigen.eigenmode_error(u1, s1)
                self.meta['u_error'] = u_error
                self.meta['s_error'] = s_error
            except RuntimeError:
                print "WARNING: Couldn't establish error norm"
                self.meta['u_error'] = 'NaN'
                self.meta['s_error'] = 'NaN'

from opescibench import Executor

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
                    self.register('%s:%s' % (stage, event), time, measure='time')
                flops = sum([eventlog[p]['flops'] for p in range(petsclog['numProcs'])])
                if flops > 0.0:
                    self.register('%s:%s' % (stage, event), flops, measure='flops')

        # Store meta data
        self.meta['dofs'] = op2.MPI.comm.allreduce(self.eigen.elastic.S.dof_count, op=mpi4py.MPI.SUM)
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
    op2.init(log_level='ERROR')

    # EigenmodeBench(N=4, degree=1, dt=0.125).main()

    from opescibench import Benchmark
    bench = Benchmark(name='EigenmodeBench')
    bench.add_parameter('--dim', type=int, default=2, help='Problem dimension')
    bench.add_parameter('--nprocs', type=int, nargs='+', default=[op2.MPI.comm.size] or [1],
                        help='Number of parallel processes')
    bench.add_parameter('--size', type=int, nargs='+', default=[8],
                        help='Mesh size (number of edges per dimension)')
    bench.add_parameter('--time', type=float, nargs=1, default=5.0,
                        help='Total runtime in seconds')
    bench.add_parameter('--dt', type=float, nargs=1, default=-1.,
                        help='Timestep size in seconds; auto-derived if dt < 0.')
    bench.add_parameter('--degree', type=int, nargs='+', default=[1],
                        help='Degree of spatial discretisation')
    bench.add_parameter('--solver', choices=('implicit', 'explicit'), default='explicit',
                        help='Coffee optimisation level; default -O3')
    bench.add_parameter('--opt', type=int, nargs='+', default=[3],
                        help='Coffee optimisation level; default -O3')
    # Additional arguments for plotting
    bench.plotter.parser.add_argument('--kernel', type=str, nargs='+', default=['uh1'],
                                      help='Name of kernels for roofline plots')
    bench.parse_args()

    if bench.args.mode == 'bench':
        # Run the model across the parameter sweep and save the result
        bench.execute(EigenmodeExecutor(), warmups=0, repeats=1)
        bench.save()
    elif bench.args.mode == 'plot':
        bench.load()
        if bench.args.plottype == 'strong':
            for field in ['velocity', 'stress']:
                figname = 'SeigenStrong_%s_%s.pdf' % (field, bench.args.solver)
                time = bench.lookup(event='%s solve:summary' % field, measure='time')
                events = ['%s:summary' % s for s in petsc_stages['%s solve' % field]]
                if bench.args.solver != 'implicit':
                    for ev in events:
                        time += bench.lookup(event=ev, measure='time')
                bench.plotter.plot_strong_scaling(figname, bench.args.nprocs, time)
                figname = 'SeigenEfficiency_%s_%s.pdf' % (field, bench.args.solver)
                bench.plotter.plot_efficiency(figname, bench.args.nprocs, time)

        elif bench.args.plottype == 'error':
            for field in ['velocity', 'stress']:
                figname = 'SeigenError_%s.pdf' % field
                time = {}
                error = {}
                events = ['%s:summary' % s for s in petsc_stages['%s solve' % field]]
                for deg in bench.args.degree:
                    label = u'P%d$_{DG}$' % deg
                    # Annoyingly, nested stage:summary data is not accumulative,
                    # so we need to sum across the relevant forms ourselves
                    time[label] = bench.lookup(event='%s solve:summary' % field,
                                               measure='time', params={'degree': deg})
                    if bench.args.solver != 'implicit':
                        for ev in events:
                            time[label] += bench.lookup(event=ev, measure='time',
                                                        params={'degree': deg})
                    error[label] = bench.lookup(event=None, measure='%s_error' % field,
                                                params={'degree': deg}, category='meta')
                bench.plotter.plot_error_cost(figname, error, time)

        elif bench.args.plottype == 'roofline':
            for kernel in bench.args.kernel:
                for pltype, plname in ploop_types.items():
                    figname = 'SeigenRoofline_%s_%s.pdf' % (kernel, pltype)
                    flops_s = {}
                    op_int = {}
                    for deg, opt in product(bench.args.degree, bench.args.opt):
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
                        bench.plotter.plot_roofline(figname, flops_s, op_int)
        else:
            raise NotImplementedErorr('Plot type %s not yet implemented' % plottype)
    else:
        RuntimeError("Must specify either 'bench' or 'plot' mode")
