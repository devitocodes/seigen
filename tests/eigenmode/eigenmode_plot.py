from eigenmode_bench import EigenmodeBench
from pybench import parser
from itertools import product
from collections import defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from os import path
from os import environ as env
import numpy as np
import json


rcParams.update({'font.size': 10})


def bandwidth_from_petsc_stream(logfile):
    import re
    regex = re.compile("Triad:\s+([0-9,\.,\,]+)")
    max_bw = 0.
    with open(logfile) as f:
        for line in f:
            match = regex.search(line)
            if match:
                max_bw = max(max_bw, float(match.group(1)))
    return max_bw


class EigenmodePlot(EigenmodeBench):
    figsize = (8, 4)
    marker = ['D', 'o', '^', 'v']

    def plot_strong_scaling(self, nprocs, regions):
        groups = ['solver', 'opt']
        xlabel = 'Number of processors'

        # Plot classic strong scaling plot
        self.plot(figsize=b.figsize, format='pdf', figname='SeigenStrong',
                  xaxis='np', xlabel=xlabel, xticklabels=nprocs,
                  groups=groups, regions=regions, kinds='loglog', axis='tight',
                  title='', labels=labels, legend={'loc': 'best'})

        # Plot parallel efficiency
        efficiency = lambda xvals, yvals: [xvals[0]*yvals[0]/(x*y)
                                           for x, y in zip(xvals, yvals)]
        self.plot(figsize=b.figsize, format='pdf', figname='SeigenStrongEfficiency',
                  ylabel='Parallel efficiency w.r.t. %d cores' % nprocs[0],
                  xaxis='np', xlabel=xlabel, xticklabels=args.parallel,
                  groups=groups, regions=regions, kinds='semilogx', axis='tight',
                  title='', labels=labels, transform=efficiency, ymin=0)

    def plot_comparison(self, degrees, regions):
        groups = ['solver', 'opt']
        degree_str = ['P%s-DG' % d for d in degrees]
        # Bar comparison between solver modes and coffee parameters
        for region in regions:
            self.plot(figsize=b.figsize, format='pdf', figname='SeigenCompare_%s' % region,
                      xaxis='degree', xvals=degrees, xticklabels=degree_str,
                      xlabel='Spatial discretisation', groups=groups, regions=[region],
                      kinds='bar', title='Performance: %s' % region, labels=labels,
                      legend={'loc': 'best'})

    def plot_error_cost(self, results, fieldname, errname):
        # Plot error-cost diagram for velocity by hand
        figname = 'SeigenError_%s.pdf' % fieldname
        fig = plt.figure(figname, figsize=self.figsize, dpi=300)
        ax = fig.add_subplot(111)

        for deg, res in results.items():
            sizes, rs = zip(*sorted(res.items(), key=itemgetter(0)))
            time = [r['timings'].values()[0]['%s solve' % fieldname] for r in rs]
            error = [r['meta'][errname] for r in rs]
            spacing = [1. / N for N in sizes]
            if len(time) > 0:
                ax.loglog(error, time, label='P%d-DG' % deg, linewidth=2,
                          linestyle='solid', marker=self.marker[deg-1])
                for i, (x, y, dx) in enumerate(zip(error, time, spacing)):
                    if i == 2:
                        xy_off = (-8, 8)
                    if i == 1:
                        xy_off = (1, 5) if deg <= 2 else (-44, -5) if deg == 4 else (-3, 5)
                    if i == 0:
                        xy_off = (-20, -14) if deg == 1 or deg == 3 else \
                                 (3, 3) if deg == 2 else (-44, -5)
                    plt.annotate("dx=%4.3f" % dx, xy=(x, y), xytext=xy_off,
                                 textcoords='offset points', size=8)

        # Manually add legend and axis labels
        ax.legend(loc='best', ncol=4, fancybox=True, fontsize=10)
        ax.set_xlabel('%s error in L2 norm' % fieldname.capitalize())
        yvals = 2 ** np.linspace(1, 5, 5, dtype=np.int32)
        ax.set_ylim(yvals[0], yvals[-1])
        ax.set_yticks(yvals)
        ax.set_yticklabels(yvals)
        ax.set_ylabel('Wall time / seconds')
        fig.savefig(path.join(self.plotdir, figname), facecolor='white',
                    orientation='landscape', format='pdf',
                    transparent=True, bbox_inches='tight')

    def plot_roofline_kernel(self, max_perf, max_bw, series, stage, kernel, label=''):
        # Roofline for individual kernels
        figname = 'SeigenRoofline-%s.pdf' % label
        fig = plt.figure(figname, figsize=self.figsize, dpi=300)
        ax = fig.add_subplot(111)

        ai_x = 2 ** np.linspace(-1, 6, 8)
        perf = ai_x * max_bw
        perf[perf > max_perf] = max_perf
        # Insert the crossover point between BW and flops limit
        idx = (perf >= max_perf).argmax()
        ai = np.insert(ai_x, idx, max_perf / max_bw)
        perf = np.insert(perf, idx, max_perf)
        ax.loglog(ai, perf, 'k-')

        data_ai = {}
        data_flops = {}
        skeys, svals = zip(*sorted(series))
        for svalues in product(*svals):
            suff = '_'.join('%s%s' % (k, v) for k, v in zip(skeys, svalues))
            fpath = path.join(self.resultsdir, '%s_%s' % (self.name, suff))
            param = dict(zip(skeys, svalues))

            # Get Flops measurements from petsc log files
            petsc_res = {}
            execfile('%s_petsc.py' % fpath, globals(), petsc_res)
            flops = [petsc_res['Stages'][stage][kernel][p]['flops']
                     for p in range(petsc_res['numProcs'])]
            time = [petsc_res['Stages'][stage][kernel][p]['time']
                    for p in range(petsc_res['numProcs'])]
            flops_s = sum(flops) / max(time) / 1.e6  # 1024 ** 2
            data_flops[(param['degree'], param['opt'])] = flops_s

            # Get arithmetic intensity from parloop calculations
            with open('%s_seigen.json' % fpath, 'r') as f:
                arith = json.loads(f.read())[stage][kernel]['ai']
                data_ai[param['degree']] = arith

        for deg, arith in data_ai.items():
            ax.plot([arith, arith], [1000, min(arith*max_bw, max_perf)], 'k:')
            plt.annotate("DG-%d" % deg, xy=(arith, 1.e4), xytext=(3, 3),
                         textcoords='offset points', size=12, rotation=-90)

        for (deg, opt), flops in data_flops.items():
            ax.plot(data_ai[deg], flops, color='k', marker=self.marker[opt-2],
                    label='Coffee-O%d' % opt if deg == 2 else None)

        ax.set_xlim(ai_x[0], ai_x[-1])
        ax.set_xticks(ai_x)
        ax.set_xticklabels(ai_x)
        ax.set_xlabel('Arithmetic intensity (Flops/Byte)')
        yvals = 2 ** np.linspace(2, 10, 9, dtype=np.int32) * 1000
        ax.set_ylim(yvals[0], yvals[-1])
        ax.set_yticks(yvals)
        ax.set_yticklabels(yvals / 1000)
        ax.set_ylabel('Performance (GFlops/s)')
        ax.legend(loc='best', fancybox=True, fontsize=10, title='Optimisation level')
        fig.savefig(path.join(self.plotdir, figname),
                    orientation='landscape', format='pdf',
                    transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    p = parser(description="Performance benchmark for 2D eigenmode test.")
    p.add_argument('mode', choices=('strong', 'error', 'comparison', 'roofline'),
                   nargs='?', default='scaling',
                   help='determines type of plot to generate')
    p.add_argument('--dim', type=int, default=3,
                   help='problem dimension to plot')
    p.add_argument('-N', '--size', type=int, nargs='+', default=[32],
                   help='mesh sizes to plot')
    p.add_argument('-d', '--degree', type=int, nargs='+', default=[4],
                   help='polynomial degrees to plot')
    p.add_argument('-T', '--time', type=float, nargs='+', default=[2.0],
                   help='total simulated time')
    p.add_argument('--dt', type=float, nargs='+', default=[0.125],
                   help='total simulated time')
    p.add_argument('--solver', nargs='+', default=['explicit'],
                   help='Solver method used ("implicit", "explicit")')
    p.add_argument('--opt', type=int, nargs='+', default=[3],
                   help='Coffee optimisation levels used')
    p.add_argument('--stream', default=None,
                   help='Stream log file from PETSc')
    p.add_argument('--max-perf', type=float, default=1900 * 4,
                   help='Stream log file from PETSc')
    args = p.parse_args()
    dim = args.dim
    degrees = args.degree or [1, 2, 3, 4]
    nprocs = args.parallel or [1]
    regions = ['stress solve', 'velocity solve', 'timestepping']
    labels = {(2, 'implicit'): 'Implicit',
              (2, 'explicit'): 'Explicit',
              (3, 'explicit'): 'Explicit, zero-tracking',
              (4, 'explicit'): 'Explicit, coffee-O4'}

    b = EigenmodePlot(benchmark='EigenmodeLF4', resultsdir=args.resultsdir,
                      plotdir=args.plotdir)
    if args.mode == 'strong':
        b.combine_series([('np', nprocs), ('dim', [args.dim]), ('size', args.size),
                          ('degree', args.degree), ('dt', args.dt), ('T', args.time),
                          ('solver', args.solver), ('opt', args.opt)])
        b.plot_strong_scaling(nprocs=nprocs, regions=regions)

    elif args.mode == 'comparison':
        b.combine_series([('np', nprocs), ('dim', [args.dim]), ('size', args.size),
                          ('degree', args.degree), ('dt', args.dt), ('T', args.time),
                          ('solver', args.solver), ('opt', args.opt)])
        b.plot_comparison(degrees=degrees, regions=regions)

    elif args.mode == 'error':
        # Need to read individual series files because meta data that
        # holds our error norms get overwritten in combine_series().
        results = defaultdict(dict)
        for d, N in product(degrees, args.size):
            r = EigenmodePlot(benchmark='EigenmodeLF4', resultsdir=args.resultsdir,
                              plotdir=args.plotdir)
            r.combine_series([('np', nprocs), ('dim', [args.dim]), ('degree', [d]),
                              ('size', [N]), ('dt', args.dt), ('T', args.time),
                              ('solver', args.solver), ('opt', args.opt)])
            if len(r.result['timings']) > 0:
                results[d][N] = r.result

        # Plot custom error-cost plots from the result set
        b.plot_error_cost(results, 'velocity', 'u_error')
        b.plot_error_cost(results, 'stress', 's_error')

    elif args.mode == 'roofline':
        if args.stream:
            stream_log = args.stream
        else:
            stream_log = path.join(env['PETSC_DIR'], 'src/benchmarks/streams/scaling.log')
        max_bw = bandwidth_from_petsc_stream(stream_log)

        series = [('np', nprocs), ('dim', [args.dim]), ('size', args.size),
                  ('degree', args.degree), ('dt', args.dt), ('T', args.time),
                  ('solver', args.solver), ('opt', args.opt)]

        # Max BW in MB/s; max perf in MFlops/s
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='sh1',
                               kernel='form0_cell_integral_otherwise', label='sh1-cell')
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='sh1',
                               kernel='form0_interior_facet_integral_otherwise', label='sh1-interior')
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='sh1',
                               kernel='form0_exterior_facet_integral_otherwise', label='sh1-exterior')
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='sh2',
                               kernel='form0_cell_integral_otherwise', label='sh2-cell')
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='sh2',
                               kernel='form0_interior_facet_integral_otherwise', label='sh2-interior')
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='sh2',
                               kernel='form0_exterior_facet_integral_otherwise', label='sh2-exterior')
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='uh1',
                               kernel='form0_cell_integral_otherwise', label='uh1-cell')
        b.plot_roofline_kernel(args.max_perf, max_bw, series, stage='uh1',
                               kernel='form0_interior_facet_integral_otherwise', label='uh1-interior')
