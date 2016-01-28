from eigenmode_bench import EigenmodeBench
from pybench import parser
from itertools import product
from collections import defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt
from os import path


class EigenmodePlot(EigenmodeBench):
    figsize = (6, 4)

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
        marker = ['D', 'o', '^', 'v']
        figname = 'SeigenError_%s.pdf' % fieldname
        fig = plt.figure(figname, figsize=b.figsize, dpi=300)
        ax = fig.add_subplot(111)

        for deg, res in results.items():
            sizes, rs = zip(*sorted(res.items(), key=itemgetter(0)))
            time = [r['timings'].values()[0]['%s solve' % fieldname] for r in rs]
            error = [r['meta'][errname] for r in rs]
            spacing = [1. / N for N in sizes]
            if len(time) > 0:
                ax.loglog(error, time, label='P%d-DG' % deg, linewidth=2,
                          linestyle='solid', marker=marker[d-1])
                for x, y, dx in zip(error, time, spacing):
                    xy_off = (3, 3) if d < 4 else (-40, -6)
                    plt.annotate("dx=%4.3f" % dx, xy=(x, y), xytext=xy_off,
                                 textcoords='offset points', size=8)

        # Manually add legend and axis labels
        ax.legend(loc='best', ncol=2, fancybox=True, prop={'size': 12})
        ax.set_xlabel('%s error in L2 norm' % fieldname.capitalize())
        ax.set_ylabel('Wall time / seconds')
        fig.savefig(path.join(b.plotdir, figname),
                    orientation='landscape', format='pdf',
                    transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    p = parser(description="Performance benchmark for 2D eigenmode test.")
    p.add_argument('mode', choices=('strong', 'error', 'comparison'),
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
