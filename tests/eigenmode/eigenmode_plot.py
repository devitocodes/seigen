from eigenmode_bench import EigenmodeBench
from pybench import parser

class EigenmodePlot(EigenmodeBench):
    figsize = (6, 4)

    def plot_strong_scaling(self, nprocs, dim, size, degrees, dt, time, explicit, opt):
        b.combine_series([('np', nprocs), ('dim', dim), ('size', size),
                          ('degree', degrees), ('dt', dt), ('T', time),
                          ('explicit', args.explicit), ('opt', args.opt)],
                         filename='EigenmodeLF4')

        groups = ['explicit', 'opt']
        xlabel = 'Number of processors'
        b.plot(figsize=b.figsize, format='pdf', figname='SeigenStrong',
               xaxis='np', xlabel=xlabel, xticklabels=args.parallel,
               groups=groups, regions=regions, kinds='loglog', axis='tight',
               title='', labels=labels, legend={'loc': 'best'})


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
    p.add_argument('--explicit', nargs='+', default=[True],
                   help='Explicit solver method used (True or False)')
    p.add_argument('--opt', type=int, nargs='+', default=[3],
                   help='Coffee optimisation levels used')
    args = p.parse_args()
    dim = args.dim
    degrees = args.degree or [1, 2, 3, 4]
    nprocs = args.parallel or [1]
    regions = ['stress solve', 'velocity solve', 'timestepping']
    labels = {(2, False): 'Implicit',
              (2, True): 'Explicit',
              (3, True): 'Explicit, zero-tracking',
              (4, True): 'Explicit, coffee-O4'}

    b = EigenmodePlot(benchmark='Eigenmode2D-Performance',
                      resultsdir=args.resultsdir, plotdir=args.plotdir)

    if args.mode == 'strong':
        b.plot_strong_scaling(nprocs=nprocs, dim=[args.dim], size=args.size,
                              degrees=degrees, dt=[0.125], time=args.time,
                              explicit=args.explicit, opt=args.opt)

    if args.mode == 'error':
        from itertools import product
        import numpy as np
        import matplotlib.pyplot as plt

        ## Hack to do an error-cost plot
        size = args.size or [32]
        keys = []
        stime = []
        utime = []
        u_err = []
        s_err = []
        us_dx = []
        marker = ['D', 'o', '^', 'v']
        for d, N in product(degrees, size):
            dt = [0.5*(1.0/N)/(2.0**(d-1))]
            r = EigenmodePlot(benchmark='Eigenmode2D-Performance',
                              resultsdir=args.resultsdir, plotdir=args.plotdir)
            r.combine_series([('dim', [dim]), ('degree', [d]), ('size', [N]), ('dt', dt),
                              ('T', [5.0]), ('explicit', [True]), ('opt', [2, 3, 4])],
                             filename='EigenmodeLF4')
            if len(r.result['timings']) > 0:
                # Record err and cost for each d-N combination we can find
                keys.append((d, N))
                us_dx.append(1. / N)
                timings = r.result['timings'].values()
                s_err.append(r.result['meta']['s_error'])
                u_err.append(r.result['meta']['u_error'])
                stime.append(timings[0]['stress solve'])
                utime.append(timings[0]['velocity solve'])

        # Plot error-cost diagram by hand
        figname = 'Eigen2DLF4_error_velocity.pdf'
        fig = plt.figure(figname, figsize=b.figsize, dpi=300)
        ax = fig.add_subplot(111)
        keys = np.array(keys)
        us_dx = np.array(us_dx)
        utime = np.array(utime)
        u_err = np.array(u_err)

        for d in degrees:
            d_keys = (keys[:,0] == d)
            d_utime = utime[d_keys]
            d_u_err = u_err[d_keys]
            d_dx = us_dx[d_keys]
            if len(d_utime) > 0:
                ax.loglog(d_u_err, d_utime, label='P%d-DG'%d, linewidth=2,
                          linestyle='solid', marker=marker[d-1])
                for x, y, dx in zip(d_u_err, d_utime, d_dx):
                    xy_off = (3, 3) if d < 4 else (-40, -6)
                    plt.annotate("dx=%4.3f"%dx, xy=(x, y), xytext=xy_off,
                                 textcoords='offset points', size=8)

        # Manually add legend and axis labels
        l = ax.legend(loc='best', ncol=2, fancybox=True, prop={'size':12})
        ax.set_xlabel('Velocity error in L2 norm')
        ax.set_ylabel('Wall time / seconds')

        from os import path
        fig.savefig(path.join(b.plotdir, figname),
                    orientation='landscape', format='pdf',
                    transparent=True, bbox_inches='tight')

        # Plot error-cost diagram by hand
        figname = 'Eigen2DLF4_error_stress.pdf'
        fig = plt.figure(figname, figsize=b.figsize, dpi=300)
        ax = fig.add_subplot(111)
        stime = np.array(stime)
        s_err = np.array(s_err)
        for d in degrees:
            d_keys = (keys[:,0] == d)
            d_stime = stime[d_keys]
            d_s_err = s_err[d_keys]
            d_dx = us_dx[d_keys]
            if len(d_stime) > 0:
                ax.loglog(d_s_err, d_stime, label='P%d-DG'%d, linewidth=2,
                          linestyle='solid', marker=marker[d-1])
                for x, y, dx in zip(d_s_err, d_stime, d_dx):
                    xy_off = (3, 3) if d < 4 else (-40, -6)
                    plt.annotate("dx=%4.3f"%dx, xy=(x, y), xytext=xy_off,
                                 textcoords='offset points', size=8)

        # Manually add legend and axis labels
        l = ax.legend(loc='best', ncol=2, fancybox=True, prop={'size':12})
        ax.set_xlabel('Stress error in L2 norm')
        ax.set_ylabel('Wall time / seconds')

        from os import path
        fig.savefig(path.join(b.plotdir, figname),
                    orientation='landscape', format='pdf',
                    transparent=True, bbox_inches='tight')

    elif args.mode == 'comparison':
        # Bar comparison between explicit/implicit and coffe -O3 parameters
        groups = ['explicit', 'opt']
        b.combine_series([('np', nprocs), ('dim', [dim]), ('size', args.size or [32]),
                          ('degree', degrees), ('dt', [0.125]), ('T', args.time or [2.0]),
                          ('explicit', [False, True]), ('opt', [2, 3, 4])],
                         filename='EigenmodeLF4')

        degree_str = ['P%s-DG' % d for d in degrees]
        for region in regions:
            b.plot(figsize=b.figsize, format='pdf', figname='Eigen2DLF4_%s'%region,
                   xaxis='degree', xvals=degrees, xticklabels=degree_str,
                   xlabel='Spatial discretisation', groups=groups, regions=[region],
                   kinds='bar', title='Performance: %s'%region, labels=labels, legend={'loc': 'best'})
