from pybench import parser
from eigenmode_bench import EigenmodeBench

class EigenmodePlot(EigenmodeBench):
    figsize = (6, 4)

if __name__ == '__main__':
    p = parser(description="Performance benchmark for 2D eigenmode test.")
    p.add_argument('--dim', type=int, default=3,
                   help='problem dimension to plot')
    p.add_argument('-N', '--size', type=int, nargs='+',
                   help='mesh sizes to plot')
    p.add_argument('-d', '--degree', type=int, nargs='+',
                   help='polynomial degrees to plot')
    args = p.parse_args()
    dim = args.dim
    degrees = args.degree or [1, 2, 3, 4]
    groups = ['explicit', 'O3']
    regions = ['stress solve', 'velocity solve', 'timestepping']
    labels = {(False, False): 'Implicit', (False, True): 'Explicit',
              (True, True): 'Explicit, zero-tracking'}

    b = EigenmodePlot(benchmark='Eigenmode2D-Performance',
                      resultsdir=args.resultsdir, plotdir=args.plotdir)
    b.combine_series([('dim', [dim]), ('size', args.size or [32]), ('dt', [0.125]),
                      ('T', [2.0]), ('explicit', [False, True]), ('O3', [False, True])],
                     filename='EigenmodeLF4')

    degree_str = ['P%s-DG' % d for d in degrees]
    for region in regions:
        b.plot(figsize=b.figsize, format='pdf', figname='Eigen2DLF4_%s'%region,
               xaxis='degree', xvals=degrees, xticklabels=degree_str,
               xlabel='Spatial discretisation', groups=groups, regions=[region],
               kinds='bar', title="", labels=labels, legend={'loc': 'best'})
