from eigenmode_2d import Eigenmode2DLF4
from pybench import Benchmark, parser
from pyop2.profiling import get_timers
from firedrake import op2

class EigenmodeBench(Benchmark):
    warmups = 0
    repeats = 1

    method = 'eigenmode2d'
    benchmark = 'Eigenmode2DLF4'

    def eigenmode2d(self, N=4, degree=1, dt=0.125, T=2.0):
        eigen = Eigenmode2DLF4(N, degree, dt)
        eigen.eigenmode2d(T=T)

        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)


if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    EigenmodeBench(N=4, degree=1, dt=0.125).main()
