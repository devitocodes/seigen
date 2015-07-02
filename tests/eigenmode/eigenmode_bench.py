from eigenmode_2d import Eigenmode2DLF4
from pybench import Benchmark, parser
from pyop2.profiling import get_timers
from firedrake import *

parameters["pyop2_options"]["profiling"] = True
parameters["pyop2_options"]["lazy_evaluation"] = False

class EigenmodeBench(Benchmark):
    warmups = 1
    repeats = 3

    method = 'eigenmode2d'
    benchmark = 'Eigenmode2DLF4'
    params = [('degree', range(1, 5))]

    def eigenmode2d(self, N=4, degree=1, dt=0.125, T=2.0, explicit=True):
        self.series['size'] = N
        self.series['dt'] = dt
        self.series['T'] = T
        self.series['explicit'] = explicit

        eigen = Eigenmode2DLF4(N, degree, dt, explicit=explicit)
        eigen.eigenmode2d(T=T)

        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)


if __name__ == '__main__':
    op2.init(log_level='ERROR')
    from ffc.log import set_level
    set_level('ERROR')

    EigenmodeBench(N=4, degree=1, dt=0.125).main()
