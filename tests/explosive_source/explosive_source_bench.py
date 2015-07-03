from explosive_source_lf4 import ExplosiveSourceLF4
from pybench import Benchmark, parser
from pyop2.profiling import get_timers
from firedrake import *

parameters["pyop2_options"]["profiling"] = True
parameters["pyop2_options"]["lazy_evaluation"] = False

class ExplosiveSource(Benchmark, ExplosiveSourceLF4):
    warmups = 0
    repeats = 1

    method = 'explosive_source'
    benchmark = 'ExplosiveSourceLF4'

    def explosive_source(self, T=0.01, h=2.5, explicit=True):
        self.series['h'] = h
        self.series['T'] = T
        self.series['explicit'] = explicit

        self.explosive_source_lf4(T=T, h=h, explicit=explicit, output=False)

        for task, timer in get_timers(reset=True).items():
            self.register_timing(task, timer.total)


if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    ExplosiveSource().main()
