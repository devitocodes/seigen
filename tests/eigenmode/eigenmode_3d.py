#!/usr/bin/env python

from firedrake import *
from elastic_wave.elastic import *
from elastic_wave.helpers import *


class Eigenmode3DLF4():

    def __init__(self, N, degree, dt, solver='explicit', output=True):
        with timed_region('mesh generation'):
            self.mesh = UnitCubeMesh(N, N, N)

        self.elastic = ElasticLF4(self.mesh, "DG", degree, dimension=3,
                                  solver=solver, output=output)

        # Constants
        self.elastic.density = 1.0
        self.elastic.dt = dt
        self.elastic.mu = 0.25
        self.elastic.l = 0.5

        print "P-wave velocity: %f" % Vp(self.elastic.mu, self.elastic.l, self.elastic.density)
        print "S-wave velocity: %f" % Vs(self.elastic.mu, self.elastic.density)

        self.A = sqrt(2*self.elastic.density*self.elastic.mu)
        self.O = pi*sqrt(2*self.elastic.mu/self.elastic.density)

    def eigenmode3d(self, T=5.0):
        # Initial conditions
        uic = Expression(('cos(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*cos(O*t)',
                          'cos(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*cos(O*t)',
                          'cos(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*cos(O*t)'), O=self.O, t=0)
        self.elastic.u0.assign(Function(self.elastic.U).interpolate(uic))
        sic = Expression((('-A*sin(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*sin(O*t)', '0', '0'),
                          ('0', '-A*sin(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*sin(O*t)', '0'),
                          ('0', '0', '-A*sin(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*sin(O*t)')),
                         A=self.A, O=self.O, t=self.elastic.dt/2.0)
        self.elastic.s0.assign(Function(self.elastic.S).interpolate(sic))

        return self.elastic.run(T)

    def eigenmode_error(self, u1, s1):
        uexact_e = Expression(('cos(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*cos(O*t)',
                               'cos(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*cos(O*t)',
                               'cos(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*cos(O*t)'), O=self.O, t=5)
        uexact = Function(self.elastic.U).interpolate(uexact_e)
        sexact_e = Expression((('-A*sin(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*sin(O*t)', '0', '0'),
                               ('0', '-A*sin(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*sin(O*t)', '0'),
                               ('0', '0', '-A*sin(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*sin(O*t)')),
                              A=self.A, O=self.O, t=5+self.elastic.dt/2.0)
        sexact = Function(self.elastic.S).interpolate(sexact_e)

        HU = VectorFunctionSpace(self.mesh, "DG", 3)
        temp = Function(HU)
        temp_test = TestFunction(HU)
        temp_trial = TrialFunction(HU)
        G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(u1-uexact))*dx
        solve(lhs(G) == rhs(G), temp)
        u_error = norm(temp)

        HU = TensorFunctionSpace(self.mesh, "DG", 3)
        temp = Function(HU)
        temp_test = TestFunction(HU)
        temp_trial = TrialFunction(HU)
        G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(s1-sexact))*dx
        solve(lhs(G) == rhs(G), temp)
        s_error = norm(temp)

        return u_error, s_error


def convergence_analysis():
    degrees = range(1, 4)
    N = [2**i for i in range(1, 4)]

    dx = [1.0/n for n in N]

    for d in degrees:
        dt = [0.5*(1.0/n)/(2.0**(d-1)) for n in N]

        f = open("error_p%d_lf4.dat" % d, "w")
        f.write("dx\tdt\tu_error\ts_error\n")
        for i in range(len(N)):
            em = Eigenmode3DLF4(N[i], d, dt[i])
            u1, s1 = em.eigenmode3d()
            u_error, s_error = em.eigenmode_error(u1, s1)
            f.write(str(dx[i]) + "\t" + str(dt[i]) + "\t" + str(u_error) + "\t" + str(s_error) + "\n")
        f.close()

if __name__ == '__main__':
    convergence_analysis()
