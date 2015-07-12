#!/usr/bin/env python

from firedrake import *
from elastic_wave.elastic_ader import *
from elastic_wave.helpers import *

class Eigenmode2DLF4():

   def __init__(self, N, degree, dt, explicit=True, output=True):
      with timed_region('mesh generation'):
         self.mesh = UnitSquareMesh(N, N)

      self.elastic = ElasticLF4(self.mesh, "DG", degree, dimension=2,
                                explicit=explicit, output=output)

      # Constants
      self.elastic.density = 1.0
      self.elastic.dt = dt
      self.elastic.mu = 0.25
      self.elastic.l = 0.5

      print "P-wave velocity: %f" % Vp(self.elastic.mu, self.elastic.l, self.elastic.density)
      print "S-wave velocity: %f" % Vs(self.elastic.mu, self.elastic.density)

      self.a = sqrt(2)*pi*Vs(self.elastic.mu, self.elastic.density)
      self.b = 2*pi*self.elastic.mu

   def eigenmode2d(self, T=5.0):
      # Initial conditions
      uic = Expression(('a*cos(pi*x[0])*sin(pi*x[1])*cos(a*t)',
                        '-a*sin(pi*x[0])*cos(pi*x[1])*cos(a*t)'), a=self.a, t=0)
      self.elastic.u0.assign(Function(self.elastic.U).interpolate(uic))
      sic = Expression((('-b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)','0'),
                        ('0','b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)')),
                       a=self.a, b=self.b, t=self.elastic.dt/2.0)
      self.elastic.s0.assign(Function(self.elastic.S).interpolate(sic))

      return self.elastic.run(T)

   def eigenmode_error(self, u1, s1):
      uexact_e = Expression(('a*cos(pi*x[0])*sin(pi*x[1])*cos(a*t)',
                             '-a*sin(pi*x[0])*cos(pi*x[1])*cos(a*t)'), a=self.a, t=5)
      uexact = Function(self.elastic.U).interpolate(uexact_e)
      sexact_e = Expression((('-b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)','0'),
                             ('0','b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)')),
                            a=self.a, b=self.b, t=5+self.elastic.dt/2.0)
      sexact = Function(self.elastic.S).interpolate(sexact_e)

      HU = VectorFunctionSpace(self.mesh, "DG", 6)
      temp = Function(HU)
      temp_test = TestFunction(HU)
      temp_trial = TrialFunction(HU)
      G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(u1-uexact))*dx
      solve(lhs(G) == rhs(G), temp)
      u_error = norm(temp)

      HU = TensorFunctionSpace(self.mesh, "DG", 6)
      temp = Function(HU)
      temp_test = TestFunction(HU)
      temp_trial = TrialFunction(HU)
      G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(s1-sexact))*dx
      solve(lhs(G) == rhs(G), temp)
      s_error = norm(temp)

      return u_error, s_error

def convergence_analysis():
   degrees = range(1, 5)
   N = [2**i for i in range(2, 6)]
   
   dx = [1.0/n for n in N]  
   
   for d in degrees:
      dt = [0.5*(1.0/n)/(2.0**(d-1)) for n in N] # Courant number of 0.5: (dx*C)/Vp
      
      f = open("error_p%d_lf4.dat" % d, "w")
      f.write("dx\tdt\tu_error\ts_error\n")
      for i in range(len(N)):
         em = Eigenmode2DLF4(N[i], d, dt[i])
         u1, s1 = em.eigenmode2d()
         u_error, s_error = em.eigenmode_error(u1, s1)
         f.write(str(dx[i]) + "\t" + str(dt[i]) + "\t" + str(u_error) + "\t" + str(s_error) + "\n")
      f.close()

if __name__ == '__main__':
   convergence_analysis()
