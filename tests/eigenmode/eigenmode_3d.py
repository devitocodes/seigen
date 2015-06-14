#!/usr/bin/env python

from firedrake import *
from elastic_wave.elastic import *
from elastic_wave.helpers import *

def run(N, degree, dt):
   mesh = UnitCubeMesh(N, N, N)
   elastic = ElasticLF4(mesh, "DG", degree, dimension=3)
   
   # Constants
   elastic.density = 1.0
   elastic.dt = dt
   elastic.mu = 0.25
   elastic.l = 0.5

   print "P-wave velocity: %f" % Vp(elastic.mu, elastic.l, elastic.density)
   print "S-wave velocity: %f" % Vs(elastic.mu, elastic.density)

   # Initial conditions
   A = sqrt(2*elastic.density*elastic.mu)
   O = pi*sqrt(2*elastic.mu/elastic.density)
   uic = Expression(('cos(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*cos(O*t)', 'cos(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*cos(O*t)', 'cos(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*cos(O*t)'), O=O, t=0)
   elastic.u0.assign(Function(elastic.U).interpolate(uic))
   sic = Expression((('-A*sin(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*sin(O*t)', '0', '0'),
                     ('0', '-A*sin(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*sin(O*t)', '0'),
                     ('0', '0', '-A*sin(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*sin(O*t)')), A=A, O=O, t=dt/2.0)
   elastic.s0.assign(Function(elastic.S).interpolate(sic))


   uexact = Function(elastic.U).interpolate(Expression(('cos(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*cos(O*t)', 'cos(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*cos(O*t)', 'cos(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*cos(O*t)'), O=O, t=5))
   sexact = Function(elastic.S).interpolate(Expression((('-A*sin(pi*x[0])*(sin(pi*x[1]) - sin(pi*x[2]))*sin(O*t)', '0', '0'),
                     ('0', '-A*sin(pi*x[1])*(sin(pi*x[2]) - sin(pi*x[0]))*sin(O*t)', '0'),
                     ('0', '0', '-A*sin(pi*x[2])*(sin(pi*x[0]) - sin(pi*x[1]))*sin(O*t)')), A=A, O=O, t=5+dt/2.0))

   T = 5.0
   u1, s1 = elastic.run(T)
   
   HU = VectorFunctionSpace(mesh, "DG", 4)
   temp = Function(HU)
   temp_test = TestFunction(HU)
   temp_trial = TrialFunction(HU)
   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(u1-uexact))*dx
   solve(lhs(G) == rhs(G), temp)
   u_error = norm(temp)

   HU = TensorFunctionSpace(mesh, "DG", 4)
   temp = Function(HU)
   temp_test = TestFunction(HU)
   temp_trial = TrialFunction(HU)
   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(s1-sexact))*dx
   solve(lhs(G) == rhs(G), temp)
   s_error = norm(temp)

   return u_error, s_error

def convergence_analysis():
   degrees = range(1, 4)
   N = [2**i for i in range(2, 5)]
   
   dx = [1.0/n for n in N]  
   
   for d in degrees:
      dt = [0.5*(1.0/n)/(2.0**(d-1)) for n in N]
      
      f = open("error_p%d_lf4.dat" % d, "w")
      f.write("dx\tdt\tu_error\ts_error\n")
      for i in range(len(N)):
         u_error, s_error = run(N[i], d, dt[i])
         f.write(str(dx[i]) + "\t" + str(dt[i]) + "\t" + str(u_error) + "\t" + str(s_error) + "\n")
      f.close()
      
   return
   
convergence_analysis()

