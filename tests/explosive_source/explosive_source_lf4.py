#!/usr/bin/env python

from firedrake import *
from elastic_wave.elastic import *
from elastic_wave.helpers import *


class ExplosiveSourceLF4():

   def generate_mesh(self, Lx=300.0, Ly=150.0, h=2.5):
      return RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)

   def explosive_source_lf4(self, T=2.5, Lx=300.0, Ly=150.0, h=2.5,
                            explicit=False, output=True):

      with timed_region('mesh generation'):
         mesh = self.generate_mesh()
         self.elastic = ElasticLF4(mesh, "DG", 2, dimension=2,
                                   explicit=explicit, output=output)

      # Constants
      D = FunctionSpace(mesh, "DG", 2)
      self.elastic.density = 1.0
      self.elastic.mu = Function(D).interpolate(Expression("x[1] >= 140 ? 3600.0 : 4500.0"))
      self.elastic.l = Function(D).interpolate(Expression("x[1] >= 140 ? 3599.3664 : 4499.208"))
      
      self.Vp = Vp(self.elastic.mu, self.elastic.l, self.elastic.density)
      self.Vs = Vs(self.elastic.mu, self.elastic.density)
      print "P-wave velocity: %f" % self.Vp
      print "S-wave velocity: %f" % self.Vs
      
      self.dx = h
      self.courant_number = 0.5
      self.elastic.dt = cfl_dt(self.dx, self.Vp, self.courant_number)
      print "Using a timestep of %f" % self.elastic.dt # This was previously hard-coded to be 0.001 s.

      # Source
      a = 159.42
      self.elastic.source_expression = Expression((("x[0] >= 44.5 && x[0] <= 45.5 && x[1] >= 148.5 && x[1] <= 149.5 ? (-1.0 + 2*a*pow(t - 0.3, 2))*exp(-a*pow(t - 0.3, 2)) : 0.0", "0.0"),
                                                   ("0.0", "x[0] >= 44.5 && x[0] <= 45.5 && x[1] >= 148.5 && x[1] <= 149.5 ? (-1.0 + 2*a*pow(t - 0.3, 2))*exp(-a*pow(t - 0.3, 2)) : 0.0")), a=a, t=0)
      self.elastic.source_function = Function(self.elastic.S)
      self.elastic.source = self.elastic.source_expression

      # Absorption
      F = FunctionSpace(mesh, "DG", 4)
      self.elastic.absorption_function = Function(F)
      self.elastic.absorption = Expression("x[0] <= 20 || x[0] >= 280 || x[1] <= 20.0 ? 1000 : 0")

      # Initial conditions
      uic = Expression(('0.0', '0.0'))
      self.elastic.u0.assign(Function(self.elastic.U).interpolate(uic))
      sic = Expression((('0','0'),
                  ('0','0')))
      self.elastic.s0.assign(Function(self.elastic.S).interpolate(sic))

      # Start the simulation
      with timed_region('elastic-run'):
         self.elastic.run(T)
         self.elastic.run(T, reverse_time=True)


if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    ExplosiveSourceLF4().explosive_source_lf4(T=1.0)
