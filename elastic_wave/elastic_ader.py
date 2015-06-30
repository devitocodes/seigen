#!/usr/bin/env python

from pyop2 import *
from pyop2.profiling import timed_region, summary
op2.init(lazy_evaluation=False)
from firedrake import *
parameters["coffee"]["O2"] = False # FIXME: Remove this one this issue has been fixed: https://github.com/firedrakeproject/firedrake/issues/425
parameters["assembly_cache"]["enabled"] = False
import mpi4py
import numpy
from math import factorial

# By Nas Banov, taken from http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction
def binomial(n,k): 
  return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )
  

class ElasticADER(object):
   """ Elastic wave equation solver using the finite element method and the ADER time-stepping scheme. """

   def __init__(self, mesh, family, degree, dimension):
      with timed_region('function setup'):
         self.mesh = mesh
         self.dimension = dimension

         self.S = TensorFunctionSpace(mesh, family, degree)
         self.U = VectorFunctionSpace(mesh, family, degree)
         # Assumes that the S and U function spaces are the same.
         print "Number of degrees of freedom: %d" % op2.MPI.comm.allreduce(self.S.dof_count, op=mpi4py.MPI.SUM)

         self.s = TrialFunction(self.S)
         self.v = TestFunction(self.S)
         self.u = TrialFunction(self.U)
         self.w = TestFunction(self.U)

         self.s0 = Function(self.S, name="StressOld")
         self.s1 = Function(self.S, name="StressNew")

         self.u0 = Function(self.U, name="VelocityOld")
         self.u1 = Function(self.U, name="VelocityNew")
         
         self.absorption_function = None
         self.source_function = None
         self.source_expression = None
         self._dt = None
         self._density = None
         self._mu = None
         self._l = None
         
         self.n = FacetNormal(self.mesh)
         self.I = Identity(self.dimension)

      with timed_region('i/o'):
         # File output streams
         self.u_stream = File("velocity.pvd")
         self.s_stream = File("stress.pvd")

   def Ax(self, k):
      Ax = [[0, 0, 0, (self.l + 2*self.mu), 0],
            [0, 0, 0, self.mu, 0],
            [0, 0, 0, 0, self.mu],
            [1.0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0]]
      if k == 0:
         # Identity matrix
         Ax = [[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]
         Ax = numpy.array(Ax)
      else:
         Ax = numpy.array(Ax)
         for i in range(1, k):
            Ax = numpy.dot(Ax, Ax)   
      return Ax
      
   def Ay(self, k):
      Ay = [[0, 0, 0, 0, self.l],
            [0, 0, 0, 0, (self.l + 2*self.mu)],
            [0, 0, 0, self.mu, 0],
            [0, 0, 1.0, 0, 0],
            [0, 1.0, 0, 0, 0]]
      if k == 0:
         # Identity matrix
         Ay = [[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]
         Ay = numpy.array(Ay)
      else:
         Ay = numpy.array(Ay)
         for i in range(1, k):
            Ay = numpy.dot(Ay, Ay)   
      return Ay

   def binomial_expansion(self, k, variable):
      F = 0
      dim = self.u1.ufl_shape[0]
      test = [self.v[0,0], self.v[1,1], self.v[0,1], self.w[0], self.w[1]]
      
      # Binomial expansion
      for c in range(0, k+1):
         Ax = self.Ax(k-c); Ay = self.Ay(c)
         print "c = ", c
         print Ax, Ay
         ux = []; uy = []
         
         if k-c == 0:
            ux = (1,)*5
         else:
            ux.append(self.s0[0,0].dx(*((0,)*(k-c))))
            ux.append(self.s0[1,1].dx(*((0,)*(k-c))))
            ux.append(self.s0[0,1].dx(*((0,)*(k-c))))
            ux.append(self.u0[0].dx(*((0,)*(k-c))))
            ux.append(self.u0[1].dx(*((0,)*(k-c))))
            
         if c == 0:
            uy = (1,)*5
         else:
            uy.append(self.s0[0,0].dx(*((1,)*(c))))
            uy.append(self.s0[1,1].dx(*((1,)*(c))))
            uy.append(self.s0[0,1].dx(*((1,)*(c))))
            uy.append(self.u0[0].dx(*((1,)*(c))))
            uy.append(self.u0[1].dx(*((1,)*(c))))

         bx = [0, 0, 0, 0, 0]; by = [0, 0, 0, 0, 0]
         for i in range(len(ux)):
            for j in range(len(ux)):
               bx[i] += Ax[i][j]*ux[j]
               by[i] += Ay[i][j]*uy[j]
         
         F += test[variable]*binomial(k, c)*bx[variable]*by[variable]*dx

      return F

   @property
   def absorption(self):
      return self.absorption_function
   @absorption.setter
   def absorption(self, expression):
      self.absorption_function.interpolate(expression)
      
   @property
   def source(self):
      return self.source_function
   @source.setter
   def source(self, expression):
      self.source_function.interpolate(expression) 


   def form_u1(self):
      """ UFL for u1 equation. """
      T = Function(self.U)
      k = 4
      for i in range(0, k+1): # k+1 because we need to include k here
         T += ((self.dt**i)/factorial(i))*self.dudt_k(i) # Re-write the time derivative in terms of spatial derivatives.
      F = self.density*inner(self.w, self.u - T)*dx
      return F

   def solver_u1(self, F):
      """ Solver object for u1. """
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.u1)
      return LinearVariationalSolver(problem)

   def form_s1(self):
      """ UFL for s1 equation. """
      T = Function(self.S)
      k = 4
      for i in range(0, k+1): # k+1 because we need to include k here
         T += ((self.dt**i)/factorial(i))*self.dsdt_k(i) # Re-write the time derivative in terms of spatial derivatives.
      F = inner(self.v, self.s - T)*dx
      return F

   def solver_s1(self, F):
      """ Solver object for s1. """
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.s1)
      return LinearVariationalSolver(problem)
      
   def f(self, w, s0, u0, n, order, absorption=None):
      """ The RHS of the velocity equation. """
      f = self.binomial_expansion(order, 3) + self.binomial_expansion(order, 4)
      return f
      
   def g(self, v, u1, I, n, l, mu, order, source=None):
      """ The RHS of the stress equation. """
      g = self.binomial_expansion(order, 0) + self.binomial_expansion(order, 1) + self.binomial_expansion(order, 2)
      return g

   def dudt_k(self, k):
      """ Recursively compute (d^k)u/(dt^k). """
      if k == 0:
         return self.u0
      if k >= 1:
         temp = Function(self.U)
         solve(inner(self.w, self.u)*dx == self.f(self.w, self.s0, self.u0, self.n, k), temp)         
         return temp

   def dsdt_k(self, k):
      """ Recursively compute (d^k)s/(dt^k). """
      if k == 0:
         return self.s0
      if k >= 1:
         temp = Function(self.S)
         solve(inner(self.v, self.s)*dx == self.g(self.v, self.u1, self.I, self.n, self.l, self.mu, k), temp)
         return temp
      
   def write(self, u=None, s=None):
      """ Write the velocity and/or stress fields to file. """
      with timed_region('i/o'):
         if(u):
            self.u_stream << u
         if(s):
            pass # FIXME: Cannot currently write tensor valued fields to a VTU file.
            #self.s_stream << s

   def run(self, T):
      """ Run the elastic wave simulation until t = T. """
      #self.write(self.u1, self.s1.split()[0]) # Write out the initial condition.
      
      with timed_region('timestepping'):
         t = self.dt
         while t <= T + 1e-12:
            print "t = %f" % t
            
            # In case the source is time-dependent, update the time 't' here.
            if(self.source):
               with timed_region('source term update'):
                  self.source_expression.t = t
                  self.source = self.source_expression
            
            # Solve for the velocity vector field.
            with timed_region('velocity solve'):
               F = self.form_u1()
               solver = self.solver_u1(F)
               solver.solve()
               self.u0.assign(self.u1)
               #sys.exit()
            # Solve for the stress tensor field.
            with timed_region('stress solve'):
               F = self.form_s1()
               solver = self.solver_s1(F)
               solver.solve()
               self.s0.assign(self.s1)
            
            # Write out the new fields
            self.write(self.u1, self.s1.split()[0])
            
            # Move onto next timestep
            t += self.dt
      
      return self.u1, self.s1
