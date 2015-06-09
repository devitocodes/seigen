#!/usr/bin/env python

# PETSc environment variables
import os
try:
   if(os.environ["PETSC_OPTIONS"] == ""):
      os.environ["PETSC_OPTIONS"] = "-log_summary"
   else:
      os.environ["PETSC_OPTIONS"] = os.environ["PETSC_OPTIONS"] + " -log_summary"
except KeyError:
   # Environment variable does not exist, so let's set it now.
   os.environ["PETSC_OPTIONS"] = "-log_summary"
print "Environment variable PETSC_OPTIONS set to: %s" % (os.environ["PETSC_OPTIONS"])

from pyop2 import *
from pyop2.profiling import timed_region, summary
op2.init(lazy_evaluation=False)
from firedrake import *
import mpi4py

class ElasticLF4(object):
   """ Elastic wave equation solver using the finite element method and a fourth-order leap-frog time-stepping scheme. """

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
         self.sh1 = Function(self.S, name="StressHalf1")
         self.stemp = Function(self.S, name="StressTemp")
         self.sh2 = Function(self.S, name="StressHalf2")
         self.s1 = Function(self.S, name="StressNew")

         self.u0 = Function(self.U, name="VelocityOld")
         self.uh1 = Function(self.U, name="VelocityHalf1")
         self.utemp = Function(self.U, name="VelocityTemp")
         self.uh2 = Function(self.U, name="VelocityHalf2")
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

   @property
   def form_uh1(self):
      """ UFL for uh1 equation. """
      F = inner(self.w, self.u)*dx - self.f(self.w, self.s0, self.u0, self.n, self.absorption)
      return F
   @property
   def solver_uh1(self):
      """ Solver object for uh1. """
      F = self.form_uh1
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.uh1)
      return LinearVariationalSolver(problem)

   @property
   def form_stemp(self):
      """ UFL for stemp equation. """
      F = inner(self.v, self.s)*dx - self.g(self.v, self.uh1, self.I, self.n, self.l, self.mu, self.source)
      return F
   @property
   def solver_stemp(self):
      """ Solver object for stemp. """
      F = self.form_stemp
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.stemp)
      return LinearVariationalSolver(problem)

   @property
   def form_uh2(self):
      """ UFL for uh2 equation. """
      F = inner(self.w, self.u)*dx - self.f(self.w, self.stemp, self.u0, self.n, self.absorption)
      return F
   @property
   def solver_uh2(self):
      """ Solver object for uh2. """
      F = self.form_uh2
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.uh2)
      return LinearVariationalSolver(problem)

   @property
   def form_u1(self):
      """ UFL for u1 equation. """
      F = self.density*inner(self.w, (self.u - self.u0)/self.dt)*dx - inner(self.w, self.uh1)*dx - ((self.dt**2)/24.0)*inner(self.w, self.uh2)*dx
      return F
   @property
   def solver_u1(self):
      """ Solver object for u1. """
      F = self.form_u1
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.u1)
      return LinearVariationalSolver(problem)
      
   @property
   def form_sh1(self):
      """ UFL for sh1 equation. """
      F = inner(self.v, self.s)*dx - self.g(self.v, self.u1, self.I, self.n, self.l, self.mu, self.source)
      return F
   @property
   def solver_sh1(self):
      """ Solver object for sh1. """
      F = self.form_sh1
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.sh1)
      return LinearVariationalSolver(problem)

   @property
   def form_utemp(self):
      """ UFL for utemp equation. """
      F = inner(self.w, self.u)*dx - self.f(self.w, self.sh1, self.u1, self.n, self.absorption)
      return F
   @property
   def solver_utemp(self):
      """ Solver object for utemp. """
      F = self.form_utemp
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.utemp)
      return LinearVariationalSolver(problem)

   @property
   def form_sh2(self):
      """ UFL for sh2 equation. """
      F = inner(self.v, self.s)*dx - self.g(self.v, self.utemp, self.I, self.n, self.l, self.mu, self.source)
      return F
   @property
   def solver_sh2(self):
      """ Solver object for sh2. """
      F = self.form_sh2
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.sh2)
      return LinearVariationalSolver(problem)

   @property
   def form_s1(self):
      """ UFL for s1 equation. """
      F = inner(self.v, (self.s - self.s0)/self.dt)*dx - inner(self.v, self.sh1)*dx - ((self.dt**2)/24.0)*inner(self.v, self.sh2)*dx
      return F
    
   @property
   def solver_s1(self):
      """ Solver object for s1. """
      F = self.form_s1
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.s1)
      return LinearVariationalSolver(problem)
   
   def f(self, w, s0, u0, n, absorption=None):
      """ The RHS of the velocity equation. """
      f = -inner(grad(w), s0)*dx + inner(avg(s0)*n('+'), w('+'))*dS + inner(avg(s0)*n('-'), w('-'))*dS
      if(absorption):
         f += -inner(w, absorption*u0)*dx
      return f
   
   def g(self, v, u1, I, n, l, mu, source=None):
      """ The RHS of the stress equation. """
      g =  - l*(v[i,j]*I[i,j]).dx(k)*u1[k]*dx + l*(jump(v[i,j], n[k])*I[i,j]*avg(u1[k]))*dS + l*(v[i,j]*I[i,j]*u1[k]*n[k])*ds - mu*inner(div(v), u1)*dx + mu*inner(avg(u1), jump(v, n))*dS - mu*inner(div(v.T), u1)*dx + mu*inner(avg(u1), jump(v.T, n))*dS + mu*inner(u1, dot(v, n))*ds + mu*inner(u1, dot(v.T, n))*ds
      if(source):
         g += inner(v, source)*dx
      return g

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
      self.write(self.u1, self.s1.split()[0]) # Write out the initial condition.
      
      # Construct the solver objects outside of the time-stepping loop.
      with timed_region('solver setup'):
         solver_uh1 = self.solver_uh1
         solver_stemp = self.solver_stemp
         solver_uh2 = self.solver_uh2
         solver_u1 = self.solver_u1
         solver_sh1 = self.solver_sh1
         solver_utemp = self.solver_utemp
         solver_sh2 = self.solver_sh2
         solver_s1 = self.solver_s1
      
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
               solver_uh1.solve()
               solver_stemp.solve()
               solver_uh2.solve()
               solver_u1.solve()
               self.u0.assign(self.u1)
            
            # Solve for the stress tensor field.
            with timed_region('stress solve'):
               solver_sh1.solve()
               solver_utemp.solve()
               solver_sh2.solve()
               solver_s1.solve()
               self.s0.assign(self.s1)
            
            # Write out the new fields
            self.write(self.u1, self.s1.split()[0])
            
            # Move onto next timestep
            t += self.dt
      
      return self.u1, self.s1
