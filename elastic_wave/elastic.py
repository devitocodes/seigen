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

class ElasticLF4(object):
   """ Elastic wave equation solver using the finite element method and a fourth-order leap-frog time-stepping scheme. """

   def __init__(self, mesh, family, degree, dimension):
      with timed_region('function setup'):
         self.mesh = mesh
         self.dimension = dimension

         self.S = FunctionSpace(mesh, family, degree)
         self.U = FunctionSpace(mesh, family, degree)

         self.WS = MixedFunctionSpace([self.S, self.S, self.S, self.S])
         self.WU = MixedFunctionSpace([self.U, self.U])

         self.s = TrialFunctions(self.WS)
         self.v = TestFunctions(self.WS)
         self.u = TrialFunctions(self.WU)
         self.w = TestFunctions(self.WU)

         self.s0 = Function(self.WS, name="StressOld")
         self.sh1 = Function(self.WS, name="StressHalf1")
         self.stemp = Function(self.WS, name="StressTemp")
         self.sh2 = Function(self.WS, name="StressHalf2")
         self.s1 = Function(self.WS, name="StressNew")

         self.u0 = Function(self.WU, name="VelocityOld")
         self.uh1 = Function(self.WU, name="VelocityHalf1")
         self.utemp = Function(self.WU, name="VelocityTemp")
         self.uh2 = Function(self.WU, name="VelocityHalf2")
         self.u1 = Function(self.WU, name="VelocityNew")
         
         self.absorption_function = None
         self.source_function = None
         self.source_expression = None
         self._dt = None
         self._density = None
         self._mu = None
         self._l = None
         
         self.n = FacetNormal(self.mesh)

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
      F = inner(self.w[0], self.u[0])*dx - self.fx(self.w, self.s0, self.u0, self.n, self.absorption)
      if(self.dimension == 2):
         F += inner(self.w[1], self.u[1])*dx - self.fy(self.w, self.s0, self.u0, self.n, self.absorption)
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
      F = inner(self.v[0], self.s[0])*dx - self.gxx(self.v, self.uh1, self.n, self.l, self.mu, self.source)
      if(self.dimension == 2):
         F += inner(self.v[1], self.s[1])*dx - self.gxy(self.v, self.uh1, self.n, self.l, self.mu)
         F += inner(self.v[2], self.s[2])*dx - self.gyx(self.v, self.uh1, self.n, self.l, self.mu)
         F += inner(self.v[3], self.s[3])*dx - self.gyy(self.v, self.uh1, self.n, self.l, self.mu, self.source)
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
      F = inner(self.w[0], self.u[0])*dx - self.fx(self.w, self.stemp, self.u0, self.n, self.absorption)
      if(self.dimension == 2):
         F += inner(self.w[1], self.u[1])*dx - self.fy(self.w, self.stemp, self.u0, self.n, self.absorption)
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
      F = self.density*inner(self.w[0], (self.u[0] - self.u0[0])/self.dt)*dx - inner(self.w[0], self.uh1[0])*dx - ((self.dt**2)/24.0)*inner(self.w[0], self.uh2[0])*dx
      if(self.dimension == 2):
         F += self.density*inner(self.w[1], (self.u[1] - self.u0[1])/self.dt)*dx - inner(self.w[1], self.uh1[1])*dx - ((self.dt**2)/24.0)*inner(self.w[1], self.uh2[1])*dx
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
      F = inner(self.v[0], self.s[0])*dx - self.gxx(self.v, self.u1, self.n, self.l, self.mu, self.source)
      if(self.dimension == 2):
         F += inner(self.v[1], self.s[1])*dx - self.gxy(self.v, self.u1, self.n, self.l, self.mu)
         F += inner(self.v[2], self.s[2])*dx - self.gyx(self.v, self.u1, self.n, self.l, self.mu)
         F += inner(self.v[3], self.s[3])*dx - self.gyy(self.v, self.u1, self.n, self.l, self.mu, self.source)
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
      F = inner(self.w[0], self.u[0])*dx - self.fx(self.w, self.sh1, self.u1, self.n, self.absorption)
      if(self.dimension == 2):
         F += inner(self.w[1], self.u[1])*dx - self.fy(self.w, self.sh1, self.u1, self.n, self.absorption)
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
      F = inner(self.v[0], self.s[0])*dx - self.gxx(self.v, self.utemp, self.n, self.l, self.mu, self.source)
      if(self.dimension == 2):
         F += inner(self.v[1], self.s[1])*dx - self.gxy(self.v, self.utemp, self.n, self.l, self.mu)
         F += inner(self.v[2], self.s[2])*dx - self.gyx(self.v, self.utemp, self.n, self.l, self.mu)
         F += inner(self.v[3], self.s[3])*dx - self.gyy(self.v, self.utemp, self.n, self.l, self.mu, self.source)
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
      F = inner(self.v[0], (self.s[0] - self.s0[0])/self.dt)*dx - inner(self.v[0], self.sh1[0])*dx - ((self.dt**2)/24.0)*inner(self.v[0], self.sh2[0])*dx
      if(self.dimension == 2):
         F += inner(self.v[1], (self.s[1] - self.s0[1])/self.dt)*dx - inner(self.v[1], self.sh1[1])*dx - ((self.dt**2)/24.0)*inner(self.v[1], self.sh2[1])*dx
         F += inner(self.v[2], (self.s[2] - self.s0[2])/self.dt)*dx - inner(self.v[2], self.sh1[2])*dx - ((self.dt**2)/24.0)*inner(self.v[2], self.sh2[2])*dx
         F += inner(self.v[3], (self.s[3] - self.s0[3])/self.dt)*dx - inner(self.v[3], self.sh1[3])*dx - ((self.dt**2)/24.0)*inner(self.v[3], self.sh2[3])*dx
      return F
    
   @property
   def solver_s1(self):
      """ Solver object for s1. """
      F = self.form_s1
      problem = LinearVariationalProblem(lhs(F), rhs(F), self.s1)
      return LinearVariationalSolver(problem)
   
   def fx(self, w, s0, u0, n, absorption=None):
      """ x-component of the velocity equation RHS. """
      fx = -inner(grad(w[0])[0], s0[0])*dx + inner(avg(s0[0]), jump(w[0], n[0]))*dS - inner(grad(w[0])[1], s0[1])*dx + inner(avg(s0[1]), jump(w[0], n[1]))*dS
      if(absorption):
         fx += -inner(w[0], absorption*u0[0])*dx
      return fx
      
   def fy(self, w, s0, u0, n, absorption=None):
      """ y-component of the velocity equation RHS. """
      fy = -inner(grad(w[1])[0], s0[2])*dx + inner(avg(s0[2]), jump(w[1], n[0]))*dS - inner(grad(w[1])[1], s0[3])*dx + inner(avg(s0[3]), jump(w[1], n[1]))*dS
      if(absorption):
         fy += -inner(w[1], absorption*u0[1])*dx
      return fy

   
   def gxx(self, v, u1, n, l, mu, source=None):
      """ xx-component of stress equation RHS. """
      gxx =  - (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
             + (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
             + (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
             - l*inner(grad(v[0])[1], u1[1])*dx \
             + l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
             + l*inner(v[0], u1[1]*n[1])*ds
      if(source):
         gxx += inner(v[0], source)*dx
      return gxx
      
   def gxy(self, v, u1, n, l, mu):
      """ xy-component of stress equation RHS. """
      return - mu*(inner(grad(v[1])[0], u1[1]))*dx \
             + mu*(inner(v[1], u1[1]*n[0]))*ds \
             + mu*(inner(jump(v[1], n[0]), avg(u1[1])))*dS \
             - mu*(inner(grad(v[1])[1], u1[0]))*dx \
             + mu*(inner(v[1], u1[0]*n[1]))*ds \
             + mu*(inner(jump(v[1], n[1]), avg(u1[0])))*dS

   def gyx(self, v, u1, n, l, mu):
      """ yx-component of stress equation RHS. """
      return - mu*(inner(grad(v[2])[0], u1[1]))*dx \
             + mu*(inner(v[2], u1[1]*n[0]))*ds \
             + mu*(inner(jump(v[2], n[0]), avg(u1[1])))*dS \
             - mu*(inner(grad(v[2])[1], u1[0]))*dx \
             + mu*(inner(v[2], u1[0]*n[1]))*ds \
             + mu*(inner(jump(v[2], n[1]), avg(u1[0])))*dS
             
   def gyy(self, v, u1, n, l, mu, source=None):  
      """ yy-component of stress equation RHS. """  
      gyy =  - l*inner(grad(v[3])[0], u1[0])*dx \
             + l*inner(v[3], u1[0]*n[0])*ds \
             + l*inner(jump(v[3], n[0]), avg(u1[0]))*dS \
             - (l + 2*mu)*inner(grad(v[3])[1], u1[1])*dx \
             + (l + 2*mu)*inner(jump(v[3], n[1]), avg(u1[1]))*dS \
             + (l + 2*mu)*inner(v[3], u1[1]*n[1])*ds
      if(source):
         gyy += inner(v[3], source)*dx
      return gyy


   def write(self, u=None, s=None):
      """ Write the velocity and/or stress fields to file. """
      with timed_region('i/o'):
         if(u):
            self.u_stream << u
         if(s):
            self.s_stream << s

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
