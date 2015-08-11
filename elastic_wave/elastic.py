#!/usr/bin/env python

from pyop2 import *
from pyop2.profiling import timed_region, summary
from pyop2.utils import cached_property
op2.init(lazy_evaluation=False)
from firedrake import *
import mpi4py
import numpy

class ElasticLF4(object):
   r""" An elastic wave equation solver, using the finite element method for spatial discretisation,
   and a fourth-order leap-frog time-stepping scheme. """

   def __init__(self, mesh, family, degree, dimension, explicit=True, output=True):
      r""" Initialise a new elastic wave simulation.
      
      :param mesh: The underlying computational mesh of vertices and edges.
      :param str family: Specify whether CG or DG should be used.
      :param int degree: Use polynomial basis functions of this degree.
      :param int dimension: The spatial dimension of the problem (1, 2 or 3).
      :param bool explicit: If False, use PETSc to solve for the solution fields. Otherwise, explicitly invert the mass matrix and perform a matrix-vector multiplication to get the solution.
      :param bool output: If True, output the solution fields to a file.
      :returns: None
      """
      with timed_region('function setup'):
         self.mesh = mesh
         self.dimension = dimension
         self.explicit = explicit
         self.output = output

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

      if self.output:
         with timed_region('i/o'):
            # File output streams
            self.u_stream = File("velocity.pvd")
            self.s_stream = File("stress.pvd")
      
   @property
   def absorption(self):
      r""" The absorption coefficient :math:`\sigma` for the absorption term
      
       .. math:: \sigma\mathbf{u}
       
      where :math:`\mathbf{u}` is the velocity field.
      """
      return self.absorption_function
   @absorption.setter
   def absorption(self, expression):
      r""" Setter function for the absorption field. 
      :param firedrake.Expression expression: The expression to interpolate onto the absorption field.
      """
      self.absorption_function.interpolate(expression)
      
   # Source term
   @property
   def source(self):
      r""" The source term on the RHS of the velocity (or stress) equation. """
      return self.source_function
   @source.setter
   def source(self, expression):
      r""" Setter function for the source field. 
      :param firedrake.Expression expression: The expression to interpolate onto the source field.
      """
      self.source_function.interpolate(expression) 

   def assemble_inverse_mass(self):
      r""" Compute the inverse of the consistent mass matrix for the velocity and stress equations.
      :returns: None
      """
      # Inverse of the (consistent) mass matrix for the velocity equation.
      self.inverse_mass_velocity = assemble(inner(self.w, self.u)*dx, inverse=True)
      self.inverse_mass_velocity.assemble()
      self.imass_velocity = self.inverse_mass_velocity.M
      # Inverse of the (consistent) mass matrix for the stress equation.
      self.inverse_mass_stress = assemble(inner(self.v, self.s)*dx, inverse=True)
      self.inverse_mass_stress.assemble()
      self.imass_stress = self.inverse_mass_stress.M
      
   @property
   def form_uh1(self):
      """ UFL for uh1 equation. """
      F = inner(self.w, self.u)*dx - self.f(self.w, self.s0, self.u0, self.n, self.absorption)
      return F

   @cached_property
   def rhs_uh1(self):
      """ RHS for uh1 equation. """
      return rhs(self.form_uh1)

   @property
   def form_stemp(self):
      """ UFL for stemp equation. """
      F = inner(self.v, self.s)*dx - self.g(self.v, self.uh1, self.I, self.n, self.l, self.mu, self.source)
      return F

   @cached_property
   def rhs_stemp(self):
      """ RHS for stemp equation. """
      return rhs(self.form_stemp)

   @property
   def form_uh2(self):
      """ UFL for uh2 equation. """
      F = inner(self.w, self.u)*dx - self.f(self.w, self.stemp, self.u0, self.n, self.absorption)
      return F

   @cached_property
   def rhs_uh2(self):
      """ RHS for uh2 equation. """
      return rhs(self.form_uh2)

   @property
   def form_u1(self):
      """ UFL for u1 equation. """
      if self.explicit:
         # Note that we have multiplied through by dt here.
         F = self.density*inner(self.w, self.u)*dx - self.density*inner(self.w, self.u0)*dx - self.dt*inner(self.w, self.uh1)*dx - ((self.dt**3)/24.0)*inner(self.w, self.uh2)*dx
      else:
         F = self.density*inner(self.w, (self.u - self.u0)/self.dt)*dx - inner(self.w, self.uh1)*dx - ((self.dt**2)/24.0)*inner(self.w, self.uh2)*dx
      return F

   @cached_property
   def rhs_u1(self):
      """ RHS for u1 equation. """
      return rhs(self.form_u1)
      
   @property
   def form_sh1(self):
      """ UFL for sh1 equation. """
      F = inner(self.v, self.s)*dx - self.g(self.v, self.u1, self.I, self.n, self.l, self.mu, self.source)
      return F

   @cached_property
   def rhs_sh1(self):
      """ RHS for sh1 equation. """
      return rhs(self.form_sh1)

   @property
   def form_utemp(self):
      """ UFL for utemp equation. """
      F = inner(self.w, self.u)*dx - self.f(self.w, self.sh1, self.u1, self.n, self.absorption)
      return F

   @cached_property
   def rhs_utemp(self):
      """ RHS for utemp equation. """
      return rhs(self.form_utemp)

   @property
   def form_sh2(self):
      """ UFL for sh2 equation. """
      F = inner(self.v, self.s)*dx - self.g(self.v, self.utemp, self.I, self.n, self.l, self.mu, self.source)
      return F

   @cached_property
   def rhs_sh2(self):
      """ RHS for sh2 equation. """
      return rhs(self.form_sh2)

   @property
   def form_s1(self):
      """ UFL for s1 equation. """
      if self.explicit:
         # Note that we have multiplied through by dt here.
         F = inner(self.v, self.s)*dx - inner(self.v, self.s0)*dx - self.dt*inner(self.v, self.sh1)*dx - ((self.dt**3)/24.0)*inner(self.v, self.sh2)*dx
      else:
         F = inner(self.v, (self.s - self.s0)/self.dt)*dx - inner(self.v, self.sh1)*dx - ((self.dt**2)/24.0)*inner(self.v, self.sh2)*dx
      return F

   @cached_property
   def rhs_s1(self):
      """ RHS for s1 equation. """
      return rhs(self.form_s1)

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

   def solve(self, rhs, matrix, result):
      r""" Solve by assembling RHS and applying inverse mass matrix.
      :param rhs: The RHS vector that the inverse mass matrix will be multiplied with.
      :param matrix: The inverse mass matrix.
      :param firedrake.Function result: The solution field.
      :returns: None"""
      F_a = assemble(rhs)
      with result.vector().dat.vec as res:
         with F_a.vector().dat.vec as F_v:
            matrix.handle.mult(F_v, res)

   def create_solver(self, form, result):
      r""" Create a solver object for a given form.
      :param ufl.Form form: The weak form of the equation that needs solving.
      :param firedrake.Function result: The field that will hold the solution.
      :returns: A LinearVariationalSolver object associated with the problem.
      """
      problem = LinearVariationalProblem(lhs(form), rhs(form), result)
      return LinearVariationalSolver(problem)

   def write(self, u=None, s=None):
      r""" Write the velocity and/or stress fields to file.
      :param firedrake.Function u: The velocity field.
      :param firedrake.Function s: The stress field.
      :returns: None
      """
      if self.output:
         with timed_region('i/o'):
            if(u):
               self.u_stream << u
            if(s):
               pass # FIXME: Cannot currently write tensor valued fields to a VTU file. See https://github.com/firedrakeproject/firedrake/issues/538
               #self.s_stream << s

   def run(self, T):
      """ Run the elastic wave simulation until t = T.
      :param float T: The finish time of the simulation.
      :returns: The final solution fields for velocity and stress.
      """
      self.write(self.u1, self.s1) # Write out the initial condition.

      if self.explicit:
         print "Generating inverse mass matrix"
         # Pre-assemble the inverse mass matrices, which should stay
         # constant throughout the simulation (assuming no mesh adaptivity).
         with timed_region('inverse mass matrix'):
            self.assemble_inverse_mass()
      else:
         print "Creating solver contexts"
         with timed_region('solver setup'):
            solver_uh1 = self.create_solver(self.form_uh1, self.uh1)
            solver_stemp = self.create_solver(self.form_stemp, self.stemp)
            solver_uh2 = self.create_solver(self.form_uh2, self.uh2)
            solver_u1 = self.create_solver(self.form_u1, self.u1)
            solver_sh1 = self.create_solver(self.form_sh1, self.sh1)
            solver_utemp = self.create_solver(self.form_utemp, self.utemp)
            solver_sh2 = self.create_solver(self.form_sh2, self.sh2)
            solver_s1 = self.create_solver(self.form_s1, self.s1)

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
            if self.explicit:
               with timed_region('velocity solve'):
                  self.solve(self.rhs_uh1, self.imass_velocity, self.uh1)
                  self.solve(self.rhs_stemp, self.imass_stress, self.stemp)
                  self.solve(self.rhs_uh2, self.imass_velocity, self.uh2)
                  self.solve(self.rhs_u1, self.imass_velocity, self.u1)
            else:
               with timed_region('velocity solve'):
                  solver_uh1.solve()
                  solver_stemp.solve()
                  solver_uh2.solve()
                  solver_u1.solve()
            self.u0.assign(self.u1)
            
            # Solve for the stress tensor field.
            if self.explicit:
               with timed_region('stress solve'):
                  self.solve(self.rhs_sh1, self.imass_stress, self.sh1)
                  self.solve(self.rhs_utemp, self.imass_velocity, self.utemp)
                  self.solve(self.rhs_sh2, self.imass_stress, self.sh2)
                  self.solve(self.rhs_s1, self.imass_stress, self.s1)
            else:
               with timed_region('stress solve'):
                  solver_sh1.solve()
                  solver_utemp.solve()
                  solver_sh2.solve()
                  solver_s1.solve()
            self.s0.assign(self.s1)
            
            # Write out the new fields
            self.write(self.u1, self.s1)
            
            # Move onto next timestep
            t += self.dt
      
      return self.u1, self.s1
