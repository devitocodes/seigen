#!/usr/bin/env python

from pyop2 import *
from pyop2.profiling import timed_region
op2.init(lazy_evaluation=False)
from firedrake import *
from elastic_wave.helpers import log
import mpi4py
from abc import ABCMeta, abstractmethod


class ElasticLF4(object):
    r""" An elastic wave equation solver, using the finite element
    method for spatial discretisation, and a fourth-order leap-frog
    time-stepping scheme.

    Note that this base class only provides the UFL equations and I/O
    methods for the model. Full scale elastic wave equation solver
    should be created using the static method:
    ElasticLF4.create(mesh, dimension, degree, solver)"""
    __metaclass__ = ABCMeta

    @staticmethod
    def create(mesh, family, degree, dimension, explicit=True, output=True):
        r""" Create an elastic wave equation solver for the given mesh
        according to specified spatial discretisation details and
        solver methods.

        :param mesh: The underlying computational mesh of vertices and edges.
        :param str family: Specify whether CG or DG should be used.
        :param int degree: Use polynomial basis functions of this degree.
        :bool explicit: If False, use PETSc to solve for the solution fields.
                        Otherwise, explicitly invert the mass matrix and perform
                        a matrix-vector multiplication to get the solution..
        :param int dimension: The spatial dimension of the problem (1, 2 or 3).
        :param bool output: If True, output the solution fields to a file.
        :returns: None
        """
        if explicit:
            return ExplicitElasticLF4(mesh, family, degree, dimension, output=output)
        else:
            return ImplicitElasticLF4(mesh, family, degree, dimension, output=output)

    def __init__(self, mesh, family, degree, dimension, explicit=True, output=True):
        r""" Initialise a new elastic wave simulation.

        :param mesh: The underlying computational mesh of vertices and edges.
        :param str family: Specify whether CG or DG should be used.
        :param int degree: Use polynomial basis functions of this degree.
        :param int dimension: The spatial dimension of the problem (1, 2 or 3).
        :param bool output: If True, output the solution fields to a file.
        :returns: None
        """
        with timed_region('function setup'):
            self.mesh = mesh
            self.dimension = dimension
            self.output = output

            self.S = TensorFunctionSpace(mesh, family, degree)
            self.U = VectorFunctionSpace(mesh, family, degree)
            # Assumes that the S and U function spaces are the same.
            dofs = op2.MPI.comm.allreduce(self.S.dof_count, op=mpi4py.MPI.SUM)
            log("Number of degrees of freedom: %d" % dofs)

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

            # Inverse mass matrices for explicit methods
            self.invmass_velocity = None
            self.invmass_stress = None

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

    @property
    def form_uh1(self):
        """ UFL for uh1 equation. """
        F = inner(self.w, self.u)*dx - self.f(self.w, self.s0, self.u0, self.n, self.absorption)
        return F

    @property
    def form_stemp(self):
        """ UFL for stemp equation. """
        F = inner(self.v, self.s)*dx - self.g(self.v, self.uh1, self.I, self.n, self.l, self.mu, self.source)
        return F

    @property
    def form_uh2(self):
        """ UFL for uh2 equation. """
        F = inner(self.w, self.u)*dx - self.f(self.w, self.stemp, self.u0, self.n, self.absorption)
        return F

    @property
    def form_u1(self):
        """ UFL for u1 equation. """
        return self.density*inner(self.w, (self.u - self.u0)/self.dt)*dx \
            - inner(self.w, self.uh1)*dx - ((self.dt**2)/24.0)*inner(self.w, self.uh2)*dx

    @property
    def form_sh1(self):
        """ UFL for sh1 equation. """
        F = inner(self.v, self.s)*dx - self.g(self.v, self.u1, self.I, self.n, self.l, self.mu, self.source)
        return F

    @property
    def form_utemp(self):
        """ UFL for utemp equation. """
        F = inner(self.w, self.u)*dx - self.f(self.w, self.sh1, self.u1, self.n, self.absorption)
        return F

    @property
    def form_sh2(self):
        """ UFL for sh2 equation. """
        F = inner(self.v, self.s)*dx - self.g(self.v, self.utemp, self.I, self.n, self.l, self.mu, self.source)
        return F

    @property
    def form_s1(self):
        """ UFL for s1 equation. """
        return inner(self.v, (self.s - self.s0)/self.dt)*dx \
            - inner(self.v, self.sh1)*dx - ((self.dt**2)/24.0)*inner(self.v, self.sh2)*dx

    def f(self, w, s0, u0, n, absorption=None):
        """ The RHS of the velocity equation. """
        f = -inner(grad(w), s0)*dx + inner(avg(s0)*n('+'), w('+'))*dS + inner(avg(s0)*n('-'), w('-'))*dS
        if(absorption):
            f += -inner(w, absorption*u0)*dx
        return f

    def g(self, v, u1, I, n, l, mu, source=None):
        """ The RHS of the stress equation. """
        g = - l*(v[i, j]*I[i, j]).dx(k)*u1[k]*dx + l*(jump(v[i, j], n[k])*I[i, j]*avg(u1[k]))*dS \
            + l*(v[i, j]*I[i, j]*u1[k]*n[k])*ds - mu*inner(div(v), u1)*dx + mu*inner(avg(u1), jump(v, n))*dS \
            - mu*inner(div(v.T), u1)*dx + mu*inner(avg(u1), jump(v.T, n))*dS \
            + mu*inner(u1, dot(v, n))*ds + mu*inner(u1, dot(v.T, n))*ds
        if(source):
            g += inner(v, source)*dx
        return g

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
                    # FIXME: Cannot currently write tensor valued fields to a VTU file.
                    # See https://github.com/firedrakeproject/firedrake/issues/538
                    # self.s_stream << s
                    pass

    @abstractmethod
    def create_solver(self, *args):
        r""" Abstract method to generate solver context for a single form."""
        return None

    @abstractmethod
    def solve(self, *args):
        r""" Abstract method to solve individual form."""
        pass

    def setup(self, *args):
        r""" Generate method-specific solver contexts for all forms."""
        log("Creating solver contexts")
        with timed_region('solver setup'):
            self.ctx_uh1 = self.create_solver(self.form_uh1, self.uh1)
            self.ctx_stemp = self.create_solver(self.form_stemp, self.stemp)
            self.ctx_uh2 = self.create_solver(self.form_uh2, self.uh2)
            self.ctx_u1 = self.create_solver(self.form_u1, self.u1)
            self.ctx_sh1 = self.create_solver(self.form_sh1, self.sh1)
            self.ctx_utemp = self.create_solver(self.form_utemp, self.utemp)
            self.ctx_sh2 = self.create_solver(self.form_sh2, self.sh2)
            self.ctx_s1 = self.create_solver(self.form_s1, self.s1)

    def run(self, T):
        """ Run the elastic wave simulation until t = T.
        :param float T: The finish time of the simulation.
        :returns: The final solution fields for velocity and stress.
        """
        # Write out the initial condition.
        self.write(self.u1, self.s1)

        # Call solver-specific setup
        self.setup()

        with timed_region('timestepping'):
            t = self.dt
            while t <= T + 1e-12:
                log("t = %f" % t)

                # In case the source is time-dependent, update the time 't' here.
                if(self.source):
                    with timed_region('source term update'):
                        self.source_expression.t = t
                        self.source = self.source_expression

                # Solve for the velocity vector field.
                with timed_region('velocity solve'):
                    self.solve(self.ctx_uh1, self.invmass_velocity, self.uh1)
                    self.solve(self.ctx_stemp, self.invmass_stress, self.stemp)
                    self.solve(self.ctx_uh2, self.invmass_velocity, self.uh2)
                    self.solve(self.ctx_u1, self.invmass_velocity, self.u1)
                self.u0.assign(self.u1)

                # Solve for the stress tensor field.
                with timed_region('stress solve'):
                    self.solve(self.ctx_sh1, self.invmass_stress, self.sh1)
                    self.solve(self.ctx_utemp, self.invmass_velocity, self.utemp)
                    self.solve(self.ctx_sh2, self.invmass_stress, self.sh2)
                    self.solve(self.ctx_s1, self.invmass_stress, self.s1)
                self.s0.assign(self.s1)

                # Write out the new fields
                self.write(self.u1, self.s1)

                # Move onto next timestep
                t += self.dt

        return self.u1, self.s1


class ImplicitElasticLF4(ElasticLF4):
    r""" Elastic equation solver that implicitly solves individual UFL
    forms by wrapping them as firedrake.LinearVariatonalProblem().
    """
    def create_solver(self, form, result):
        r""" Create a solver object for a given form.
        :param ufl.Form form: The weak form of the equation that needs solving.
        :param firedrake.Function result: The field that will hold the solution.
        :returns: A LinearVariationalSolver object associated with the problem.
        """
        problem = LinearVariationalProblem(lhs(form), rhs(form), result)
        return LinearVariationalSolver(problem)

    def solve(self, solver, *args):
        solver.solve()


class ExplicitElasticLF4(ElasticLF4):
    r""" Elastic equation solver that explicitly solves individual UFL
    forms by assembling RHS vectors and multiplying them with the
    according global inverse mass matrix.
    """
    @property
    def form_u1(self):
        """ UFL for u1 equation. """
        # Note that we have multiplied through by dt here.
        return self.density*inner(self.w, self.u)*dx - self.density*inner(self.w, self.u0)*dx \
            - self.dt*inner(self.w, self.uh1)*dx - ((self.dt**3)/24.0)*inner(self.w, self.uh2)*dx

    @property
    def form_s1(self):
        """ UFL for s1 equation. """
        # Note that we have multiplied through by dt here.
        return inner(self.v, self.s)*dx - inner(self.v, self.s0)*dx \
            - self.dt*inner(self.v, self.sh1)*dx - ((self.dt**3)/24.0)*inner(self.v, self.sh2)*dx

    def create_solver(self, form, *args):
        r""" Solution context for explicit methods is the compiled RHS kernel object."""
        return rhs(form)

    def solve(self, rhs, matrix, result):
        r""" Solve by assembling into RHS vector and applying inverse mass matrix.
        :param rhs: The RHS vector that the inverse mass matrix will be multiplied with.
        :param matrix: The inverse mass matrix.
        :param firedrake.Function result: The solution field.
        :returns: None"""
        F_a = assemble(rhs)
        with result.dat.vec as res:
            with F_a.dat.vec_ro as F_v:
                matrix.handle.mult(F_v, res)

    def setup(self):
        r""" Pre-assembles the inverse of the consistent mass matrix for the
        velocity and stress equations.
        :returns: None
        """
        log("Generating inverse mass matrices")
        # Inverse of the (consistent) mass matrix for the velocity equation.
        self.inverse_mass_velocity = assemble(inner(self.w, self.u)*dx, inverse=True)
        self.inverse_mass_velocity.assemble()
        self.invmass_velocity = self.inverse_mass_velocity.M
        # Inverse of the (consistent) mass matrix for the stress equation.
        self.inverse_mass_stress = assemble(inner(self.v, self.s)*dx, inverse=True)
        self.inverse_mass_stress.assemble()
        self.invmass_stress = self.inverse_mass_stress.M

        # Setup RHS assembly objects
        super(ExplicitElasticLF4, self).setup()
