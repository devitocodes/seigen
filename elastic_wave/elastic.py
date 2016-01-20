#!/usr/bin/env python

from pyop2 import *
from pyop2.profiling import timed_region
from pyop2.utils import cached_property
from pyop2.fusion import loop_chain
from pyop2.base import _trace
from firedrake import *
from firedrake.petsc import PETSc
from elastic_wave.helpers import log
import mpi4py
import numpy as np
import coffee.base as ast


def calculate_sdepth(num_solves, num_unroll, extra_halo):
    """The sdepth is calculated through the following formula:

        sdepth = 1 if sequential else 1 + num_solves*num_unroll + extra_halo

    Where:

    :arg num_solves: number of solves per loop chain iteration
    :arg num_unroll: unroll factor for the loop chain
    :arg extra_halo: to expose the nonexec region to the tiling engine
    """
    if MPI.parallel:
        return 1 + num_solves*num_unroll + extra_halo
    else:
        return 1


class ElasticLF4(object):
    r""" An elastic wave equation solver, using the finite element method for spatial discretisation,
    and a fourth-order leap-frog time-stepping scheme. """

    # Constants
    loop_chain_length = 28
    num_solves = 8

    def __init__(self, mesh, family, degree, dimension, solver='explicit', output=True):
        r""" Initialise a new elastic wave simulation.

        :param mesh: The underlying computational mesh of vertices and edges.
        :param str family: Specify whether CG or DG should be used.
        :param int degree: Use polynomial basis functions of this degree.
        :param int dimension: The spatial dimension of the problem (1, 2 or 3).
        :param str mode: Solver mode, recognised values are:
            'implicit': Use PETSc KSP solver to solve for the solution fields.
            'explicit': Explicitly invert the mass matrix and perform a
                        matrix-vector multiplication using PETSc MatMult.
            'parloop': Explicitly invert the mass matrix and perform a
                       matrix-vector multiplication using a PyOP2 Parloop.
        :param bool output: If True, output the solution fields to a file.
        :returns: None
        """
        with timed_region('function setup'):
            self.mesh = mesh
            self.dimension = dimension
            self.solver = solver
            self.explicit = solver in ['explicit', 'parloop', 'fusion', 'tile']
            self.output = output

            self.tile_size = 1000
            self.extra_halo = 0
            self.tiling_mode = 'hard'
            if solver in ['fusion', 'tile']:
                self.num_unroll = 1
                s_depth = calculate_sdepth(self.num_solves,
                                           self.num_unroll,
                                           self.extra_halo)
                self.mesh.topology.init(s_depth=s_depth)
                # This is only to print out info related to tiling
                if solver == 'tile':
                    slope(mesh, debug=True)
                    self.tiling_mode = 'tile'
            else:
                self.num_unroll = 0

            self.S = TensorFunctionSpace(mesh, family, degree, name='S')
            self.U = VectorFunctionSpace(mesh, family, degree, name='U')
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

            # AST cache
            self.asts = {}

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

    def matrix_to_dat(self, massmatrix, functionspace):
        # Copy the velocity mass matrix into a Dat
        arity = sum(functionspace.topological.dofs_per_entity)*functionspace.topological.dim
        dat = Dat(DataSet(self.mesh.cell_set, arity*arity), dtype='double')
        istart, iend = massmatrix.handle.getOwnershipRange()
        idxs = [PETSc.IS().createGeneral(np.arange(i, i+arity, dtype=np.int32),
                                         comm=PETSc.COMM_SELF)
                for i in range(istart, iend, arity)]
        submats = massmatrix.handle.getSubMatrices(idxs, idxs)
        for i, m in enumerate(submats):
            dat.data[i] = m[:, :].flatten()
        return dat

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
            F = self.density*inner(self.w, self.u)*dx - self.density*inner(self.w, self.u0)*dx \
                - self.dt*inner(self.w, self.uh1)*dx - ((self.dt**3)/24.0)*inner(self.w, self.uh2)*dx
        else:
            F = self.density*inner(self.w, (self.u - self.u0)/self.dt)*dx - inner(self.w, self.uh1)*dx \
                - ((self.dt**2)/24.0)*inner(self.w, self.uh2)*dx
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
            F = inner(self.v, self.s)*dx - inner(self.v, self.s0)*dx \
                - self.dt*inner(self.v, self.sh1)*dx \
                - ((self.dt**3)/24.0)*inner(self.v, self.sh2)*dx
        else:
            F = inner(self.v, (self.s - self.s0)/self.dt)*dx \
                - inner(self.v, self.sh1)*dx \
                - ((self.dt**2)/24.0)*inner(self.v, self.sh2)*dx
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
        g = - l*(v[i, j]*I[i, j]).dx(k)*u1[k]*dx + l*(jump(v[i, j], n[k])*I[i, j]*avg(u1[k]))*dS \
            + l*(v[i, j]*I[i, j]*u1[k]*n[k])*ds - mu*inner(div(v), u1)*dx + mu*inner(avg(u1), jump(v, n))*dS \
            - mu*inner(div(v.T), u1)*dx + mu*inner(avg(u1), jump(v.T, n))*dS \
            + mu*inner(u1, dot(v, n))*ds + mu*inner(u1, dot(v.T, n))*ds
        if(source):
            g += inner(v, source)*dx
        return g

    def solve_petsc(self, rhs, matrix, result):
        r""" Solve by assembling RHS and applying inverse mass matrix using PETSc MatMult.
        :param rhs: The RHS vector that the inverse mass matrix will be multiplied with.
        :param matrix: The inverse mass matrix.
        :param firedrake.Function result: The solution field.
        :returns: None"""
        F_a = assemble(rhs)
        with result.dat.vec as res:
            with F_a.dat.vec_ro as F_v:
                matrix.handle.mult(F_v, res)

    def solve_parloop(self, rhs, matrix_asdat, result):
        r""" Solve by assembling RHS and applying inverse mass matrix using a PyOP2 Parloop.
        :param rhs: The RHS vector that the inverse mass matrix will be multiplied with.
        :param matrix: The inverse mass matrix.
        :param firedrake.Function result: The solution field.
        :returns: None"""
        F_a = assemble(rhs)
        ast_matmul = self.ast_matmul(F_a)

        # Create the par loop (automatically added to the trace of loops to be executed)
        kernel = op2.Kernel(ast_matmul, ast_matmul.name)
        op2.par_loop(kernel, self.mesh.cell_set,
                     matrix_asdat(op2.READ),
                     F_a.dat(op2.READ, F_a.cell_node_map()),
                     result.dat(op2.WRITE, result.cell_node_map()))

    def ast_matmul(self, F_a):
        """Generate an AST for a PyOP2 kernel performing a matrix-vector multiplication."""

        # The number of dofs on each element is /ndofs*cdim/
        F_a_fs = F_a.function_space()
        ndofs = sum(F_a_fs.topological.dofs_per_entity)
        cdim = F_a_fs.dim
        name = 'mat_vec_mul_kernel_%s' % F_a_fs.name

        identifier = (ndofs, cdim, name)
        if identifier in self.asts:
            return self.asts[identifier]

        # Craft the AST
        body = ast.Incr(ast.Symbol('C', ('i/%d' % cdim, 'i%%%d' % cdim)),
                        ast.Prod(ast.Symbol('A', ('i',), ((ndofs*cdim, 'j*%d + k' % cdim),)),
                                 ast.Symbol('B', ('j', 'k'))))
        body = ast.c_for('k', cdim, body).children[0]
        body = [ast.Assign(ast.Symbol('C', ('i/%d' % cdim, 'i%%%d' % cdim)), '0.0'),
                ast.c_for('j', ndofs, body).children[0]]
        body = ast.Root([ast.c_for('i', ndofs*cdim, body).children[0]])
        funargs = [ast.Decl('double*', 'A'), ast.Decl('double**', 'B'), ast.Decl('double**', 'C')]
        fundecl = ast.FunDecl('void', name, funargs, body, ['static', 'inline'])

        # Track the AST for later fast retrieval
        self.asts[identifier] = fundecl

        return fundecl

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
        _trace.evaluate_all()
        if self.output:
            with timed_region('i/o'):
                if(u):
                    self.u_stream << u
                if(s):
                    # FIXME: Cannot currently write tensor valued fields to a VTU file.
                    # See https://github.com/firedrakeproject/firedrake/issues/538
                    # self.s_stream << s
                    pass

    def run(self, T):
        """ Run the elastic wave simulation until t = T.
        :param float T: The finish time of the simulation.
        :returns: The final solution fields for velocity and stress.
        """
        self.write(self.u1, self.s1)  # Write out the initial condition.

        if self.explicit:
            log("Generating inverse mass matrix")
            # Pre-assemble the inverse mass matrices, which should stay
            # constant throughout the simulation (assuming no mesh adaptivity).
            with timed_region('inverse mass matrix'):
                self.assemble_inverse_mass()
                if self.solver in ['parloop', 'fusion', 'tile']:
                    self.imass_velocity = self.matrix_to_dat(self.imass_velocity, self.U)
                    self.imass_stress = self.matrix_to_dat(self.imass_stress, self.S)

            # Set RHS as individual solver contexts
            ctx_uh1 = self.rhs_uh1
            ctx_stemp = self.rhs_stemp
            ctx_uh2 = self.rhs_uh2
            ctx_u1 = self.rhs_u1
            ctx_sh1 = self.rhs_sh1
            ctx_utemp = self.rhs_utemp
            ctx_sh2 = self.rhs_sh2
            ctx_s1 = self.rhs_s1
        else:
            self.imass_velocity = None
            self.imass_stress = None

            log("Creating ctx contexts")
            with timed_region('ctx setup'):
                ctx_uh1 = self.create_solver(self.form_uh1, self.uh1)
                ctx_stemp = self.create_solver(self.form_stemp, self.stemp)
                ctx_uh2 = self.create_solver(self.form_uh2, self.uh2)
                ctx_u1 = self.create_solver(self.form_u1, self.u1)
                ctx_sh1 = self.create_solver(self.form_sh1, self.sh1)
                ctx_utemp = self.create_solver(self.form_utemp, self.utemp)
                ctx_sh2 = self.create_solver(self.form_sh2, self.sh2)
                ctx_s1 = self.create_solver(self.form_s1, self.s1)

        # Set self.solve according to solver mode
        if self.solver == 'explicit':
            solve = self.solve_petsc
        elif self.solver in ['parloop', 'fusion', 'tile']:
            solve = self.solve_parloop
        else:
            solve = lambda ctx, imass, res: ctx.solve()

        with timed_region('timestepping'):
            t = self.dt
            while t <= T + 1e-12:
                log("t = %f" % t)

                with loop_chain("main1", tile_size=self.tile_size, num_unroll=self.num_unroll,
                                mode=self.tiling_mode, extra_halo=self.extra_halo,
                                partitioning='chunk'):
                    # In case the source is time-dependent, update the time 't' here.
                    if(self.source):
                        with timed_region('source term update'):
                            self.source_expression.t = t
                            self.source = self.source_expression

                    with timed_region('velocity solve'):
                        # Solve for the velocity vector field.
                        solve(ctx_uh1, self.imass_velocity, self.uh1)
                        solve(ctx_stemp, self.imass_stress, self.stemp)
                        solve(ctx_uh2, self.imass_velocity, self.uh2)
                        solve(ctx_u1, self.imass_velocity, self.u1)
                    self.u0.assign(self.u1)

                    with timed_region('stress solve'):
                        # Solve for the stress tensor field.
                        solve(ctx_sh1, self.imass_stress, self.sh1)
                        solve(ctx_utemp, self.imass_velocity, self.utemp)
                        solve(ctx_sh2, self.imass_stress, self.sh2)
                        solve(ctx_s1, self.imass_stress, self.s1)
                    self.s0.assign(self.s1)

                # Write out the new fields
                self.write(self.u1, self.s1)

                # Move onto next timestep
                t += self.dt

        return self.u1, self.s1
