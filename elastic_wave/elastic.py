#!/usr/bin/env python

from pyop2 import *
from pyop2.profiling import timed_region
from pyop2.base import _trace
from firedrake import *
from firedrake.petsc import PETSc
from elastic_wave.helpers import log
import mpi4py
from abc import ABCMeta, abstractmethod
import numpy as np
import coffee.base as ast
from contextlib import contextmanager


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
    def create(mesh, family, degree, dimension, solver="explicit", output=True):
        r""" Create an elastic wave equation solver for the given mesh
        according to specified spatial discretisation details and
        solver methods.

        :param mesh: The underlying computational mesh of vertices and edges.
        :param str family: Specify whether CG or DG should be used.
        :param int degree: Use polynomial basis functions of this degree.
        :param str solver: Solver mode, recognised values are:
            'implicit': Use PETSc KSP solver to solve for the solution fields.
            'explicit': Explicitly invert the mass matrix and perform a
                        matrix-vector multiplication using PETSc MatMult.
            'parloop': Explicitly invert the mass matrix and perform a
                       matrix-vector multiplication using a PyOP2 Parloop.
            'fusion': Experimental mode that enables PyOP2 kernel fusion
                      to fuse cell and facet loops.
            'tiling': Experimental mode that activates loop tiling via
                      PyOP2 and SLOPE.
        :param int dimension: The spatial dimension of the problem (1, 2 or 3).
        :param bool output: If True, output the solution fields to a file.
        :returns: None
        """
        if solver == "implicit":
            return ImplicitElasticLF4(mesh, family, degree, dimension, output=output)
        elif solver == "explicit":
            return ExplicitElasticLF4(mesh, family, degree, dimension, output=output)
        elif solver == "parloop":
            return TilingElasticLF4(mesh, family, degree, dimension,
                                    output=output, tiling_mode=None)
        elif solver == 'fusion':
            return TilingElasticLF4(mesh, family, degree, dimension,
                                    output=output, tiling_mode="hard")
        elif solver == 'tiling':
            return TilingElasticLF4(mesh, family, degree, dimension,
                                    output=output, tiling_mode="tile")
        else:
            raise ValueError("Unknown solver mode. Must be one of: implicit, explicit, parloop")

    def __init__(self, mesh, family, degree, dimension, output=True):
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

            # Inverse mass matrices for explicit methods
            self.invmass_velocity = None
            self.invmass_stress = None

        # Initialise internal profiling
        parameters["seigen"]['trace'] = {}
        parameters["seigen"]['initial'] = True

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

    @property
    def loop_context(self):
        r""" Empty context manager that is used as a placeholder
        instead of the PyOP2 context managers required for advanced
        fusion/tiling modes."""
        @contextmanager
        def empty_loop_context():
            yield
        return empty_loop_context

    def run(self, T):
        """ Run the elastic wave simulation until t = T.
        :param float T: The finish time of the simulation.
        :returns: The final solution fields for velocity and stress.
        """
        # Write out the initial condition.
        self.write(self.u1, self.s1)

        # Call solver-specific setup
        with PETSc.Log.Stage("Setup"):
            self.setup()
            _trace.evaluate_all()

        with timed_region('timestepping'):
            t = self.dt
            while t <= T + 1e-12:
                log("t = %f" % t)

                with self.loop_context():
                    # In case the source is time-dependent, update the time 't' here.
                    if(self.source):
                        with timed_region('source term update'):
                            self.source_expression.t = t
                            self.source = self.source_expression

                    # Solve for the velocity vector field.
                    with timed_region('velocity solve'):
                        self.solve(self.ctx_uh1, self.invmass_velocity, self.uh1, stage='uh1')
                        self.solve(self.ctx_stemp, self.invmass_stress, self.stemp, stage='stemp')
                        self.solve(self.ctx_uh2, self.invmass_velocity, self.uh2, stage='uh2')
                        self.solve(self.ctx_u1, self.invmass_velocity, self.u1, stage='u1')
                        self.u0.assign(self.u1)
                        _trace.evaluate_all()

                    # Solve for the stress tensor field.
                    with timed_region('stress solve'):
                        self.solve(self.ctx_sh1, self.invmass_stress, self.sh1, stage='sh1')
                        self.solve(self.ctx_utemp, self.invmass_velocity, self.utemp, stage='utemp')
                        self.solve(self.ctx_sh2, self.invmass_stress, self.sh2, stage='sh2')
                        self.solve(self.ctx_s1, self.invmass_stress, self.s1, stage='s1')
                        self.s0.assign(self.s1)
                        _trace.evaluate_all()

                # Write out the new fields
                self.write(self.u1, self.s1)

                # Move onto next timestep
                t += self.dt

                parameters['seigen']['initial'] = False

        if parameters['seigen']['profiling']:
            parameters['seigen']['profiling'] = {}
            for stage, trace in parameters['seigen']['trace'].items():
                parameters['seigen']['profiling'][stage] = {}
                for pl in trace:
                    name = pl.kernel.name
                    if name == "copy":
                        continue
                    parameters['seigen']['profiling'][stage][name] = {}
                    mem_b = 0
                    for arg in pl.args:
                        mem = arg.data.cdim * arg.data.dtype.itemsize
                        if arg.map:
                            mem *= arg.map.arity
                        mem_b += mem * 2 if arg.access._mode == "INC" else mem
                    parameters['seigen']['profiling'][stage][name]['bytes'] = mem_b
                    parameters['seigen']['profiling'][stage][name]['flops'] = pl.kernel.num_flops
                    parameters['seigen']['profiling'][stage][name]['ai'] = float(pl.kernel.num_flops) / mem_b
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

    def solve(self, solver, *args, **kwargs):
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

    def solve(self, rhs, matrix, result, stage="Explicit Solve"):
        r""" Solve by assembling into RHS vector and applying inverse mass matrix.
        :param rhs: The RHS vector that the inverse mass matrix will be multiplied with.
        :param matrix: The inverse mass matrix.
        :param firedrake.Function result: The solution field.
        :returns: None"""
        with PETSc.Log.Stage(stage):
            F_a = assemble(rhs)
            if parameters['seigen']['initial']:
                # Store compute kernels in the PyOP2._trace
                parameters['seigen']['trace'][stage] = _trace._trace
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


class TilingElasticLF4(ExplicitElasticLF4):
    r""" Experimental elastic equation solver that uses explicit
    solves for individual UFL forms and facilitates PyOP2-level loop
    fusion and loop tiling via SLOPE. These advanced solver modes all
    require a PyOP2 kernel to perform the multiplication of assembled
    RHSs and the diagonal block entries of the inverse mass matrices,
    which in turn allows us to fuse these into the main loops via
    tiling.
    """

    # Fusion/tiling-specific constants
    loop_chain_length = 28
    num_solves = 8
    tile_size = 1000
    extra_halo = 0

    def __init__(self, mesh, *args, **kwargs):
        r""" Tiling and loop fusion require increased halos (s-depth),
        which is currently done by explicitly calling
        mesh.init(sdepth)."""
        self.tiling_mode = kwargs.pop("tiling_mode", None)
        self.num_unroll = 0 if self.tiling_mode is None else 1
        if self.tiling_mode is not None:
            s_depth = self.calculate_sdepth(self.num_solves,
                                            self.num_unroll,
                                            self.extra_halo)
            mesh.topology.init(s_depth=s_depth)
        if self.tiling_mode == 'tile':
            slope(mesh, debug=True)
        super(TilingElasticLF4, self).__init__(mesh, *args, **kwargs)

        # AST cache
        self.asts = {}

    def calculate_sdepth(self, num_solves, num_unroll, extra_halo):
        r""" The sdepth for large halo regions is calculated as:

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

    def matrix_to_dat(self, massmatrix, functionspace):
        r""" Copy the clock diagonal entries of the velocity mass
        matrix into a pyop2.Dat"""
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

    def setup(self, *args, **kwargs):
        r""" After creating inverse mass matrices we extract diagonal
        block entries into a pyop2.Dat."""
        super(TilingElasticLF4, self).setup(*args, **kwargs)
        # Convert inverse mass matrices to PyOP2 Dats
        self.invmass_velocity = self.matrix_to_dat(self.invmass_velocity, self.U)
        self.invmass_stress = self.matrix_to_dat(self.invmass_stress, self.S)

    def ast_matmul(self, F_a):
        """Generate an AST for a PyOP2 kernel performing a matrix-vector multiplication.

        :param F_a: Assembled firedrake.Function object for the RHS"""

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

    def solve(self, rhs, matrix, result):
        r""" Solve by assembling RHS and applying inverse mass matrix using a PyOP2 Parloop.
        :param rhs: The RHS vector that the inverse mass matrix will be multiplied with.
        :param matrix: The inverse mass matrix.
        :param firedrake.Function result: The solution field.
        :returns: None"""
        F_a = assemble(rhs)
        ast_matmul = self.ast_matmul(F_a)

        # Create the par loop (automatically added to the trace of loops to be executed)
        kernel = op2.Kernel(ast_matmul, ast_matmul.name)
        op2.par_loop(kernel, self.mesh.cell_set, matrix(op2.READ),
                     F_a.dat(op2.READ, F_a.cell_node_map()),
                     result.dat(op2.WRITE, result.cell_node_map()))

    @property
    def loop_context(self):
        r""" Inject pyop2.loop_chain context to facilitate fusion and tiling across kernels."""
        from pyop2.fusion import loop_chain
        @contextmanager
        def tiling_loop_context():
            with loop_chain("main1", tile_size=self.tile_size, num_unroll=self.num_unroll,
                            mode=self.tiling_mode, extra_halo=self.extra_halo, partitioning='chunk'):
                yield
        return tiling_loop_context
