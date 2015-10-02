from firedrake import *
from firedrake.petsc import PETSc

from pyop2.utils import cached_property
from pyop2.profiling import timed_region
from pyop2.base import _trace, Dat, DataSet
from pyop2.configuration import configuration
from pyop2.fusion import loop_chain

import coffee.base as ast

from math import *
import mpi4py
import numpy as np
from time import time

from utils.benchmarking import parser, output_time

# This is an explicit DG method: we invert the mass matrix and perform a matrix-vector multiplication to get the solution

class ElasticLF4(object):
    r""" An elastic wave equation solver, using the finite element method for spatial discretisation,
    and a fourth-order leap-frog time-stepping scheme. """

    def __init__(self, mesh, family, degree, dimension, output=True, tiling=None):
        r""" Initialise a new elastic wave simulation.

        :param mesh: The underlying computational mesh of vertices and edges.
        :param str family: Specify whether CG or DG should be used.
        :param int degree: Use polynomial basis functions of this degree.
        :param int dimension: The spatial dimension of the problem (1, 2 or 3).
        :param bool output: If True, output the solution fields to a file.
        :param dict tiling: Parameters driving tiling (tile size, unroll factor, mode, ...)
        :returns: None
        """
        with timed_region('function setup'):
            self.mesh = mesh
            self.dimension = dimension
            self.output = output

            self.S = TensorFunctionSpace(mesh, family, degree, name='S')
            self.U = VectorFunctionSpace(mesh, family, degree, name='U')
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

            # Tiling options
            self.tiling_size = tiling['tile_size']
            self.tiling_uf = tiling['num_unroll']
            self.tiling_mode = tiling['mode']

        if self.output:
            with timed_region('i/o'):
                # File output streams
                self.u_stream = File("/data/output/velocity.pvd")
                self.s_stream = File("/data/output/stress.pvd")

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

    def copy_massmatrix_into_dat(self):
        # Copy the velocity mass matrix into a Dat
        arity = sum(self.U._dofs_per_entity)*self.U.cdim
        self.velocity_mass_asdat = Dat(DataSet(self.mesh.cell_set, arity*arity), dtype='double')
        for i in range(self.mesh.num_cells()):
            indices = PETSc.IS().createGeneral(i*arity + np.arange(arity, dtype=np.int32))
            val = self.imass_velocity.handle.getSubMatrix(indices, indices)
            self.velocity_mass_asdat.data[i] = val[:, :].flatten()

        # Copy the stress mass matrix into a Dat
        arity = sum(self.S._dofs_per_entity)*self.S.cdim
        self.stress_mass_asdat = Dat(DataSet(self.mesh.cell_set, arity*arity), dtype='double')
        for i in range(self.mesh.num_cells()):
            indices = PETSc.IS().createGeneral(i*arity + np.arange(arity, dtype=np.int32))
            val = self.imass_stress.handle.getSubMatrix(indices, indices)
            self.stress_mass_asdat.data[i] = val[:, :].flatten()

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
        # Note that we have multiplied through by dt here.
        F = self.density*inner(self.w, self.u)*dx - self.density*inner(self.w, self.u0)*dx - self.dt*inner(self.w, self.uh1)*dx - ((self.dt**3)/24.0)*inner(self.w, self.uh2)*dx
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
        # Note that we have multiplied through by dt here.
        F = inner(self.v, self.s)*dx - inner(self.v, self.s0)*dx - self.dt*inner(self.v, self.sh1)*dx - ((self.dt**3)/24.0)*inner(self.v, self.sh2)*dx
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
        g = - l*(v[i, j]*I[i, j]).dx(k)*u1[k]*dx + l*(jump(v[i, j], n[k])*I[i, j]*avg(u1[k]))*dS + l*(v[i, j]*I[i, j]*u1[k]*n[k])*ds - mu*inner(div(v), u1)*dx + mu*inner(avg(u1), jump(v, n))*dS - mu*inner(div(v.T), u1)*dx + mu*inner(avg(u1), jump(v.T, n))*dS + mu*inner(u1, dot(v, n))*ds + mu*inner(u1, dot(v.T, n))*ds
        if(source):
            g += inner(v, source)*dx
        return g

    def solve(self, rhs, matrix_asdat, result):
        F_a = assemble(rhs)

        # The number of dofs on each element is /ndofs*cdim/. If using vector-valued function spaces,
        # the total number of values is /ndofs*cdim/
        F_a_fs = F_a.function_space()
        ndofs = sum(F_a_fs._dofs_per_entity)
        cdim = F_a_fs.cdim

        # The AST of the mat-vel mul
        body = ast.Incr(ast.Symbol('C', ('i/%d' % cdim, 'i%%%d' % cdim)),
                        ast.Prod(ast.Symbol('A', ('i',), (('N', 'j*%d + k' % cdim),)),
                                 ast.Symbol('B', ('j', 'k'))))
        body = ast.c_for('k', cdim, body).children[0]
        body = [ast.Assign(ast.Symbol('C', ('i/%d' % cdim, 'i%%%d' % cdim)), '0.0'),
                ast.c_for('j', ndofs, body).children[0]]
        body = ast.Root([ast.Decl('int', 'N', ast.Prod(ndofs, cdim), ['const']),
                         ast.c_for('i', 'N', body).children[0]])
        funargs = [ast.Decl('double*', 'A'), ast.Decl('double**', 'B'), ast.Decl('double**', 'C')]
        name = 'mat_vec_mul_kernel_%s' % F_a.function_space().name
        fundecl = ast.FunDecl('void', name, funargs, body, ['static', 'inline'])

        # Create the par loop (automatically added to the trace of loops to be executed)
        kernel = op2.Kernel(fundecl, name)
        op2.par_loop(kernel, self.mesh.cell_set,
                     matrix_asdat(op2.READ),
                     F_a.dat(op2.READ, F_a.cell_node_map()),
                     result.dat(op2.WRITE, result.cell_node_map()))

    def write(self, u=None, s=None):
        r""" Write the velocity and/or stress fields to file.
        :param firedrake.Function u: The velocity field.
        :param firedrake.Function s: The stress field.
        :returns: None
        """
        if self.output:
            _trace.evaluate_all()
            with timed_region('i/o'):
                if(u):
                    self.u_stream << u
                if(s):
                    pass  # FIXME: Cannot currently write tensor valued fields to a VTU file. See https://github.com/firedrakeproject/firedrake/issues/538
                    #self.s_stream << s

    def run(self, T):
        """ Run the elastic wave simulation until t = T.
        :param float T: The finish time of the simulation.
        :returns: The final solution fields for velocity and stress.
        """
        self.write(self.u1, self.s1)  # Write out the initial condition.

        print "Generating inverse mass matrix"
        # Pre-assemble the inverse mass matrices, which should stay
        # constant throughout the simulation (assuming no mesh adaptivity).
        with timed_region('inverse mass matrix'):
            self.assemble_inverse_mass()
            self.copy_massmatrix_into_dat()

        with timed_region('timestepping'):
            t = self.dt
            while t <= T + 1e-12:
                print "t = %f" % t
                with loop_chain("main1", tile_size=self.tiling_size, num_unroll=self.tiling_uf, mode=self.tiling_mode):
                    # In case the source is time-dependent, update the time 't' here.
                    if(self.source):
                        with timed_region('source term update'):
                            self.source_expression.t = t
                            self.source = self.source_expression

                    # Solve for the velocity vector field.
                    self.solve(self.rhs_uh1, self.velocity_mass_asdat, self.uh1)
                    self.solve(self.rhs_stemp, self.stress_mass_asdat, self.stemp)
                    self.solve(self.rhs_uh2, self.velocity_mass_asdat, self.uh2)
                    self.solve(self.rhs_u1, self.velocity_mass_asdat, self.u1)
                    self.u0.assign(self.u1)

                    # Solve for the stress tensor field.
                    self.solve(self.rhs_sh1, self.stress_mass_asdat, self.sh1)
                    self.solve(self.rhs_utemp, self.velocity_mass_asdat, self.utemp)
                    self.solve(self.rhs_sh2, self.stress_mass_asdat, self.sh2)
                    self.solve(self.rhs_s1, self.stress_mass_asdat, self.s1)
                    self.s0.assign(self.s1)

                # Write out the new fields
                self.write(self.u1, self.s1)

                # Move onto next timestep
                t += self.dt

        return self.u1, self.s1


# Helper stuff

def Vp(mu, l, density):
    r""" Calculate the P-wave velocity, given by

     .. math:: \sqrt{\frac{(\lambda + 2\mu)}{\rho}}

    where :math:`\rho` is the density, and :math:`\lambda` and :math:`\mu` are the first and second Lame parameters, respectively.

    :param mu: The second Lame parameter.
    :param l: The first Lame parameter.
    :param density: The density.
    :returns: The P-wave velocity.
    :rtype: float
    """
    return sqrt((l + 2*mu)/density)


def Vs(mu, density):
    r""" Calculate the S-wave velocity, given by

     .. math:: \sqrt{\frac{\mu}{\rho}}

    where :math:`\rho` is the density, and :math:`\lambda` and :math:`\mu` is the second Lame parameter.

    :param mu: The second Lame parameter.
    :param density: The density.
    :returns: The P-wave velocity.
    :rtype: float
    """
    return sqrt(mu/density)


# Test cases

class ExplosiveSourceLF4():

    def generate_mesh(self, Lx=300.0, Ly=150.0, h=2.5):
        mesh = RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)
        mesh.init(s_depth=1)
        # This is only to print out info related to tiling
        slope(mesh, debug=True)
        return mesh

    def explosive_source_lf4(self, T=2.5, Lx=300.0, Ly=150.0, h=2.5, output=True, tiling=None):

        with timed_region('mesh generation'):
            mesh = self.generate_mesh()
            self.elastic = ElasticLF4(mesh, "DG", 2, dimension=2, output=output, tiling=tiling)

        # Tiling info
        tile_size = tiling['tile_size']

        # Constants
        self.elastic.density = 1.0
        self.elastic.dt = 0.001
        self.elastic.mu = 3600.0
        self.elastic.l = 3599.3664

        print "P-wave velocity: %f" % Vp(self.elastic.mu, self.elastic.l, self.elastic.density)
        print "S-wave velocity: %f" % Vs(self.elastic.mu, self.elastic.density)

        # Source
        a = 159.42
        self.elastic.source_expression = Expression((("x[0] >= 44.5 && x[0] <= 45.5 && x[1] >= 148.5 && x[1] <= 149.5 ? (-1.0 + 2*a*pow(t - 0.3, 2))*exp(-a*pow(t - 0.3, 2)) : 0.0", "0.0"),
                                                     ("0.0", "x[0] >= 44.5 && x[0] <= 45.5 && x[1] >= 148.5 && x[1] <= 149.5 ? (-1.0 + 2*a*pow(t - 0.3, 2))*exp(-a*pow(t - 0.3, 2)) : 0.0")), a=a, t=0)
        self.elastic.source_function = Function(self.elastic.S)
        self.elastic.source = self.elastic.source_expression

        # Absorption
        F = FunctionSpace(mesh, "DG", 4, name='F')
        self.elastic.absorption_function = Function(F)
        self.elastic.absorption = Expression("x[0] <= 20 || x[0] >= 280 || x[1] <= 20.0 ? 1000 : 0")

        # Initial conditions
        uic = Expression(('0.0', '0.0'))
        self.elastic.u0.assign(Function(self.elastic.U).interpolate(uic))
        sic = Expression((('0', '0'), ('0', '0')))
        self.elastic.s0.assign(Function(self.elastic.S).interpolate(sic))

        # Run the simulation
        start = time()
        self.elastic.run(T)
        end = time()

        # Print runtime summary
        output_time(start, end, tofile=True, fs=self.elastic.U, tile_size=tile_size)


if __name__ == '__main__':
    op2.init(log_level='WARNING')
    from ffc.log import set_level
    set_level('ERROR')

    # Remove trace bound to avoid running inspections over and over
    configuration['lazy_max_trace_length'] = 0
    # Switch on PyOP2 profiling
    configuration['profiling'] = True

    # Parse the input
    args = parser(profile=False)
    tiling = {
        'num_unroll': args.num_unroll,
        'tile_size': args.tile_size,
        'mode': args.fusion_mode
    }
    profile = args.profile

    if not profile:
        ExplosiveSourceLF4().explosive_source_lf4(T=2.5, tiling=tiling)
    else:
        import cProfile
        cProfile.run('ExplosiveSourceLF4().explosive_source_lf4(T=0.1, tiling=tiling)',
                     'log.cprofile')
