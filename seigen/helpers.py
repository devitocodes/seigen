from math import *
from pyop2 import op2
from pyop2.mpi import COMM_WORLD as comm


def log(s):
    """Internal logging method for parallel output to stdout

    :arg str s: The message to print.
    """
    if comm.rank == 0:
        print(s)


def Vp(mu, l, density):
    r""" Calculate the P-wave velocity, given by

     .. math:: \sqrt{\frac{(\lambda + 2\mu)}{\rho}}

    where :math:`\rho` is the density, and :math:`\lambda` and :math:`\mu` are the first and second Lame parameters, respectively.

    :param mu: The second Lame parameter.
    :param l: The first Lame parameter.
    :param density: The density :math:`\rho`.
    :returns: The P-wave velocity.
    :rtype: float
    """
    return sqrt((l + 2*mu)/density)


def Vs(mu, density):
    r""" Calculate the S-wave velocity, given by

     .. math:: \sqrt{\frac{\mu}{\rho}}

    where :math:`\rho` is the density, and :math:`\lambda` and :math:`\mu` is the second Lame parameter.

    :param mu: The second Lame parameter.
    :param density: The density :math:`\rho`.
    :returns: The S-wave velocity.
    :rtype: float
    """
    return sqrt(mu/density)


def cfl_dt(dx, Vp, courant_number):
    r""" Computes the maximum permitted value for the timestep math:`\delta t` with respect to the CFL condition.
    :param float dx: The characteristic element length.
    :param float Vp: The P-wave velocity.
    :param float courant_number: The desired Courant number
    :returns: The maximum permitted timestep, math:`\delta t`.
    :rtype: float
    """
    return (courant_number*dx)/Vp


def get_dofs(mesh, p):
    r""" Compute the total number of degrees of freedom for the stress and velocity fields when running over MPI.

    :param mesh: Any Firedrake-compatible mesh.
    :param p: The polynomial order of the function spaces.
    """
    S = TensorFunctionSpace(mesh, 'DG', p, name='S')
    U = VectorFunctionSpace(mesh, 'DG', p, name='U')
    S_tot_dofs = op2.MPI.comm.allreduce(S.dof_count, op=mpi4py.MPI.SUM)
    U_tot_dofs = op2.MPI.comm.allreduce(U.dof_count, op=mpi4py.MPI.SUM)
    return S_tot_dofs, U_tot_dofs
