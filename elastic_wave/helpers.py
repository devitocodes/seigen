from math import *
from pyop2 import op2, MPI


def log(s):
    """Internal logging method for parallel output to stdout

    :arg str s: The message to print.
    """
    if op2.MPI.comm.rank == 0:
        print s


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


def calculate_sdepth(num_solves, num_unroll, extra_halo):
    """Calculates the number of additional halo layers required for
    fusion and tiling runs through the following formula:

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
