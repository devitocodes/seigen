from math import *
from pyop2 import op2


def log(s):
    """Internal logging method for parallel output to stdout"""
    if op2.MPI.comm.rank == 0:
        print s


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


def cfl_dt(dx, Vp, courant_number):
    r""" Computes the maximum permitted value for the timestep math:`\delta t` with respect to the CFL condition.
    :param float dx: The characteristic element length.
    :param float Vp: The P-wave velocity.
    :param float courant_number: The desired Courant number
    :returns: The maximum permitted timestep, math:`\delta t`.
    :rtype: float
    """
    return (courant_number*dx)/Vp
