from math import *

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

def l(fs, Vp, Vs, density):
   r""" Calculate the second Lame parameter from the velocity models. 
   
   :param Vp: The P-wave velocity.
   :param Vs: The S-wave velocity.
   :param density: The density.
   :returns: The second Lame parameter $\lambda$.
   """
   return project(density*(Vp**2 - 2*(Vs**2)), fs, name="LameLambda")
   
def mu(fs, Vs, density):
   r""" Calculate the first Lame parameter from the velocity models. 
   
   :param Vs: The S-wave velocity.
   :param density: The density.
   :returns: The first Lame parameter $\mu$.
   """
   return project(density*(Vs**2), fs, name="LameMu")

