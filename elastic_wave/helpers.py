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
