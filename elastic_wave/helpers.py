from math import *

def Vp(mu, l, density):
   """ Return the P-wave velocity. """
   return sqrt((l + 2*mu)/density)
   
def Vs(mu, density):
   """ Return the S-wave velocity. """
   return sqrt(mu/density)
