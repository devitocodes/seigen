import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import vtktools
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
from math import exp

def V_p():
   return 1.0
 
def rho():
   return 1.0

def ux_ic(x):
   return exp(-50*(x-1)**2)

def sxx_ic(x):
   return -exp(-50*(x-1)**2)
   
def ux_exact(x, t):
   x_minus = x - V_p()*t
   x_plus = x + V_p()*t
   ux = 0.5*(ux_ic(x_minus) + ux_ic(x_plus) + (1.0/(rho()*V_p()))*(sxx_ic(x_plus) - sxx_ic(x_minus)))
   return ux
   
def sxx_exact(x, t):
   x_minus = x - V_p()*t
   x_plus = x + V_p()*t
   sxx = 0.5*(sxx_ic(x_minus) + sxx_ic(x_plus) + rho()*V_p()*(ux_ic(x_plus) - ux_ic(x_minus)))
   return sxx


Lx = 4.0
dx = 1e-2
Nx = int(Lx/dx)
x_array = numpy.linspace(0, Lx, Nx)
ux_array = numpy.zeros(len(x_array))
ux_error_array = numpy.zeros(len(x_array))

dt = 0.0014
times = [0.25, 1.0, 2.0]

for t in times:
   index = int(t/dt)
   f = vtktools.vtu('velocity_%d.vtu' % index)
   f.GetFieldNames()
   for i in range(len(x_array)):
      value = vtktools.vtu.ProbeData(f, numpy.array([[x_array[i], 0.5, 0]]), 'VelocityNew')
      ux_array[i] = value[0][0]      
      ux_error_array[i] = 10.0*(ux_exact(x_array[i], t) - ux_array[i])

   fig = plt.figure()
   plt.plot(x_array, ux_array, 'k-', label=r"$\Delta x = %.2f\ \mathrm{m}$" % dx)
   plt.plot(x_array, ux_error_array, 'r-', label=r"$10 \times \epsilon$")
   
   plt.legend()
   plt.xlabel(r"$x$ (m)")
   plt.ylabel(r"$x$-component of velocity (ms$^{-1}$)")
   plt.axis([0, 4, -0.1, 1.1])
   plt.grid("on")
   plt.savefig("ux_%f.pdf" % t)

