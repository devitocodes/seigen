import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import vtktools
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

Lx = 4.0
h = 1e-2
Nx = int(Lx/h)
x_array = numpy.linspace(0, Lx, Nx)
ux_array = numpy.zeros(len(x_array))

dt = 0.0014
times = [0.25, 1.0, 2.0]

for t in times:
   index = int(0.25/dt)
   f = vtktools.vtu('velocity_%d.vtu' % index)
   f.GetFieldNames()
   ux = []
   for i in range(len(x_array)):
      value = vtktools.vtu.ProbeData(f, numpy.array([[x_array[i], 0.5, 0]]), 'VelocityNew')
      ux_array[i] = value[0][0]

   fig = plt.figure(1)
   plt.plot(x_array, ux_array, 'k-')
   plt.legend()
   plt.xlabel(r"$x$ (m)")
   plt.ylabel(r"$x$-component of velocity (ms$^{-1}$)")
   plt.axis([0, 4, 0, 1.1])
   plt.grid("on")
   plt.savefig("ux_%f.pdf" % t)

