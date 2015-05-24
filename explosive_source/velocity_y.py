import os
import numpy
import pylab
import vtktools
      
Lx = 300.0
Nx = 100
x_array = numpy.linspace(0, Lx, Nx)

T = 2.5
dt = 0.005

t = []
velocity_y = []
for i in range(0, int(T/dt), 5):
   t.append(dt*i)
   f = vtktools.vtu('velocity_%d.vtu' % i)
   f.GetFieldNames()
      
   value = vtktools.vtu.ProbeData(f, numpy.array([[45.0, 149.0, 0]]), 'function_27')
   velocity_y.append(-value[0][1])


print velocity_y

pylab.plot(t, velocity_y, 'k-', label="y-component")
pylab.legend()
pylab.xlabel(r"$t$ (s)")
pylab.ylabel(r"Velocity (ms$^{-1}$)")
pylab.axis([0.0, 1.0, -5e-5, 3e-5])
pylab.grid("on")
pylab.show()
