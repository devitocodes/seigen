import os
import numpy
import pylab
import vtktools
      
Lx = 300.0
h = 2.5
Nx = int(Lx/h)
x_array = numpy.linspace(0, Lx, Nx)

T = 2.5
dt = 0.0025

t = []
uy_c1 = []
uy_c2 = []
uy_c3 = []
for i in range(0, int(T/dt), 5):
   t.append(dt*i)
   f = vtktools.vtu('velocity_%d.vtu' % i)
   f.GetFieldNames()
      
   value = vtktools.vtu.ProbeData(f, numpy.array([[45.0, 149.0, 0]]), 'function_27')
   uy_c1.append(-value[0][1])

   value = vtktools.vtu.ProbeData(f, numpy.array([[90.0, 149.0, 0]]), 'function_27')
   uy_c2.append(-value[0][1])
   
   value = vtktools.vtu.ProbeData(f, numpy.array([[140.0, 149.0, 0]]), 'function_27')
   uy_c3.append(-value[0][1])

pylab.figure(0)
pylab.plot(t, uy_c1, 'k-', label="Sensor C1")
pylab.legend()
pylab.xlabel(r"$t$ (s)")
pylab.ylabel(r"$y$-component of velocity (ms$^{-1}$)")
pylab.axis([0.0, 1.0, -5e-5, 3e-5])
pylab.grid("on")
pylab.savefig("uy_c1.pdf")

pylab.figure(1)
pylab.plot(t, uy_c2, 'k-', label="Sensor C1")
pylab.legend()
pylab.xlabel(r"$t$ (s)")
pylab.ylabel(r"$y$-component of velocity (ms$^{-1}$)")
pylab.axis([0.5, 1.5, -8e-6, 8e-6])
pylab.grid("on")
pylab.savefig("uy_c2.pdf")

pylab.figure(2)
pylab.plot(t, uy_c3, 'k-', label="Sensor C1")
pylab.legend()
pylab.xlabel(r"$t$ (s)")
pylab.ylabel(r"$y$-component of velocity (ms$^{-1}$)")
pylab.axis([1.0, 2.5, -6e-6, 6e-6])
pylab.grid("on")
pylab.savefig("uy_c3.pdf")
