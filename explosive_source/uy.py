import os
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import vtktools
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

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
      
   value = vtktools.vtu.ProbeData(f, numpy.array([[45.0, 149.0, 0]]), 'VelocityNew')
   uy_c1.append(-value[0][1])

   value = vtktools.vtu.ProbeData(f, numpy.array([[90.0, 149.0, 0]]), 'VelocityNew')
   uy_c2.append(-value[0][1])
   
   value = vtktools.vtu.ProbeData(f, numpy.array([[140.0, 149.0, 0]]), 'VelocityNew')
   uy_c3.append(-value[0][1])

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(t, uy_c1, 'k-', label="Sensor C1")
ax.legend()
#plt.xlabel(r"$t$ (s)")
#plt.ylabel(r"$y$-component of velocity (ms$^{-1}$)")
#plt.axis([0, 1, -5e-5, 3e-5])
#plt.grid("on")
plt.savefig("uy_c1.pdf")


fig = plt.figure(1)
fig.clear()
ax = fig.add_subplot(111)
ax.plot(t, uy_c2, 'k-', label="Sensor C2")
ax.legend()
#plt.xlabel(r"$t$ (s)")
#plt.ylabel(r"$y$-component of velocity (ms$^{-1}$)")
#plt.axis([0.5, 1.5, -8e-6, 8e-6])
#plt.grid("on")
plt.savefig("uy_c2.pdf")

f =plt.figure(1)
f.clear()
plt.plot(t, uy_c3, 'k-', label="Sensor C3")
plt.legend()
plt.xlabel(r"$t$ (s)")
plt.ylabel(r"$y$-component of velocity (ms$^{-1}$)")
plt.axis([1.0, 2.5, -8e-6, 8e-6])
plt.grid("on")
plt.savefig("uy_c3.pdf")
