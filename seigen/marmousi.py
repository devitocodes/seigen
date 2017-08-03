from firedrake import *
import numpy

def create_marmousi_model(path):
   data = numpy.loadtxt(path).reshape((384, 122))
   
   class Marmousi(Expression):
      def eval(self, value, x):         
         i = numpy.floor(x[0]/24.0)
         j = numpy.floor(x[1]/24.0)
         value[0] = data[i][-j]
         
   m = Marmousi()
   return m
   
m = create_marmousi_model("data/marmhard.dat")

Lx = 9192
Ly = 2904
h = 24
mesh = RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)
fs = FunctionSpace(mesh, "DG", 1)
f = Function(fs).interpolate(m)
File("marmousi.pvd") << f

