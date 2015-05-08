#!/usr/bin/env python

from pyop2 import *
from firedrake import *

op2.init(lazy_evaluation=False)
parameters['form_compiler']['quadrature_degree'] = 4
parameters["coffee"]["O2"] = False

mesh = UnitSquareMesh(10, 10)

S = TensorFunctionSpace(mesh, "CG", 1)
U = VectorFunctionSpace(mesh, "CG", 1)

v = TestFunction(S)
w = TestFunction(U)
s = TrialFunction(S)
u = TrialFunction(U)
s_old = Function(S)
u_old = Function(U)


# Constants
density = 3500.0 #kgm**-3
T = 10.0
dt = 0.2

# Weak forms
F = inner(v, (s - s_old)/dt)*dx


t = 0.0
while t < T:
   
   # Solve for the stress tensor
   
   # Solve for the velocity vector
   
   # Move onto next timestep
   t += dt
