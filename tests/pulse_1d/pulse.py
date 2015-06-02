#!/usr/bin/env python

from firedrake import *
import os

# PETSc environment variables
try:
   if(os.environ["PETSC_OPTIONS"] == ""):
      os.environ["PETSC_OPTIONS"] = "-log_summary"
   else:
      os.environ["PETSC_OPTIONS"] = os.environ["PETSC_OPTIONS"] + " -log_summary"
except KeyError:
   # Environment variable does not exist, so let's set it now.
   os.environ["PETSC_OPTIONS"] = "-log_summary"
   
#parameters['form_compiler']['quadrature_degree'] = 4
#parameters["coffee"]["O2"] = False

Lx = 4.0
h = 1e-2
mesh = IntervalMesh(int(Lx/h), Lx)

S = FunctionSpace(mesh, "DG", 2)
U = FunctionSpace(mesh, "DG", 2)
dimension = 1

s = TrialFunction(S)
v = TestFunction(S)

u = TrialFunction(U)
w = TestFunction(U)

s0 = Function(S)
s1 = Function(S)

u0 = Function(U)
u1 = Function(U)

# Constants
density = 1.0
T = 2.0
dt = 0.0014
mu = 0.25
l = 0.5

n = FacetNormal(mesh)
absorption = Function(U).interpolate(Expression("x[0] <= 3.5 && x[0] >= 0.5 ? 0 : 0.1"))

# Weak forms
F_u = density*inner(w, (u - u0)/dt)*dx \
      + inner(grad(w)[0], s0)*dx - inner(avg(s0), jump(w, n[0]))*dS + inner(w, absorption*u0)*dx

F_s = inner(v, (s - s0)/dt)*dx \
      + (l + 2*mu)*inner(grad(v)[0], u1)*dx \
      - (l + 2*mu)*inner(jump(v, n[0]), avg(u1))*dS \
      - (l + 2*mu)*inner(v, u1*n[0])*ds

problem_u = LinearVariationalProblem(lhs(F_u), rhs(F_u), u1)
solver_u = LinearVariationalSolver(problem_u)

problem_s = LinearVariationalProblem(lhs(F_s), rhs(F_s), s1)
solver_s = LinearVariationalSolver(problem_s)

output_u = File("velocity.pvd")
output_s = File("stress.pvd")

# Initial conditions
uic = Expression(('exp(-50*pow((x[0]-1), 2))'))
u0.assign(Function(U).interpolate(uic))
sic = Expression(('-exp(-50*pow((x[0]-1), 2))'))
s0.assign(Function(S).interpolate(sic))

t = dt
while t <= T + 1e-12:
   print "t = %f" % t
   
   # Solve for the velocity vector
   solver_u.solve()
   u0.assign(u1)
   
   # Solve for the stress tensor
   solver_s.solve()
   s0.assign(s1)
   
   # Move onto next timestep
   t += dt
   
   #G = inner(w, u)*dx - inner(w, u1-uexact)*dx
   #solve(lhs(G) == rhs(G), temp)

   output_u << u1
   #output_s << s1

