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
   
parameters['form_compiler']['quadrature_degree'] = 4
parameters["coffee"]["O2"] = False

mesh = UnitSquareMesh(20, 20)

S = FunctionSpace(mesh, "DG", 1)
U = FunctionSpace(mesh, "DG", 1)
dimension = 2

WS = MixedFunctionSpace([S, S, S, S])
WU = MixedFunctionSpace([U, U])

s = TrialFunctions(WS)
v = TestFunctions(WS)

u = TrialFunctions(WU)
w = TestFunctions(WU)

s0 = Function(WS)
s1 = Function(WS)

u0 = Function(WU)
u1 = Function(WU)

# Constants
density = 1.0
T = 5.0
dt = 0.01
mu = 0.25
l = 0.5

Vp = sqrt((l + 2*mu)/density) # P-wave velocity
Vs = sqrt(mu/density) # S-wave velocity

n = FacetNormal(mesh)

# Weak forms
#F_u = density*inner(w, (u - u0)/dt)*dx + inner(grad(w), s0)*dx #- inner(dot(jump(w), avg(s0)), n('+'))*dS - inner(dot(jump(w), avg(s0)), n('-'))*dS
#F_s = inner(v, (s - s0)/dt)*dx - inner(v, l*(div(u1))*Identity(dimension))*dx - inner(v, mu*(grad(u1) + grad(u1).T))*dx # - inner(dot(jump(v), avg(u1)), n('+'))*dS - inner(dot(jump(v), avg(u1)), n('-'))*dS

F_u = density*inner(w[0], (u[0] - u0[0])/dt)*dx \
      + inner(grad(w[0])[0], s0[0])*dx - inner(avg(s0[0]), jump(w[0], n[0]))*dS \
      + inner(grad(w[0])[1], s0[1])*dx - inner(avg(s0[1]), jump(w[0], n[1]))*dS

F_u += density*inner(w[1], (u[1] - u0[1])/dt)*dx \
       + inner(grad(w[1])[0], s0[2])*dx - inner(avg(s0[2]), jump(w[1], n[0]))*dS \
       + inner(grad(w[1])[1], s0[3])*dx - inner(avg(s0[3]), jump(w[1], n[1]))*dS \

F_s = inner(v[0], (s[0] - s0[0])/dt)*dx \
      + (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
      - (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
      - (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
      + l*inner(grad(v[0])[1], u1[1])*dx \
      - l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
      - l*inner(v[0], u1[1]*n[1])*ds

F_s += inner(v[1], (s[1] - s0[1])/dt)*dx \
       + mu*(inner(grad(v[1])[0], u1[1]))*dx \
       - mu*(inner(v[1], u1[1]*n[0]))*ds \
       - mu*(inner(jump(v[1], n[0]), avg(u1[1])))*dS \
       + mu*(inner(grad(v[1])[1], u1[0]))*dx \
       - mu*(inner(v[1], u1[0]*n[1]))*ds \
       - mu*(inner(jump(v[1], n[1]), avg(u1[0])))*dS \

F_s += inner(v[2], (s[2] - s0[2])/dt)*dx \
       + mu*(inner(grad(v[2])[0], u1[1]))*dx \
       - mu*(inner(v[2], u1[1]*n[0]))*ds \
       - mu*(inner(jump(v[2], n[0]), avg(u1[1])))*dS \
       + mu*(inner(grad(v[2])[1], u1[0]))*dx \
       - mu*(inner(v[2], u1[0]*n[1]))*ds \
       - mu*(inner(jump(v[2], n[1]), avg(u1[0])))*dS \

F_s += inner(v[3], (s[3] - s0[3])/dt)*dx \
       + l*inner(grad(v[3])[0], u1[0])*dx \
       - l*inner(v[3], u1[0]*n[0])*ds \
       - l*inner(jump(v[3], n[0]), avg(u1[0]))*dS \
       + (l + 2*mu)*inner(grad(v[3])[1], u1[1])*dx \
       - (l + 2*mu)*inner(jump(v[3], n[1]), avg(u1[1]))*dS \
       - (l + 2*mu)*inner(v[3], u1[1]*n[1])*ds \

problem_u = LinearVariationalProblem(lhs(F_u), rhs(F_u), u1)
solver_u = LinearVariationalSolver(problem_u)

problem_s = LinearVariationalProblem(lhs(F_s), rhs(F_s), s1)
solver_s = LinearVariationalSolver(problem_s)

output_u = File("velocity.pvd")
output_s = File("stress.pvd")

# Initial conditions
a = sqrt(2)*pi*Vs
b = 2*pi*mu
uic = Expression(('a*cos(pi*x[0])*sin(pi*x[1])*cos(a*t)','-a*sin(pi*x[0])*cos(pi*x[1])*cos(a*t)'), a=a, t=0)
u0.assign(Function(WU).interpolate(uic))
sic = Expression((('-b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)','0'),
                   ('0','b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)')), a=a, b=b, t=dt/2.0)
s0.assign(Function(WS).interpolate(sic))


uexact = Function(WU).interpolate(Expression(('a*cos(pi*x[0])*sin(pi*x[1])*cos(a*t)','-a*sin(pi*x[0])*cos(pi*x[1])*cos(a*t)'), a=a, t=5))
sexact = Function(WS).interpolate(Expression((('-b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)','0'),
                   ('0','b*sin(pi*x[0])*sin(pi*x[1])*sin(a*t)')), a=a, b=b, t=5+dt/2.0))

t = dt
temp = Function(U)
temp_test = TestFunction(U)
temp_trial = TrialFunction(U)
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

   output_u << u1
   output_s << s1.split()[0]

   #print "|u-uexact| = %f" % errornorm(u1.split()[0], uexact.split()[0])
   #print "|s-sexact| = %f" % errornorm(s1, sexact)
   
G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(u1.split()[0]-uexact.split()[0]))*dx
solve(lhs(G) == rhs(G), temp)
print norm(temp)
File("test.pvd") << temp
