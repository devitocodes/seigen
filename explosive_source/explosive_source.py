#!/usr/bin/env python

from firedrake import *
#parameters["coffee"]["O2"] = False

mesh = Mesh("src/domain.msh")

S = FunctionSpace(mesh, "DG", 2)
U = FunctionSpace(mesh, "DG", 2)
dimension = 2

WS = MixedFunctionSpace([S, S, S, S])
WU = MixedFunctionSpace([U, U])

s = TrialFunctions(WS)
v = TestFunctions(WS)

u = TrialFunctions(WU)
w = TestFunctions(WU)

s0 = Function(WS, name="StressOld")
s1 = Function(WS, name="StressNew")

u0 = Function(WU, name="VelocityOld")
u1 = Function(WU, name="VelocityNew")

# Constants
density = 1.0
T = 2.5
dt = 0.0025
mu = 3600.0
l = 3599.3664

Vp = sqrt((l + 2*mu)/density) # P-wave velocity
Vs = sqrt(mu/density) # S-wave velocity
print Vp, Vs

n = FacetNormal(mesh)

a = 159.42
source_expression = Expression("x[0] >= 44.5 && x[0] <= 45.5 && x[1] >= 148.5 && x[1] <= 149.5 ? (-1.0 + 2*a*pow(t - 0.3, 2))*exp(-a*pow(t - 0.3, 2)) : 0.0", a=a, t=0)
HU = FunctionSpace(mesh, "DG", 4)
source = Function(HU)
source.interpolate(source_expression)

absorption = Function(U).interpolate(Expression("x[0] <= 20 || x[0] >= 280 || x[1] <= 20.0 ? 10 : 0"))

# Weak forms
F_u = density*inner(w[0], (u[0] - u0[0])/dt)*dx \
      + inner(grad(w[0])[0], s0[0])*dx - inner(avg(s0[0]), jump(w[0], n[0]))*dS + inner(w[0], absorption*u[0])*dx \
      + inner(grad(w[0])[1], s0[1])*dx - inner(avg(s0[1]), jump(w[0], n[1]))*dS

F_u += density*inner(w[1], (u[1] - u0[1])/dt)*dx \
       + inner(grad(w[1])[0], s0[2])*dx - inner(avg(s0[2]), jump(w[1], n[0]))*dS \
       + inner(grad(w[1])[1], s0[3])*dx - inner(avg(s0[3]), jump(w[1], n[1]))*dS + inner(w[1], absorption*u[1])*dx \

F_s = inner(v[0], (s[0] - s0[0])/dt)*dx \
      + (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
      - (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
      - (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
      + l*inner(grad(v[0])[1], u1[1])*dx \
      - l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
      - l*inner(v[0], u1[1]*n[1])*ds - inner(v[0], source)*dx

F_s += inner(v[1], (s[1] - s0[1])/dt)*dx \
       + mu*(inner(grad(v[1])[0], u1[1]))*dx \
       - mu*(inner(v[1], u1[1]*n[0]))*ds \
       - mu*(inner(jump(v[1], n[0]), avg(u1[1])))*dS \
       + mu*(inner(grad(v[1])[1], u1[0]))*dx \
       - mu*(inner(v[1], u1[0]*n[1]))*ds \
       - mu*(inner(jump(v[1], n[1]), avg(u1[0])))*dS

F_s += inner(v[2], (s[2] - s0[2])/dt)*dx \
       + mu*(inner(grad(v[2])[0], u1[1]))*dx \
       - mu*(inner(v[2], u1[1]*n[0]))*ds \
       - mu*(inner(jump(v[2], n[0]), avg(u1[1])))*dS \
       + mu*(inner(grad(v[2])[1], u1[0]))*dx \
       - mu*(inner(v[2], u1[0]*n[1]))*ds \
       - mu*(inner(jump(v[2], n[1]), avg(u1[0])))*dS

F_s += inner(v[3], (s[3] - s0[3])/dt)*dx \
       + l*inner(grad(v[3])[0], u1[0])*dx \
       - l*inner(v[3], u1[0]*n[0])*ds \
       - l*inner(jump(v[3], n[0]), avg(u1[0]))*dS \
       + (l + 2*mu)*inner(grad(v[3])[1], u1[1])*dx \
       - (l + 2*mu)*inner(jump(v[3], n[1]), avg(u1[1]))*dS \
       - (l + 2*mu)*inner(v[3], u1[1]*n[1])*ds - inner(v[3], source)*dx

problem_u = LinearVariationalProblem(lhs(F_u), rhs(F_u), u1)
solver_u = LinearVariationalSolver(problem_u)

problem_s = LinearVariationalProblem(lhs(F_s), rhs(F_s), s1)
solver_s = LinearVariationalSolver(problem_s)

output_u = File("velocity.pvd")
output_s = File("stress.pvd")

# Initial conditions
uic = Expression(('0.0', '0.0'))
u0.assign(Function(WU).interpolate(uic))
sic = Expression((('0','0'),
                  ('0','0')))
s0.assign(Function(WS).interpolate(sic))

output_u << u1
output_s << s1.split()[0]
   
t = dt
while t <= T + 1e-12:
   print "t = %f" % t
   source_expression.t = t
   source.interpolate(source_expression)
   
   # Solve for the velocity vector
   solver_u.solve()
   u0.assign(u1)
   
   # Solve for the stress tensor
   solver_s.solve()
   s0.assign(s1)

   output_u << u1
   output_s << s1.split()[0]
   
   # Move onto next timestep
   t += dt

