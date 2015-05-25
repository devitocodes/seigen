#!/usr/bin/env python

from firedrake import *
#parameters["coffee"]["O2"] = False

Lx = 4.0
Ly = 1.0
h = 1e-2
mesh = RectangleMesh(int(Lx/h), int(Ly/h), Lx, Ly)

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
T = 2.0
dt = 0.0014
mu = 0.25
l = 0.5

Vp = sqrt((l + 2*mu)/density) # P-wave velocity
Vs = sqrt(mu/density) # S-wave velocity

n = FacetNormal(mesh)
absorption = Function(U).interpolate(Expression("x[0] <= 3.5 && x[0] >= 0.5 && x[1] >= 0.0 && x[1] <= 1.0 ? 0 : 1.0"))

# Weak forms
F_u = density*inner(w[0], (u[0] - u0[0])/dt)*dx \
      + inner(grad(w[0])[0], s0[0])*dx - inner(avg(s0[0]), jump(w[0], n[0]))*dS - inner(s0[0], w[0]*n[0])*ds - inner(w[0], absorption*u[0])*dx \
      + inner(grad(w[0])[1], s0[1])*dx - inner(avg(s0[1]), jump(w[0], n[1]))*dS - inner(s0[1], w[0]*n[1])*ds

#F_u += density*inner(w[1], (u[1] - u0[1])/dt)*dx \
#       + inner(grad(w[1])[0], s0[2])*dx - inner(avg(s0[2]), jump(w[1], n[0]))*dS - inner(s0[2], w[1]*n[0])*ds \
#       + inner(grad(w[1])[1], s0[3])*dx - inner(avg(s0[3]), jump(w[1], n[1]))*dS - inner(s0[3], w[1]*n[1])*ds - inner(w[1], absorption*u[1])*dx

F_s = inner(v[0], (s[0] - s0[0])/dt)*dx \
      + (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
      - (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
      - (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
      + l*inner(grad(v[0])[1], u1[1])*dx \
      - l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
      - l*inner(v[0], u1[1]*n[1])*ds # Remove the exterior facet integrals with u1[1] to weakly apply a no-normal flow

#F_s += inner(v[1], (s[1] - s0[1])/dt)*dx \
#       + mu*(inner(grad(v[1])[0], u1[1]))*dx \
#       - mu*(inner(v[1], u1[1]*n[0]))*ds \
#       - mu*(inner(jump(v[1], n[0]), avg(u1[1])))*dS \
#       + mu*(inner(grad(v[1])[1], u1[0]))*dx \
#       - mu*(inner(v[1], u1[0]*n[1]))*ds \
#       - mu*(inner(jump(v[1], n[1]), avg(u1[0])))*dS \

#F_s += inner(v[2], (s[2] - s0[2])/dt)*dx \
#       + mu*(inner(grad(v[2])[0], u1[1]))*dx \
#       - mu*(inner(v[2], u1[1]*n[0]))*ds \
#       - mu*(inner(jump(v[2], n[0]), avg(u1[1])))*dS \
#       + mu*(inner(grad(v[2])[1], u1[0]))*dx \
#       - mu*(inner(v[2], u1[0]*n[1]))*ds \
#       - mu*(inner(jump(v[2], n[1]), avg(u1[0])))*dS \

#F_s += inner(v[3], (s[3] - s0[3])/dt)*dx \
#       + l*inner(grad(v[3])[0], u1[0])*dx \
#       - l*inner(v[3], u1[0]*n[0])*ds \
#       - l*inner(jump(v[3], n[0]), avg(u1[0]))*dS \
#       + (l + 2*mu)*inner(grad(v[3])[1], u1[1])*dx \
#       - (l + 2*mu)*inner(jump(v[3], n[1]), avg(u1[1]))*dS \
#       - (l + 2*mu)*inner(v[3], u1[1]*n[1])*ds \

problem_u = LinearVariationalProblem(lhs(F_u), rhs(F_u), u1)
solver_u = LinearVariationalSolver(problem_u)

problem_s = LinearVariationalProblem(lhs(F_s), rhs(F_s), s1)
solver_s = LinearVariationalSolver(problem_s)

output_u = File("velocity.pvd")
output_s = File("stress.pvd")

# Initial conditions
uic = Expression(('exp(-50*pow((x[0]-1), 2))', '0'))
u0.assign(Function(WU).interpolate(uic))
sic = Expression((('-exp(-50*pow((x[0]-1), 2))', '0'),
                   ('0', '0')))
s0.assign(Function(WS).interpolate(sic))

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

