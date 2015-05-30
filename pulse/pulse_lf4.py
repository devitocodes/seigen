#!/usr/bin/env python

from firedrake import *
import numpy

def fx(w, s0, n, absorption):
   return -inner(grad(w[0])[0], s0[0])*dx + inner(avg(s0[0]), jump(w[0], n[0]))*dS - inner(grad(w[0])[1], s0[1])*dx + inner(avg(s0[1]), jump(w[0], n[1]))*dS - inner(w[0], absorption*u[0])*dx

def gxx(v, u1, n, l, mu):
   return - (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
          + (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
          + (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
          - l*inner(grad(v[0])[1], u1[1])*dx \
          + l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
          + l*inner(v[0], u1[1]*n[1])*ds

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

s0 = Function(WS, name="StressOld")
sh1 = Function(WS)
stemp = Function(WS)
sh2 = Function(WS)
s1 = Function(WS, name="StressNew")

u0 = Function(WU, name="VelocityOld")
uh1 = Function(WU)
utemp = Function(WU)
uh2 = Function(WU)
u1 = Function(WU, name="VelocityNew")

# Constants
density = 1.0
T = 2.0
dt = 0.0025
mu = 0.25
l = 0.5

Vp = sqrt((l + 2*mu)/density) # P-wave velocity
Vs = sqrt(mu/density) # S-wave velocity

n = FacetNormal(mesh)
absorption = Function(U).interpolate(Expression("x[0] >= 3.5 || x[0] <= 0.5 ? 100.0 : 0"))

# uh1 solve
F_uh1 = inner(w[0], u[0])*dx - fx(w, s0, n, absorption)
problem_uh1 = LinearVariationalProblem(lhs(F_uh1), rhs(F_uh1), uh1)
solver_uh1 = LinearVariationalSolver(problem_uh1)

# stemp solve
F_stemp = inner(v[0], s[0])*dx - gxx(v, uh1, n, l, mu)
problem_stemp = LinearVariationalProblem(lhs(F_stemp), rhs(F_stemp), stemp)
solver_stemp = LinearVariationalSolver(problem_stemp)

# uh2 solve
F_uh2 = inner(w[0], u[0])*dx - fx(w, stemp, n, absorption)
problem_uh2 = LinearVariationalProblem(lhs(F_uh2), rhs(F_uh2), uh2)
solver_uh2 = LinearVariationalSolver(problem_uh2)

# u1 solve
F_u1 = density*inner(w[0], (u[0] - u0[0])/dt)*dx - inner(w[0], uh1[0])*dx - ((dt**2)/24.0)*inner(w[0], uh2[0])*dx
problem_u1 = LinearVariationalProblem(lhs(F_u1), rhs(F_u1), u1)
solver_u1 = LinearVariationalSolver(problem_u1)

# sh1 solve
F_sh1 = inner(v[0], s[0])*dx - gxx(v, u1, n, l, mu)
problem_sh1 = LinearVariationalProblem(lhs(F_sh1), rhs(F_sh1), sh1)
solver_sh1 = LinearVariationalSolver(problem_sh1)

# utemp solve
F_utemp = inner(w[0], u[0])*dx - fx(w, sh1, n, absorption)
problem_utemp = LinearVariationalProblem(lhs(F_utemp), rhs(F_utemp), utemp)
solver_utemp = LinearVariationalSolver(problem_utemp)

# sh2 solve
F_sh2 = inner(v[0], s[0])*dx - gxx(v, utemp, n, l, mu)
problem_sh2 = LinearVariationalProblem(lhs(F_sh2), rhs(F_sh2), sh2)
solver_sh2 = LinearVariationalSolver(problem_sh2)

# s1 solve
F_s1 = inner(v[0], (s[0] - s0[0])/dt)*dx - inner(v[0], sh1[0])*dx - ((dt**2)/24.0)*inner(v[0], sh2[0])*dx
problem_s1 = LinearVariationalProblem(lhs(F_s1), rhs(F_s1), s1)
solver_s1 = LinearVariationalSolver(problem_s1)

output_u = File("velocity.pvd")
output_s = File("stress.pvd")

# Initial conditions
uic = Expression(('exp(-50*pow((x[0]-1), 2))', '0'))
u0.assign(Function(WU).interpolate(uic))
sic = Expression((('-exp(-50*pow((x[0]-1), 2))', '0'),
                   ('0', '0')))
s0.assign(Function(WS).interpolate(sic))

t = 0
while t < T:
   t += dt
   print "t = %f" % t
   
   # Solve for the velocity vector
   solver_uh1.solve()
   solver_stemp.solve()
   solver_uh2.solve()
   solver_u1.solve()
   u0.assign(u1)
   
   # Solve for the stress tensor
   solver_sh1.solve()
   solver_utemp.solve()
   solver_sh2.solve()
   solver_s1.solve()
   s0.assign(s1)

   if(numpy.isclose(t, 0.25) or numpy.isclose(t, 1.0) or numpy.isclose(t, 2.0)):
      output_u << u1
      #output_s << s1

