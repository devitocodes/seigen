#!/usr/bin/env python

from firedrake import *

def fx(w, s0, n, absorption):
   return -inner(grad(w[0])[0], s0[0])*dx + inner(avg(s0[0]), jump(w[0], n[0]))*dS - inner(grad(w[0])[1], s0[1])*dx + inner(avg(s0[1]), jump(w[0], n[1]))*dS - inner(w[0], absorption*u[0])*dx
def fy(w, s0, n, absorption):
   return -inner(grad(w[1])[0], s0[2])*dx + inner(avg(s0[2]), jump(w[1], n[0]))*dS - inner(grad(w[1])[1], s0[3])*dx + inner(avg(s0[3]), jump(w[1], n[1]))*dS - inner(w[1], absorption*u[1])*dx

def gxx(v, u1, n, l, mu, source):
   return - (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
          + (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
          + (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
          - l*inner(grad(v[0])[1], u1[1])*dx \
          + l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
          + l*inner(v[0], u1[1]*n[1])*ds + inner(v[0], source)*dx

def gxy(v, u1, n, l, mu):
   return - mu*(inner(grad(v[1])[0], u1[1]))*dx \
          + mu*(inner(v[1], u1[1]*n[0]))*ds \
          + mu*(inner(jump(v[1], n[0]), avg(u1[1])))*dS \
          - mu*(inner(grad(v[1])[1], u1[0]))*dx \
          + mu*(inner(v[1], u1[0]*n[1]))*ds \
          + mu*(inner(jump(v[1], n[1]), avg(u1[0])))*dS

def gyx(v, u1, n, l, mu):
   return - mu*(inner(grad(v[2])[0], u1[1]))*dx \
          + mu*(inner(v[2], u1[1]*n[0]))*ds \
          + mu*(inner(jump(v[2], n[0]), avg(u1[1])))*dS \
          - mu*(inner(grad(v[2])[1], u1[0]))*dx \
          + mu*(inner(v[2], u1[0]*n[1]))*ds \
          + mu*(inner(jump(v[2], n[1]), avg(u1[0])))*dS
          
def gyy(v, u1, n, l, mu, source):    
   return - l*inner(grad(v[3])[0], u1[0])*dx \
          + l*inner(v[3], u1[0]*n[0])*ds \
          + l*inner(jump(v[3], n[0]), avg(u1[0]))*dS \
          - (l + 2*mu)*inner(grad(v[3])[1], u1[1])*dx \
          + (l + 2*mu)*inner(jump(v[3], n[1]), avg(u1[1]))*dS \
          + (l + 2*mu)*inner(v[3], u1[1]*n[1])*ds + inner(v[3], source)*dx



mesh = Mesh("src/domain.msh")

S = FunctionSpace(mesh, "DG", 3)
U = FunctionSpace(mesh, "DG", 3)
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
T = 2.5
dt = 0.001
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

absorption = Function(U).interpolate(Expression("x[0] <= 20 || x[0] >= 280 || x[1] <= 20.0 ? 1000 : 0"))

# uh1 solve
F_uh1 = inner(w[0], u[0])*dx - fx(w, s0, n, absorption)
F_uh1 += inner(w[1], u[1])*dx - fy(w, s0, n, absorption)
problem_uh1 = LinearVariationalProblem(lhs(F_uh1), rhs(F_uh1), uh1)
solver_uh1 = LinearVariationalSolver(problem_uh1)

# stemp solve
F_stemp = inner(v[0], s[0])*dx - gxx(v, uh1, n, l, mu, source)
F_stemp += inner(v[1], s[1])*dx - gxy(v, uh1, n, l, mu)
F_stemp += inner(v[2], s[2])*dx - gyx(v, uh1, n, l, mu)
F_stemp += inner(v[3], s[3])*dx - gyy(v, uh1, n, l, mu, source)
problem_stemp = LinearVariationalProblem(lhs(F_stemp), rhs(F_stemp), stemp)
solver_stemp = LinearVariationalSolver(problem_stemp)

# uh2 solve
F_uh2 = inner(w[0], u[0])*dx - fx(w, stemp, n, absorption)
F_uh2 += inner(w[1], u[1])*dx - fy(w, stemp, n, absorption)
problem_uh2 = LinearVariationalProblem(lhs(F_uh2), rhs(F_uh2), uh2)
solver_uh2 = LinearVariationalSolver(problem_uh2)

# u1 solve
F_u1 = density*inner(w[0], (u[0] - u0[0])/dt)*dx - inner(w[0], uh1[0])*dx - ((dt**2)/24.0)*inner(w[0], uh2[0])*dx
F_u1 += density*inner(w[1], (u[1] - u0[1])/dt)*dx - inner(w[1], uh1[1])*dx - ((dt**2)/24.0)*inner(w[1], uh2[1])*dx
problem_u1 = LinearVariationalProblem(lhs(F_u1), rhs(F_u1), u1)
solver_u1 = LinearVariationalSolver(problem_u1)

# sh1 solve
F_sh1 = inner(v[0], s[0])*dx - gxx(v, u1, n, l, mu, source)
F_sh1 += inner(v[1], s[1])*dx - gxy(v, u1, n, l, mu)
F_sh1 += inner(v[2], s[2])*dx - gyx(v, u1, n, l, mu)
F_sh1 += inner(v[3], s[3])*dx - gyy(v, u1, n, l, mu, source)
problem_sh1 = LinearVariationalProblem(lhs(F_sh1), rhs(F_sh1), sh1)
solver_sh1 = LinearVariationalSolver(problem_sh1)

# utemp solve
F_utemp = inner(w[0], u[0])*dx - fx(w, sh1, n, absorption)
F_utemp += inner(w[1], u[1])*dx - fy(w, sh1, n, absorption)
problem_utemp = LinearVariationalProblem(lhs(F_utemp), rhs(F_utemp), utemp)
solver_utemp = LinearVariationalSolver(problem_utemp)

# sh2 solve
F_sh2 = inner(v[0], s[0])*dx - gxx(v, utemp, n, l, mu, source)
F_sh2 += inner(v[1], s[1])*dx - gxy(v, utemp, n, l, mu)
F_sh2 += inner(v[2], s[2])*dx - gyx(v, utemp, n, l, mu)
F_sh2 += inner(v[3], s[3])*dx - gyy(v, utemp, n, l, mu, source)
problem_sh2 = LinearVariationalProblem(lhs(F_sh2), rhs(F_sh2), sh2)
solver_sh2 = LinearVariationalSolver(problem_sh2)

# s1 solve
F_s1 = inner(v[0], (s[0] - s0[0])/dt)*dx - inner(v[0], sh1[0])*dx - ((dt**2)/24.0)*inner(v[0], sh2[0])*dx
F_s1 += inner(v[1], (s[1] - s0[1])/dt)*dx - inner(v[1], sh1[1])*dx - ((dt**2)/24.0)*inner(v[1], sh2[1])*dx
F_s1 += inner(v[2], (s[2] - s0[2])/dt)*dx - inner(v[2], sh1[2])*dx - ((dt**2)/24.0)*inner(v[2], sh2[2])*dx
F_s1 += inner(v[3], (s[3] - s0[3])/dt)*dx - inner(v[3], sh1[3])*dx - ((dt**2)/24.0)*inner(v[3], sh2[3])*dx
problem_s1 = LinearVariationalProblem(lhs(F_s1), rhs(F_s1), s1)
solver_s1 = LinearVariationalSolver(problem_s1)

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
   
   output_u << u1
   #output_s << s1.split()[0]
   
   # Move onto next timestep
   t += dt

