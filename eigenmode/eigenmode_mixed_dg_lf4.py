#!/usr/bin/env python

from firedrake import *

def fx(w, s0, n):
   return -inner(grad(w[0])[0], s0[0])*dx + inner(avg(s0[0]), jump(w[0], n[0]))*dS - inner(grad(w[0])[1], s0[1])*dx + inner(avg(s0[1]), jump(w[0], n[1]))*dS
def fy(w, s0, n):
   return -inner(grad(w[1])[0], s0[2])*dx + inner(avg(s0[2]), jump(w[1], n[0]))*dS - inner(grad(w[1])[1], s0[3])*dx + inner(avg(s0[3]), jump(w[1], n[1]))*dS

def gxx(v, u1, n, l, mu):
   return - (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
          + (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
          + (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
          - l*inner(grad(v[0])[1], u1[1])*dx \
          + l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
          + l*inner(v[0], u1[1]*n[1])*ds

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
          
def gyy(v, u1, n, l, mu):    
   return - l*inner(grad(v[3])[0], u1[0])*dx \
          + l*inner(v[3], u1[0]*n[0])*ds \
          + l*inner(jump(v[3], n[0]), avg(u1[0]))*dS \
          - (l + 2*mu)*inner(grad(v[3])[1], u1[1])*dx \
          + (l + 2*mu)*inner(jump(v[3], n[1]), avg(u1[1]))*dS \
          + (l + 2*mu)*inner(v[3], u1[1]*n[1])*ds


def run(N, degree, dt):
   mesh = UnitSquareMesh(N, N)

   S = FunctionSpace(mesh, "DG", degree)
   U = FunctionSpace(mesh, "DG", degree)
   dimension = 2

   WS = MixedFunctionSpace([S, S, S, S])
   WU = MixedFunctionSpace([U, U])

   s = TrialFunctions(WS)
   v = TestFunctions(WS)

   u = TrialFunctions(WU)
   w = TestFunctions(WU)

   s0 = Function(WS)
   sh1 = Function(WS)
   stemp = Function(WS)
   sh2 = Function(WS)
   s1 = Function(WS)

   u0 = Function(WU)
   uh1 = Function(WU)
   utemp = Function(WU)
   uh2 = Function(WU)
   u1 = Function(WU)
   
   # Constants
   density = 1.0
   T = 5.0
   dt = dt
   mu = 0.25
   l = 0.5

   Vp = sqrt((l + 2*mu)/density) # P-wave velocity
   Vs = sqrt(mu/density) # S-wave velocity

   n = FacetNormal(mesh)

   # uh1 solve
   F_uh1 = inner(w[0], u[0])*dx - fx(w, s0, n)
   F_uh1 += inner(w[1], u[1])*dx - fy(w, s0, n)
   problem_uh1 = LinearVariationalProblem(lhs(F_uh1), rhs(F_uh1), uh1)
   solver_uh1 = LinearVariationalSolver(problem_uh1)
   
   # stemp solve
   F_stemp = inner(v[0], s[0])*dx - gxx(v, uh1, n, l, mu)
   F_stemp += inner(v[1], s[1])*dx - gxy(v, uh1, n, l, mu)
   F_stemp += inner(v[2], s[2])*dx - gyx(v, uh1, n, l, mu)
   F_stemp += inner(v[3], s[3])*dx - gyy(v, uh1, n, l, mu)
   problem_stemp = LinearVariationalProblem(lhs(F_stemp), rhs(F_stemp), stemp)
   solver_stemp = LinearVariationalSolver(problem_stemp)

   # uh2 solve
   F_uh2 = inner(w[0], u[0])*dx - fx(w, stemp, n)
   F_uh2 += inner(w[1], u[1])*dx - fy(w, stemp, n)
   problem_uh2 = LinearVariationalProblem(lhs(F_uh2), rhs(F_uh2), uh2)
   solver_uh2 = LinearVariationalSolver(problem_uh2)

   # u1 solve
   F_u1 = density*inner(w[0], (u[0] - u0[0])/dt)*dx - inner(w[0], uh1[0])*dx - ((dt**2)/24.0)*inner(w[0], uh2[0])*dx
   F_u1 += density*inner(w[1], (u[1] - u0[1])/dt)*dx - inner(w[1], uh1[1])*dx - ((dt**2)/24.0)*inner(w[1], uh2[1])*dx
   problem_u1 = LinearVariationalProblem(lhs(F_u1), rhs(F_u1), u1)
   solver_u1 = LinearVariationalSolver(problem_u1)
   
   # sh1 solve
   F_sh1 = inner(v[0], s[0])*dx - gxx(v, u1, n, l, mu)
   F_sh1 += inner(v[1], s[1])*dx - gxy(v, u1, n, l, mu)
   F_sh1 += inner(v[2], s[2])*dx - gyx(v, u1, n, l, mu)
   F_sh1 += inner(v[3], s[3])*dx - gyy(v, u1, n, l, mu)
   problem_sh1 = LinearVariationalProblem(lhs(F_sh1), rhs(F_sh1), sh1)
   solver_sh1 = LinearVariationalSolver(problem_sh1)

   # utemp solve
   F_utemp = inner(w[0], u[0])*dx - fx(w, sh1, n)
   F_utemp += inner(w[1], u[1])*dx - fy(w, sh1, n)
   problem_utemp = LinearVariationalProblem(lhs(F_utemp), rhs(F_utemp), utemp)
   solver_utemp = LinearVariationalSolver(problem_utemp)

   # sh2 solve
   F_sh2 = inner(v[0], s[0])*dx - gxx(v, utemp, n, l, mu)
   F_sh2 += inner(v[1], s[1])*dx - gxy(v, utemp, n, l, mu)
   F_sh2 += inner(v[2], s[2])*dx - gyx(v, utemp, n, l, mu)
   F_sh2 += inner(v[3], s[3])*dx - gyy(v, utemp, n, l, mu)
   problem_sh2 = LinearVariationalProblem(lhs(F_sh2), rhs(F_sh2), sh2)
   solver_sh2 = LinearVariationalSolver(problem_sh2)
   
   # s1 solve
   F_s1 = inner(v[0], (s[0] - s0[0])/dt)*dx - inner(v[0], sh1[0])*dx - ((dt**2)/24.0)*inner(v[0], sh2[0])*dx
   F_s1 += inner(v[1], (s[1] - s0[1])/dt)*dx - inner(v[1], sh1[1])*dx - ((dt**2)/24.0)*inner(v[1], sh2[1])*dx
   F_s1 += inner(v[2], (s[2] - s0[2])/dt)*dx - inner(v[2], sh1[2])*dx - ((dt**2)/24.0)*inner(v[2], sh2[2])*dx
   F_s1 += inner(v[3], (s[3] - s0[3])/dt)*dx - inner(v[3], sh1[3])*dx - ((dt**2)/24.0)*inner(v[3], sh2[3])*dx
   problem_s1 = LinearVariationalProblem(lhs(F_s1), rhs(F_s1), s1)
   solver_s1 = LinearVariationalSolver(problem_s1)

   #output_u = File("velocity.pvd")
   #output_s = File("stress.pvd")

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
   H = FunctionSpace(mesh, "DG", 6)
   temp = Function(H)
   temp_test = TestFunction(H)
   temp_trial = TrialFunction(H)
   while t <= T + 1e-12:
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
      
      # Move onto next timestep
      t += dt

      #output_u << u1
      #output_s << s1.split()[0]
      
      #print "|u-uexact| = %f" % errornorm(u1.split()[0], uexact.split()[0])
      #print "|s-sexact| = %f" % errornorm(s1, sexact)
      
   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(u1.split()[0]-uexact.split()[0]))*dx
   solve(lhs(G) == rhs(G), temp)
   ux_error = norm(temp)

   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(u1.split()[1]-uexact.split()[1]))*dx
   solve(lhs(G) == rhs(G), temp)
   uy_error = norm(temp)

   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(s1.split()[0]-sexact.split()[0]))*dx
   solve(lhs(G) == rhs(G), temp)
   sxx_error = norm(temp)

   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(s1.split()[1]-sexact.split()[1]))*dx
   solve(lhs(G) == rhs(G), temp)
   sxy_error = norm(temp)

   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(s1.split()[2]-sexact.split()[2]))*dx
   solve(lhs(G) == rhs(G), temp)
   syx_error = norm(temp)

   G = inner(temp_test, temp_trial)*dx - inner(temp_test, abs(s1.split()[3]-sexact.split()[3]))*dx
   solve(lhs(G) == rhs(G), temp)
   syy_error = norm(temp)

   return ux_error, uy_error, sxx_error, sxy_error, syx_error, syy_error

def convergence_analysis():
   degrees = range(1, 5)
   N = [2**i for i in range(2, 6)]
   
   dx = [1.0/n for n in N]  
   
   for d in degrees:
      dt = [0.5*(1.0/n)/(2.0**(d-1)) for n in N] # Courant number of 0.25: (dx*C)/Vp
      
      f = open("error_u_p%d_lf4.dat" % d, "w")
      f.write("dx\tdt\tux_error\tuy_error\tsxx_error\tsxy_error\tsyx_error\tsyy_error\n")
      for i in range(len(N)):
         ux_error, uy_error, sxx_error, sxy_error, syx_error, syy_error = run(N[i], d, dt[i])
         f.write(str(dx[i]) + "\t" + str(dt[i]) + "\t" + str(ux_error) + "\t" + str(uy_error) + "\t" + str(sxx_error) + "\t" + str(sxy_error) + "\t" + str(syx_error) + "\t" + str(syy_error) + "\n")
      f.close()
      
   return
   
convergence_analysis()

