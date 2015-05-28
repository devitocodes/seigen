import pylab

degrees = range(1, 5)

for d in degrees:
   f = open("error_u_p%d_lf4.dat" % d, "r")
   f.readline() # Skip the header
   dx = []
   ux_error = []
   for line in f:
      s = line.split()
      dx.append(float(s[0]))
      ux_error.append(float(s[2]))

   second_order = [1.0e-2*(1.0/4.0)**i for i in range(0, len(ux_error))]
   third_order = [1.0e-2*(1.0/8.0)**i for i in range(0, len(ux_error))]
   fourth_order = [1.0e-2*(1.0/16.0)**i for i in range(0, len(ux_error))]
   
   pylab.figure()
   pylab.loglog(dx, ux_error, 'g--o', label="Velocity")
   
   if(d == 1):
      pylab.loglog(dx, second_order, label="Second-order convergence")
   elif(d == 2):
      pylab.loglog(dx, third_order, label="Third-order convergence")
   else:
      pylab.loglog(dx, fourth_order, label="Fourth-order convergence")
   pylab.legend(loc='best')
   pylab.xlabel("Characteristic element length")
   pylab.ylabel("Error in L2 norm")
   pylab.savefig("error_u_p%d_lf4.png" % d)
