import pylab

degrees = range(1, 5)

for d in degrees:
   f = open("error_p%d_lf2.dat" % d, "r")
   f.readline() # Skip the header
   dx = []
   ux_error = []
   for line in f:
      s = line.split()
      dx.append(float(s[0]))
      ux_error.append(float(s[2]))

   second_order = [1.0e-2*(1.0/4.0)**i for i in range(0, len(ux_error))]
   
   pylab.figure()
   pylab.loglog(dx, ux_error, 'g--o', label="Velocity")
   
   pylab.loglog(dx, second_order, label="Second-order convergence")
   pylab.legend(loc='best')
   pylab.xlabel("Characteristic element length")
   pylab.ylabel("Error in L2 norm")
   pylab.savefig("error_u_p%d_lf4.pdf" % d)
