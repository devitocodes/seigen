import pylab

dx = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
uerror = [0.012296, 0.000695, 0.000044, 0.000016, 0.000005, 0.000001]
serror = [0.030805, 0.006410, 0.001573, 0.000392, 0.000098, 0.000024]
second_order = [1.0e-2*(1.0/4.0)**i for i in range(0, len(uerror))]
fourth_order = [1.0e-2*(1.0/16.0)**i for i in range(0, len(uerror))]

pylab.loglog(dx, uerror, 'g--o', label="Velocity")
pylab.loglog(dx, serror, 'r--o', label="Stress")
pylab.loglog(dx, second_order, label="Second-order convergence")
pylab.loglog(dx, fourth_order, label="Fourth-order convergence")
pylab.legend(loc='best')
pylab.xlabel("Characteristic element length")
pylab.ylabel("Error in L2 norm")
pylab.savefig("error.png")
