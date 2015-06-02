#!/usr/bin/env python

from firedrake import *
from elastic_wave.elastic import *
from elastic_wave.helpers import *

mesh = Mesh("src/domain.msh")
elastic = ElasticLF4(mesh, "DG", 1, dimension=2)

# Constants
elastic.density = 1.0
elastic.dt = 0.001
elastic.mu = 3600.0
elastic.l = 3599.3664

print "P-wave velocity: %f" % Vp(elastic.mu, elastic.l, elastic.density)
print "S-wave velocity: %f" % Vs(elastic.mu, elastic.density)

# Source
a = 159.42
elastic.source_expression = Expression("x[0] >= 44.5 && x[0] <= 45.5 && x[1] >= 148.5 && x[1] <= 149.5 ? (-1.0 + 2*a*pow(t - 0.3, 2))*exp(-a*pow(t - 0.3, 2)) : 0.0", a=a, t=0)
HU = FunctionSpace(mesh, "DG", 4)
elastic.source_function = Function(HU)
elastic.source = elastic.source_expression

# Absorption
elastic.absorption_function = Function(elastic.U)
elastic.absorption = Expression("x[0] <= 20 || x[0] >= 280 || x[1] <= 20.0 ? 1000 : 0")

# Initial conditions
uic = Expression(('0.0', '0.0'))
elastic.u0.assign(Function(elastic.WU).interpolate(uic))
sic = Expression((('0','0'),
                  ('0','0')))
elastic.s0.assign(Function(elastic.WS).interpolate(sic))

# Start the simulation
T = 2.5
elastic.run(T)
