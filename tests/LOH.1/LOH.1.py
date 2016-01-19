#!/usr/bin/env python

from firedrake import *
from elastic_wave.elastic import *
from elastic_wave.helpers import *

with timed_region('mesh setup'):
    mesh = Mesh("src/domain.msh")
elastic = ElasticLF4(mesh, "DG", 1, dimension=3)

# Constants
C = FunctionSpace(mesh, "CG", 1)
D = FunctionSpace(mesh, "DG", 1)
elastic.density = Function(D).interpolate(Expression("x[2] >= 1000 ? 2600 : 2700"))
elastic.dt = 0.001
elastic.mu = Function(C).interpolate(Expression("x[2] >= 1000 ? 10.4e9 : 32.4e9"))
elastic.l = Function(C).interpolate(Expression("x[2] >= 1000 ? 20.8e9 : 32.4e9"))

print "P-wave velocity in Medium 1: %f" % Vp(10.4e9, 20.8e9, 2600)
print "S-wave velocity in Medium 1: %f" % Vs(10.4e9, 2600)
print "P-wave velocity in Medium 2: %f" % Vp(32.4e9, 32.4e9, 2700)
print "S-wave velocity in Medium 2: %f" % Vs(32.4e9, 2700)

# Source
smoothness = 0.1
elastic.source_expression = Expression((("0.0", "x[0] == 0 && x[1] == 0.0 && x[2] >= 1000 && x[2] <= 3000 ? t/(pow(smoothness, 2))*exp(-t/smoothness) : 0.0", "0.0"),
                                       ("x[0] == 0 && x[1] == 0.0 && x[2] >= 1000 && x[2] <= 3000 ? t/(pow(smoothness, 2))*exp(-t/smoothness) : 0.0", "0.0", "0.0"),
                                       ("0.0", "0.0", "0.0")), smoothness=smoothness, t=0)
elastic.source_function = Function(elastic.S)
elastic.source = elastic.source_expression

# Absorption
elastic.absorption_function = Function(D)
elastic.absorption = Expression("x[2] <= 0 && ( x[0] <= -14000 || x[0] >= 14000 || x[1] <= -14000 || x[1] >= 14000 ) ? 1000 : 0")

# Initial conditions
uic = Expression(("0.0", "0.0", "0.0"))
elastic.u0.assign(Function(elastic.U).interpolate(uic))
sic = Expression((("0", "0", "0"),
                  ("0", "0", "0"),
                  ("0", "0", "0")))
elastic.s0.assign(Function(elastic.S).interpolate(sic))

# Start the simulation
T = 9.0
elastic.run(T)
