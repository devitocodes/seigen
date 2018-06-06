# Summary

This test case considers the propagation of a smooth wave across a 2D rectangular domain, defined by 0 <= x <= 4 m. Not only does this allow one to check that the wave is propagating at the expected velocity, it also demonstrates that the wave does not dissipate over time as a result of the non-dissipative nature of the numerical scheme.

To run this simulation, execute "python pulse_1d_lf4.py" at the command line. The solution fields will be written to VTK files which can be visualised in programs such as ParaView.
