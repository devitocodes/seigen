Introduction
============

Overview
--------

Seigen is a seismological model based on the elastic wave equation. The governing equations are solved using the Galerkin finite element method and (currently) a fourth-order leapfrog time-stepping scheme.

The `Firedrake <http://firedrakeproject.org>`_ framework is used to automate the solution process. The weak form of the elastic wave equation is written in a high-level, near-mathematical language known as the Unified Form Language (UFL). This UFL is then compiled down into low-level optimised C code and executed efficiently, as a parallel loop over all the cells of the computational mesh, to perform the assembly of the discrete system of equations and then solve it.


