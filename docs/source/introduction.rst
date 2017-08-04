Introduction
============

Overview
--------

`Seigen <https://github.com/opesci/seigen>`_ is a seismological modelling package that solves the elastic wave equation on unstructured meshes. It is part of the `OPESCI <http://www.opesci.org/>`_ project.

The Galerkin finite element method is used to discretise the governing equation in space, and a fourth-order leapfrog scheme advances the discretised equations forward in time. Seigen depends on the `Firedrake <http://firedrakeproject.org>`_ framework which introduces a considerable amount of automation into the numerical modelling process. The weak form of the elastic wave equation is written in a high-level, near-mathematical language known as the Unified Form Language (UFL). This UFL is then compiled down into optimised C code and executed efficiently, as a parallel loop over all the cells of the computational mesh, to perform the assembly of the discrete system of equations and then solve it.

License
-------

Seigen is open-source software and is released under the MIT license. See the file ``LICENSE.md`` for more information.
