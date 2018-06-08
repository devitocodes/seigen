---
title: 'Seigen: An application of code generation to high-order seismic modelling'
tags:
  - seismology
  - seismic modelling
  - numerical modelling
  - code generation
  - finite element method
  - elastic wave equation
authors:
  - name: Christian T. Jacobs
    orcid: 0000-0002-0034-4650
    affiliation: 1
  - name: Michael Lange
    orcid: 0000-0002-3232-0127
    affiliation: 1
  - name: Fabio Luporini
    orcid: 0000-0001-7161-2942
    affiliation: 1
  - name: Gerard J. Gorman
    orcid: 0000-0003-0563-3678
    affiliation: 1
affiliations:
 - name: Imperial College London
   index: 1
date: 8 June 2018
bibliography: paper.bib
---

# Summary

## Motivation behind code generation

Seismological modelling has traditionally involved the use of finite difference methods (FDMs) to solve the equations governing the propagation of acoustic and elastic waves [@Kelly_etal_1976, @Virieux_1986, @Graves_1996]. Such methods are relatively easy to implement, and direct addressing and regular memory access patterns means that good performance can be readily achieved [@Liu_etal_2014]. Therefore, a lot of effort has focused on optimising these models for use in seismic imaging on a wide range of hardware architectures.

One way to increase the accuracy and computational efficiency is to use higher-order methods. As the order increases, the grid size (and therefore the timestep) can also increase. This is particularly important for elastic wave modelling because the shear waves (S-waves) generated have a smaller wavelength than the pressure wave (P-wave) and therefore require higher resolution. However, as the order of the FDM stencils increases, so does the cost of creating optimised implementations of these methods on modern many-core and multi-core platforms.

Interest in stencil languages and compilers such as Liszt [@DeVito_etal_2011] and Pochoir [@Tang_etal_2011] has rapidly grown in recent years. Using a stencil language in effect allows for a seperation of concerns between those developing the numerical algorithms, and compiler specialists who optimise these parallel patterns onto different computer architectures. FDMs are readily expressed in stencil languages and this provides a programmatic route to generating implementations of high-order FDM stencils. However, FDM methods themselves have their own problems; the inherent structured nature of the underlying computational mesh is unable to follow realistic topography [@Liu_etal_2014], and is unsuitable for adapting to features in the subsurface model (e.g. one ideally would like a coarser mesh in salt domes where the wave velocity is much higher than the surrounding medium).

Promising results have recently been published on high-order discontinuous Galerkin (DG) finite element methods (FEMs) [@DumbserKaser_2006, @KaserDumbser_2006, @delaPuente_etal_2008, @delaPuente_etal_2009, @Delcourte_etal_2009, @Hermann_etal_2011, @Wenk_etal_2013, @Tago_etal_2014, @MerceratGlinsky_2015]. FEMs are more suitable for simulations requiring unstructured meshes that conform well to arbitrary geometries. Much like high-order finite difference stencils, it would be beneficial if higher-order DG FEM methods could be generated with ease.

Code generation techniques provided by the FEniCS framework [@Logg_etal_2012, @LoggWells_2010] and, more recently, the Firedrake framework [@Rathgeber_etal_2017], have provided a way to produce optimised finite element-based models from a high-level problem description. Model developers provide the weak form of the equations they wish to solve using a domain-specific language called the Unified Form Language (UFL) [@Alnaes_etal_2014], which then gets optimised and compiled down into C code [@Luporini_etal_2015]. A finite element tabulator called FIAT [@Kirby_2004] is used to construct arbitrary order polynomial spaces which is ideal for higher-order DG methods. Much like the FDM stencil compilers, these code generation frameworks provide a separation of concerns between application specialists and parallel computing experts. Furthermore, code generation techniques can potentially yield significant performance benefits over traditional 'static'/hard-coded finite element models which may be difficult or costly to optimise by hand.

In addition to many-core CPUs, the field of seismic modelling has already begun to benefit from using more exotic hardware architectures such as GPUs [@WeissShragge_2013]. In the case of the Firedrake framework, the code generation workflow allows the model code to be targeted towards a particular hardware architecture using the PyOP2 library [@Rathgeber_etal_2012, @Markall_etal_2013], thereby future-proofing the model in the face of future high-performance computer architectures.

## Seigen

Seigen is one of the first known applications of code generation techniques to high-order, finite element-based seismological modelling. The elastic wave equation and its discretisation procedure are implemented in the high-level UFL domain-specific language, which does not need to be modified by the user. Setting up a simulation simply requires writing a short Python-based problem description file; this file imports Seigen's elastic wave solver module and specifies, for example, constants such as the Lame parameters and the timestep size, initial conditions, and a source/absorption term. The user can also select an arbitrary-order function space for the spatial discretisation, and a fourth-order leapfrog algorithm advances the equations forward in time.

Upon execution of this problem description file, the Firedrake automated modelling framework converts the weak formulation of the governing equations and any additional expressions (e.g. for the user-specified initial conditions, source term, etc) into several C kernels which are then compiled and executed efficiently in parallel over the computational mesh to perform the simulation. All solution fields are written out in the VTK format [@Schroeder_etal_2006] for visualisation with tools such as ParaView [@Ayachit_2015].

The source code is available on GitHub (https://github.com/opesci/seigen) and has been released under the MIT License.

# Acknowledgements

This work is part of the Open Performance portablE SeismiC Imaging (OPESCI) project, funded by the Intel Parallel Computing Center at Imperial College London and SENAI CIMATEC.

# References

