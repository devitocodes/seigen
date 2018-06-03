Solvers
=======

The model equations are discretised using the Galerkin finite element method, and are advanced forward in time using a fourth-order leapfrog scheme. The underlying Firedrake framework permits the use of a range of basis function types and high-order function spaces, and it is assumed in Seigen that both the velocity and stress fields share the same function space.

Both implicit and explicit solvers are available, encapsulated within the ``ElasticLF4`` class. The former solves the individual finite element variational problems implicitly using a linear solver, while the latter explicitly solves the variational problems by assembling RHS vectors and multiplying them with the according global inverse mass matrix.
