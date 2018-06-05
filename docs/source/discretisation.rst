Discretisation and solution procedure
=====================================

The model equations are discretised using the Galerkin finite element method, and are advanced forward in time using a fourth-order leapfrog scheme. These are described below.

Spatial discretisation
----------------------

The discretisation of the governing equations in space is performed using the Galerkin finite element method. As a first step, the weak form of the elastic wave equation is derived by multiplying through by a test function :math:`\mathbf{w} \in H^1(\Omega)^d`, where :math:`H^1(\Omega)^d` is the first Hilbertian Sobolev space of dimension :math:`d`, and integrating over the domain :math:`\Omega` by parts on the RHS. For the velocity equation in the velocity-stress formulation, this yields

.. math:: \int_\Omega{\mathbf{w}\cdot\rho\frac{\partial\mathbf{u}}{\partial t}}\ \mathrm{dV} = -\int_\Omega{\mathbb{T}\cdot\nabla\mathbf{w}}\ \mathrm{dV} + \int_{\partial\Omega_e}{(\mathbf{w}\cdot\mathbb{T}|_{\partial\Omega_e})\cdot\mathbf{n}_e}\ \mathrm{dS} + \int_{\partial\Omega}{(\mathbf{w}\cdot\mathbb{T})\cdot\mathbf{n}}\ \mathrm{ds}
    :label: weak_form_velocity

where :math:`\mathbf{n}` is the vector normal to the domain boundary, :math:`\partial\Omega`, and :math:`\mathbf{n}_e` is the vector normal to the facet of element :math:`e`, :math:`\partial\Omega_e`. Given this weak form, a solution :math:`\mathbf{u} \in H^1(\Omega)^d` is sought for all test functions :math:`\mathbf{w} \in H^1(\Omega)^d`.

Discrete representations for both :math:`\mathbf{w}` and :math:`\mathbf{u}`, given by a linear combination of basis functions :math:`\{\phi_i\}_{i=1}^{N_e}`, which are discontinuous across element facets, are then used to replace them in :eq:`weak_form_velocity`:

.. math:: \mathbf{w} = \sum_{i=1}^{N_e} \phi_i\mathbf{w}_i,
    :label: discrete_test_function

.. math:: \mathbf{u} = \sum_{j=1}^{N_e} \phi_j\mathbf{u}_j,
    :label: discrete_velocity_trial_function

where :math:`N` is the total number of solution nodes. The stress field :math:`\mathbb{T}` is also represened by its own set of basis functions :math:`\{\psi_i\}_{i=1}^{N_e}`

.. math:: \mathbb{T} = \sum_{k=1}^{N_e} \psi_k\mathbb{T}_k.
    :label: discrete_stress_trial_function

Note that the underlying Firedrake framework permits the use of a range of basis function types and high-order function spaces, and it is assumed in Seigen that both the velocity and stress fields share the same function space.

By substituting :eq:`discrete_test_function`, :eq:`discrete_velocity_trial_function` and :eq:`discrete_stress_trial_function` into :eq:`weak_form_velocity`, and applying the fact that :math:`\mathbf{w}_i` are arbitrary, this yields

.. math:: \sum_{j=1}^{N_e}\int_{\Omega_e}{\phi_i\cdot\rho\phi_j}\ \mathrm{dV}\ \frac{\partial\mathbf{u}_j}{\partial t} = -\sum_{k=1}^{N_e}\int_{\Omega_e}{\psi_k\cdot\nabla\phi_i}\ \mathrm{dV}\ \mathbb{T}_k + \sum_{k=1}^{N_e}\int_{\partial\Omega_e}{(\phi_i\cdot\mathbb{T}|_{\partial\Omega_e})\cdot\mathbf{n}_e}\ \mathrm{dS} + \sum_{k=1}^{N_e}\int_{\partial\Omega}{(\phi_i\cdot\mathbb{T})\cdot\mathbf{n}}\ \mathrm{ds}
    :label: weak_form_velocity

for all :math:`\phi_i` in element :math:`e`. The volume integrals and interior facet integrals have been restricted to an individual element :math:`e` here because the discontinuous formulation results in each element becoming its own independent problem.

First-order upwinding
~~~~~~~~~~~~~~~~~~~~~

As a result of the discontinuous nature of the solution fields and test functions at element boundaries, the values of :math:`\mathbb{T}|_{\partial\Omega_e}` and :math:`\mathbf{u}|_{\partial\Omega_e}` in the facet integral terms must be treated carefully. For this work we use a first-order upwinding scheme such that the average values of :math:`\mathbb{T}` and :math:`\mathbf{u}` across the facets are used.

The discrete system
~~~~~~~~~~~~~~~~~~~

The end result is a discrete system of size :math:`N \times N` for both the velocity and stress equations:

.. math:: \mathbf{M}\frac{\partial\mathbf{u}}{\partial t} = \mathbf{D}\mathbb{T},

where

.. math:: \mathbf{M}_{ij} = \int_{\Omega_e}{\phi_i\rho\phi_j\ \mathrm{dV}},

.. math:: \mathbf{D}_{ij} = -\int_{\Omega_e}{\psi_j\cdot\nabla\phi_i}\ \mathrm{dV} + \int_{\partial\Omega_e}{(\phi_i\cdot\mathbb{T}|_{\partial\Omega_e})\cdot\mathbf{n}_e}\ \mathrm{dS} + \int_{\partial\Omega}{(\phi_i\cdot\mathbb{T})\cdot\mathbf{n}}\ \mathrm{ds}

The coefficients of the discrete representation of :math:`\mathbf{u}` must be solved for using a numerical solution method. A similar procedure can be performed on the stress equation. 

Temporal discretisation
-----------------------

A fourth-order explicit leap-frog scheme is used to treat the time derivatives in the elastic wave equation. This is based on a truncated Taylor series expansion of the velocity and stress fields whilst staggering their solutions by half a time unit. Hence, the velocity is first solved to obtain a solution at time level :math:`n+1` using information about the stress at time :math:`n+\frac{1}{2}`. This new velocity solution is then in turn used to solve for the stress field at time :math:`n+\frac{3}{2}`. Mathematically, this is written as

.. math:: \mathbf{u}^{n+1} = \mathbf{u}^{n} + \Delta t \frac{\partial\mathbf{u}^{n+\frac{1}{2}}}{\partial t} + \frac{\Delta t^3}{24}\frac{\partial^3\mathbf{u}^{n+\frac{1}{2}}}{\partial t^3}

.. math:: \mathbb{T}^{n+\frac{3}{2}} = \mathbb{T}^{n + \frac{1}{2}} + \Delta t \frac{\partial\mathbb{T}^{n+1}}{\partial t} + \frac{\Delta t^3}{24}\frac{\partial^3\mathbb{T}^{n+1}}{\partial t^3}

where the higher-order terms (:math:`O(\Delta t^5)`) have been neglected. The remaining time derivatives can be evaluated using auxiliary fields which need to be solved for at each time-step:

.. math:: \mathbf{u}^{n+1} = \mathbf{u}^{n} + \Delta t \mathbf{u}^{n+\frac{1}{2}}_{\bigstar} + \frac{\Delta t^3}{24}\mathbf{u}^{n+\frac{1}{2}}_{\bigstar\bigstar}

.. math:: \mathbb{T}^{n+\frac{3}{2}} = \mathbb{T}^{n + \frac{1}{2}} + \Delta t \mathbb{T}^{n+1}_{\bigstar} + \frac{\Delta t^3}{24} \mathbb{T}^{n+1}_{\bigstar\bigstar}

These auxiliary fields are defined as

.. math:: \mathbf{u}^{n+\frac{1}{2}}_{\bigstar} = f(\mathbb{T}^{n + \frac{1}{2}})
.. math:: \mathbb{T}^{n + \frac{1}{2}}_{\bullet} = g(\mathbf{u}^{n+\frac{1}{2}}_{\bigstar})
.. math:: \mathbf{u}^{n+\frac{1}{2}}_{\bigstar\bigstar} = f(\mathbb{T}^{n + \frac{1}{2}}_{\bullet})

and

.. math:: \mathbb{T}^{n+1}_{\bigstar} = g(\mathbf{u}^{n+1})
.. math:: \mathbf{u}^{n+1}_{\bullet} = f(\mathbb{T}^{n+1}_{\bigstar})
.. math:: \mathbb{T}^{n+1}_{\bigstar\bigstar} = g(\mathbf{u}^{n+1}_{\bullet})

where :math:`f` and :math:`g` are the right-hand sides of the velocity and stress equations (from the velocity-stress formulation of the elastic wave equation), respectively.

Both implicit and explicit solvers are available, encapsulated within the ``ElasticLF4`` class. The former solves the individual finite element variational problems (defined above) implicitly using a linear solver, while the latter explicitly solves the variational problems by assembling RHS vectors and multiplying them with the according global inverse mass matrix.
