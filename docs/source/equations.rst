Model equations
===============

Elastic wave equation
---------------------

Seigen solves the elastic wave equation for two primitive variables, velocity and stress, denoted by :math:`\mathbf{u} = \mathbf{u}(x,t)` and :math:`\mathbb{T} = \mathbb{T}(x,t)` respectively. The equation often written down as a second-order linear PDE,

.. math:: \rho\frac{\partial^2\mathbf{r}}{\partial t^2} = \nabla\cdot\mathbb{T}

where :math:`\mathbf{r}` is the displacement vector. However, for convenience, Seigen considers the velocity-stress formulation such that the equation is split into two first-order linear PDEs,

.. math:: \rho\frac{\partial\mathbf{u}}{\partial t} = \nabla\cdot\mathbb{T},

.. math:: \frac{\partial\mathbb{T}}{\partial t} = \lambda\left(\nabla\cdot\mathbf{u}\right)\mathbb{I} + \mu\left(\nabla\mathbf{u} + \nabla\mathbf{u}^\mathrm{T}\right).

The stress tensor :math:`\mathbb{T}` is defined by

.. math:: \mathbb{T} = \lambda\left(\nabla\cdot\mathbf{u}\right)\mathbb{I} + \mu\left(\nabla\mathbf{u} + \nabla\mathbf{u}^\mathrm{T}\right),

where :math:`\mathbb{I}` is the identity tensor, and :math:`\lambda` and :math:`\mu` are the two Lame parameters.

Initial conditions
------------------

In order for the governing equations to be marched forward in time, the initial conditions

.. math:: \mathbb{T}(x,t=0) = \mathbb{T}^0

.. math:: \mathbf{u}(x,t=0) = \mathbf{u}^0

must be specified.


Boundary conditions
-------------------

Seigen can enforce conditions for velocity or stress along a specified boundary.

Free-surface condition
~~~~~~~~~~~~~~~~~~~~~~

In the strong sense, the free-surface condition enforces 

.. math:: \mathbb{T} \cdot \mathbf{n} = 0,

where :math:`\mathbf{n}` is the normal vector pointing outwards at the boundary. Note that this is defined here in the strong sense; the condition is in fact enforced weakly with the finite element method.
