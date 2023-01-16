Numerical Scheme Used
=====================

quTARANG uses a pseudo specral scheme, TSSP (Time Splitting Spectral method) :cite:p:`bao2003numerical` to solve the dynamics of the GPE.

The main advantage of using the TSSP scheme is that it is unconditionally stable scheme. The dimensionless for of GPE is given by

.. math:: \iota \frac{\partial\psi(\vec{r},t)}{\partial t}= -\frac{1}{2}\nabla^2\psi(\vec{r},t) + V(\vec{r},t)\psi(\vec{r},t) + g|\psi(\vec{r},t)|^2\psi(\vec{r},t)
   :label: eq:GPE
     


For time interval :math:`\Delta t` between :math:`t=t_n` and :math:`t=t_{n+1}`, one can solve above equation numerically by splitting it into two steps. 
The first step is

.. math::

    \iota \partial_t\psi = -\frac{1}{2}\nabla^2\psi

The second step is

.. math::

    \iota \partial_t\psi = V\psi + g|\psi|^2\psi


By taking a fourier trasnform of :eq:`eq:GPE` , one can convert the PDE into a list of PDEs which can be solved exactly in Fourier space and the wavefunction in real space can be retrieved by taking an inverse fourier transform.
For :math:`t \ \epsilon \ [t_n,t_{n+1}]`, :math:`|\psi|^2`  remains almost constant therefore, eq(\ref{eq:sstep2}), now just an ODE, can be solved exactly in :math:`t_n` and :math:`t_{n+1}`

Between :math:`t_n` and :math:`t_{n+1}`, the two steps are connected through strang splitting:

.. math::

    \psi_n^{(1)} = \psi_n e^{-\iota(V + g|\psi_n|^2)\frac{\Delta t}{2}} \\
    \hat{\psi}_n^{(2)} = \hat{\psi}_n^{(1)}e^{-\iota\frac{\vec{k}^2}{2}\Delta t} \\
    \psi_{n+1} = \psi_n^{(2)} e^{-\iota(V + g|\psi_n^{(2)}|^2)\frac{\Delta t}{2}}


where, :math:`\hat{\psi}^{(1)}` is Fourier transform of :math:`\psi^{(1)}`` and :math:`\psi_n^{(2)}` is inverse Fourier transform of :math:`\hat{\psi}_n^{(2)}`.

One can calculate the ground state for a given system  by using imaginary time proppogation method wherein all the eigenstates except the groundstate of the system decay with time. In imaginary time propagation method, :math:`t` is replaced by -:math:`\iota t` and then evolved.


.. bibliography::   