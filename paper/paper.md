---
title: 'quTARANG: A python GPE solver to study turbulence in quantum system'
tags:
  - Python
  - Quantum turbulence
  - Bose-Einstein condensate
  - Quantum Fluids


authors:
  - name: Sachin Singh Rawat
    orcid: 0000-0002-5701-7247
    equal-contrib: true
    affiliation: 1
  - name: Shawan Kumar Jha
    orcid: 0000-0003-4582-8787
    equal-contrib: true 
    affiliation: 2
  - name: Mahendra Kumar Verma
    orcid: 0000-0002-3380-4561
    affiliation: 1
  - name: Pankaj Kumar Mishra
    orcid: 0000-0003-4907-4724
    affiliation: 2

affiliations:
 - name: Department of Physics, Indian Institute of Technology - Kanpur, Uttar Pradesh - 208016, India
   index: 1
 - name: Department of Physics, Indian Institute of Technology - Guwahati, Asam - 781039, India
   index: 2

date: 17 Jan 2023

bibliography: resources/paper.bib
---
# Summary
Turbulence is a phenomenon associated with the chaotic nature of flows in space and time solely arising due to the nonlinear nature of the interactions. Turbulence in classical fluids, as best characterised by the Navier-Stokes equation, remains unresolved to this day. The complex nonlinear interactions at various length and time scales make it a difficult problem to handle analytically as well as numerically. One recent approach to shed some light on this long-standing problem has been the study of turbulence in quantum fluids. The zero viscosity and quantized vortices of quantum fluid systems like Bose-Einstein Condensates (BECs) [@madeira2020quantum] distinguish it from its classical counterparts.
BEC is a state of matter where Bose particles occupy the ground state upon cooling to a very low temperature and thus can be represented by a macroscopic wave function. One can model the dynamics of BECs using the mean-field Gross-Pitaevskii Equation (GPE) given by [@muruganandam2009fortran]
\begin{equation}\label{eqn:GPE}
i\hbar\partial_t\psi(\vec{r},t) = -\frac{\hbar^2}{2m}\nabla^2\psi(\vec{r},t) + V(\vec{r},t)\psi(\vec{r},t) + NU_0|\psi(\vec{r},t)|^2\psi(\vec{r},t),
\end{equation}

where $\psi(\vec{r},t)$ is the macroscopic complex wave function, $m$ is the atomic mass, $V(\vec{r},t)$ is the trapping potential, $N$ is the number of particles, $\displaystyle U_0=(4\pi\hslash^2a_s)/m$ is the nonlinear interaction parameter and $a_s$ denotes the scattering length for the interaction of the atomic particles.

Our quantum simulator code ``quTARANG`` is primarily designed for studying quantum turbulence in BECs by solving the GPE in laminar as well as in the turbulent regime.

# Statement of Need
``quTARANG`` is a robust and easy-to-use application that solves the GPE with Graphics Processing Unit (GPU). GPUs are specialized units primarily designed for image processing and for performing massive multigrid simulations at high speed. They have a large number of parallelizing units compared to CPUs, due to which they are being widely used to speed up code. There are no packages available in python that can solve turbulent GPE in 2D and 3D on both CPUs and GPUs. There exist, however, some software packages in other languages that can solve the GPE such as GPELab [@Antoine2014], Massively Parallel Trotter-Suzuki Solver [@Wittek2013], CUDA-enabled GPUE [@schloss2018gpue], a split-step Crank-Nicolson based Fortran code [@muruganandam2009fortran] and MPI-OpenMP enabled Gross-Pitaevskii Solver (GPS) [@kobayashi2021quantum]. 

![A comparison of the dynamical evolution of the root-mean-square size of the condensate in the $x$ ($\sigma_x$), $y$ ($\sigma_y$), and $z$ ($\sigma_z$) direction obtained from our simulation (for 3D GPE) and those obtained by Bao et al. [@bao2003numerical]. A perfect match is being obtained.\label{fig:dynamic}](resources/dynamics.jpeg){width=70%}

# Numerical Scheme and functionalities

We have chosen the frequency ($\omega^{-1}$) as the time scale, oscillator length $a_0 = \sqrt{\hbar/m\omega}$ as the characteristic length scale and the harmonic oscillator ground state energy $\hbar\omega$ as the energy scale. With this formalism, the non-dimensional variables can be written as $t'=\omega t$, $\vec{r}'=\vec{r}/a_0$ and $\psi' = a_0^{3/2}\psi$. In what follows, we omit the prime $(')$ from the variables. The non-dimensional form of the GPE is given by [@muruganandam2009fortran]

\begin{equation}\label{eqn:ndGPE}
i\partial_t\psi(\vec{r},t)= -\frac{1}{2}\nabla^2\psi(\vec{r},t) + V(\vec{r},t)\psi(\vec{r},t) +g|\psi(\vec{r},t)|^2\psi(\vec{r},t)
\end{equation}
where $g$ is the non-dimensional interaction parameter. 

``quTARANG`` uses a pseudo-spectral scheme, Time-splitting spectral (TSSP) method [@bao2003numerical], to solve the dynamics of the GPE. The main advantage of using the TSSP scheme is that it is unconditionally stable and conserves the total particle number.

The ground state calculations in ``quTARANG`` are done by using TSSP with an imaginary time propagation method. In this method, one replaces $t$ with $-it$. As we propagate in imaginary time, the eigenstates with higher energies begin to decay faster than the ground state as a result of which only the ground state survives. The wavefunction needs to be normalised at each time step in order to conserve total particle number.

``quTARANG`` is equipped with various features, which include:

1. Ground state calculations for different potentials such as harmonic and anharmonic trap, optical lattice potential, time-dependent potential and stochastic potential.
2. Long-time dynamical evolution of different states using either CPUs or GPUs.
3. Computation of different quantities relevant to the study of turbulence phenomenon in BECs, such as components of kinetic energy (KE) and various spectra (compressible KE spectrum, incompressible KE spectrum and particle number spectrum).

|   $Dimension$        | **$g$**      | **$r_{rms}$** | **$r^*_{rms}$** | **$\mu$** | **$\mu^*$**|
|:------------:|:------------:|:-------------:|:----------------:|:----------:|:----------:|
|              | -2.5097      | 0.87771       | 0.87759          | 0.49987   | 0.49978     |
|              | 0            | 1.00000       | 1.00000          | 1.00000   | 1.00000     |
|              | 3.1371       | 1.10504       | 1.10513          | 1.42009   | 1.42005     |
|     2        | 12.5484      | 1.30656       | 1.30687          | 2.25609   | 2.25583     |
|              | 62.742       | 1.78722       | 1.78817          | 4.61136   | 4.60982     |
|              | 313.71       | 2.60122       | 2.60441          | 10.07639  | 10.06825    |
|              | 627.42       | 3.07914       | 3.08453          | 14.20569  | 14.18922    |
|              |              |               |                  |           |             |
|              | 0            | 1             | 1                | 3.0000    | 3.0000      |
|              | 18.81        | 1.3778        | 1.3249           | 4.3618    | 4.3611      |
|              | 94.05        | 1.8222        | 1.7742           | 6.6824    | 6.6797      |
|     3        | 188.1        | 2.0881        | 2.0411           | 8.3718    | 8.3671      |
|              | 940.5        | 2.8912        | 2.8424           | 14.9663   | 14.9487     |
|              | 1881         | 3.3268        | 3.2758           | 19.5058   | 19.4751     |
|              | 7524         | 4.3968        | 4.3408           | 33.5623   | 33.4677     |
|              | 15048        | 5.0497        | 4.9922           | 44.1894   | 44.0234     |
: The condensate width ($r_{rms}$) and chemical potential ($\mu$) obtained for the ground state using ``quTARANG``. $r^*_{rms}$ and $\mu^*$ are the corresponding values from [@muruganandam2009fortran] for comparison. \label{table:gstate}

# Results
We have calculated the ground state and dynamics for given sets of initial conditions and compared them with the standard results. 

1. **Validation of ground state** : The initial condition for 2D and 3D cases are given as:

    2D : $\psi(\vec{r},0) = \left(\frac{1}{\pi}\right)^{1/2}e^{-\frac{(x^2+y^2)}{2}}$, \ $V(\vec{r}) = \frac{1}{2}(x^2 + y^2)$ 

    3D : $\psi(\vec{r},0) = \left(\frac{1}{\pi}\right)^{3/4}e^{-\frac{(x^2+y^2+z^2)}{2}}$, \ $V(\vec{r}) = \frac{1}{2}(x^2 + y^2 + 4z^2)$ 

    The ground state obtained by ``quTARANG`` has been compared with that obtained by using the finite difference code of Muruganandam and Adhikari [@muruganandam2009fortran]. The results for harmonic potential well for 2D and 3D are in good agreement with Muruganandam and Adhikari [@muruganandam2009fortran] as shown in \autoref{table:gstate}. The root-mean-square size ($r_{rms}$) of the condensate is defined as $r_{rms} = \left(\int r^2 |\psi(\vec{r}, t)|^2dV \right)^{1/2}$.

2.  **Validation of dynamics** : We have validated the dynamic evolution of a state by comparing our results with Bao et al. [@bao2003numerical] for the following condition: 
$$\psi(\vec{r},0)=\frac{(\gamma_y\gamma_z)^{1/4}}{\sqrt{(\pi\epsilon_1)^{3/4}}}e^{-\frac{(x^2+\gamma_yy^2+\gamma_zz^2)}{2\epsilon_1}}, \ \ \ V(\vec{r},0)=\frac{1}{2}(x^2+\gamma_y^2y^2+\gamma_z^2z^2),$$
where $\gamma_y = 2.0$, $\gamma_z = 4.0$, $\epsilon_1 = 0.25$ and $g = 0.1$.
Fig \ref{fig:dynamic} shows the comparisons of the rms size of the condensate in $x$, $y$ and $z$ directions at different times calculated by using our code and those obtained by Bao et al. [@bao2003numerical]. The results obtained from the ``quTARANG`` are in good agreement with the results obtained from Bao et al. for the same initial conditions.

# References
