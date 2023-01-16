Introduction to quTARANG
========================

**quTARANG**  is a fast, robust solver equipped with the functionality of running on both CPU as well as on a single GPU. It is designed to study turbulence in 
quantum fluids specially in Bose-Einstein condensate (BECs). It is based on the spectral method which is used to solve Gross-Pitaevskii equation (GPE) and the spectral method 
makes it more accurate then alternate finite difference methods. 
The Gross Pitaevskii equation well describes the Bose-Einstein condensate at absolute zero
temperature where all particles are in the lowest quantum state. The GPE is a Schrodinger equation 
with a nonlinear interaction term that takes into account the interaction between the particles. 
The GPE is given by
 
 .. math::

   \iota\hbar\partial_t\psi=-\frac{\hbar^2}{2m}\nabla^2\psi+V(\vec{r},t)\psi+g|\psi|^2\psi