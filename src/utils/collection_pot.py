from gpe.set_device import xp

def qho(x, y=0, z=0, w = [1, 1, 1]):
    """Quantum harmonic oscillator potential
    """
    return 0.5 * (w[0]*x**2 + w[1]*y**2 + w[2]*y**2)

def optical(x, y=0, z=0, A = [1, 1, 1], w = [1, 1, 1], phase = [0, 0, 0]):
    """Optical lattice potential.
    """
    return A[0]*xp.sin(w[0]*x+phase[0])**2 + A[1]*xp.sin(w[1]*y+phase[1])**2 + A[2]*xp.sin(w[2]*z+phase[2])**2

