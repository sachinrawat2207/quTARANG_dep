from gpe.set_device import xp

def wfc_qho_1D(x, y=0, z=0):
    return xp.pi**(-0.25) * xp.exp(-0.5*x**2) + 0j

def wfc_qho_2D(x, y=0, z=0):
    return xp.pi**(-0.50) * xp.exp(-0.5*(x**2 + y**2)) + 0j

def wfc_qho_3D(x, y=0, z=0):
    return xp.pi**(-0.75) * xp.exp(-0.5*(x**2 + y**2 + z**2)) + 0j

