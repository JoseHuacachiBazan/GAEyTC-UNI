import numpy as np
import matplotlib.pyplot as plt

# Parámetros del SM
MZ_ESM = 91.1876  # GeV
g = 0.652
sin2W = 0.23126

# Rango extremadamente reducido de v2
# (cambiar según la sensibilidad que quieras estudiar)
v2 = np.linspace(0, 50, 500)   # GeV

# Ecuación del MZ en el 331
prefactor = (g**2 / 4) * (1 / (1 - sin2W)) * 2
MZ_331 = np.sqrt(MZ_ESM**2 + prefactor * v2**2)

# Gráfica
plt.figure()
plt.plot(v2, MZ_331)

plt.xlabel(r'$v_2 \;[\mathrm{GeV}]$')
plt.ylabel(r'$M_Z \;[\mathrm{GeV}]$')
plt.title('Variación de $M_Z$ (modelo 331) en un rango pequeño de $v_2$')
plt.grid(True)
plt.tight_layout()
plt.show()
