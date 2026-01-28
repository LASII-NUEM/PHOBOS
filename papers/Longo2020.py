from framework import fitting_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

freqs = np.logspace(3, np.log10(5e6), 402) #Hz
omegas = 2*np.pi*freqs #rad/s

#THF + water
params = [5.16, 4, 7.54, 4.34, 7.41, 1.37, 0.6, 1.64]
scaling = [1e3, 1e-8, 1e2, 1e-7, 1e4, 1, 1, 1e-3]
z_thf = fitting_utils.Longo2020(params, [omegas, scaling])
z_thf = z_thf.astype("complex")

#hydrate
params = [5.10, 3.57, 1.15, 2.23, 2.66, 5.46, 0.6, 8]
scaling = [1e6, 1, 1e4, 1e-14, 1e5, 1e-6, 1, 1e-7]
z_hyd = fitting_utils.Longo2020(params, [omegas, scaling])
z_hyd = z_hyd.astype("complex")

plt.figure(1)
plt.plot(z_thf.real, -z_thf.imag, color="red")
plt.legend(["R||C + C||(R + R||Q)"])
plt.xlabel("Re(Z)")
plt.ylabel("-Im(Z)")
plt.xlim([2000, 7000])
plt.ylim([0, 4000])
plt.show()

plt.figure(2)
plt.plot(z_hyd.real, -z_hyd.imag, color="blue")
plt.legend(["R||C + C||(R + R||Q)"])
plt.xlabel("Re(Z)")
plt.ylabel("-Im(Z)")
plt.xlim([0, 300000])
plt.ylim([0, 80000])
plt.show()