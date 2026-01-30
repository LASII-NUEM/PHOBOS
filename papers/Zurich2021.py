from framework import fitting_utils, equivalent_circuits
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

freqs = np.logspace(-3, np.log10(5e6), 1000) #Hz
omegas = 2*np.pi*freqs #rad/s

#tap water impedance
params = [80.45, 4.59, 0.845, 1.184, 3320, 1.834]
scaling = [1, 1e-5, 1, 1e6, 1, 1e-10]
z_complex = equivalent_circuits.Zurich2021(params, [omegas, scaling])
z_complex = z_complex.astype("complex")

plt.figure(1)
plt.plot(z_complex.real, -z_complex.imag, color="tab:orange")
plt.legend(["R - CPE||Zws - R||C"])
plt.xlabel("Re(Z)")
plt.ylabel("-Im(Z)")
plt.xlim(0, 2e5)
plt.ylim(0, 2e5)
plt.show()
