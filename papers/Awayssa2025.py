from framework import fitting_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

freqs = np.logspace(1, 6, 500) #Hz
omegas = 2*np.pi*freqs #rad/s

#tap water impedance
params = [175, 179, 6.345, 7.47, 38.29]
scaling = [1, 1, 1e-6, 1e-9, 1e-12]
z_complex = fitting_utils.Awayssa2025(params, [omegas, scaling])
z_complex = z_complex.astype("complex")

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(freqs, np.abs(z_complex), color="tab:blue")
plt.xscale('log')
plt.legend(["Proposed"])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.ylim(150, 400)
plt.xlim(10, 1e6)
plt.show()

plt.subplot(1,2,2)
plt.plot(freqs, np.rad2deg(np.angle(z_complex)), color="tab:blue")
plt.xscale('log')
plt.legend(["Proposed"])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase")
plt.ylim(-22, 0)
plt.xlim(10, 1e6)
plt.show()

plt.figure(2)
plt.plot(z_complex.real , -z_complex.imag, color="tab:blue")
plt.xlabel('Re(Z)')
plt.ylabel('-Im(Z)')
plt.grid()
plt.show()