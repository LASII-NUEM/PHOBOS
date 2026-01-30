from framework import fitting_utils, equivalent_circuits
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

scaling = [1, 1, 1, 1, 1, 1]
freqs = np.logspace(-1, 6, 201) #Hz
omegas = 2*np.pi*freqs #rad/s

#Exp1
params = [0.00398, 0.008, 1.109, 0.8, 0.0034, 0.0872]
Z_exp1 = equivalent_circuits.Fouquet2005(params, [omegas, scaling]) #complex impedance
Z_exp1 = Z_exp1.astype("complex")

#Exp3
params = [0.00406, 0.0123, 1.080, 0.8, 0.0094, 0.0818]
Z_exp3 = equivalent_circuits.Fouquet2005(params, [omegas, scaling]) #complex impedance
Z_exp3 = Z_exp3.astype("complex")

#Exp5
params = [0.004, 0.0147, 1.102, 0.8, 0.0172, 0.0784]
Z_exp5 = equivalent_circuits.Fouquet2005(params, [omegas, scaling]) #complex impedance
Z_exp5 = Z_exp5.astype("complex")

#Exp12
params = [0.00416, 0.0163, 0.936, 0.8, 0.0312, 0.0947]
Z_exp12 = equivalent_circuits.Fouquet2005(params, [omegas, scaling]) #complex impedance
Z_exp12 = Z_exp12.astype("complex")

plt.figure()
leg = []
plt.plot(Z_exp1.real, -Z_exp1.imag, color="black")
leg.append('Exp1')
plt.plot(Z_exp3.real, -Z_exp3.imag, color="black", linestyle="--")
leg.append('Exp3')
plt.plot(Z_exp5.real, -Z_exp5.imag, color="black", linestyle="-.")
leg.append('Exp5')
plt.plot(Z_exp12.real, -Z_exp12.imag, color="black", linestyle=":")
leg.append('Exp12')
plt.xlabel('Re(Z)')
plt.ylabel('-Im(Z)')
plt.legend(leg)
plt.xlim([0.004, 0.052])
plt.ylim([-10e-3, 25e-3])
plt.grid()
plt.show()
