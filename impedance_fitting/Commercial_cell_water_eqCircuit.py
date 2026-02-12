from framework import file_lcr, characterization_utils
import numpy as np
from impedance.models.circuits import Randles, CustomCircuit
from impedance.visualization import plot_nyquist
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def MSE(z_hat, z):
    SSE = np.sum((np.abs(z)-np.abs(z_hat))**2) #sum of squared errors
    return SSE/len(z_hat)

# LCR spectroscopy acquisition
LCR_spec_air_obj = file_lcr.read('../data/test_media_12_01/c0.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)
LCR_spec_c1_obj = file_lcr.read('../data/test_media_12_01/tap.csv', n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean)

#dielectric parameters
c1_eps_real, c1_eps_imag = characterization_utils.dielectric_params_corrected(LCR_spec_c1_obj, LCR_spec_air_obj, LCR_spec_c1_obj.freqs) #compute the spectrum based on the experimental data
c1_z_real, c1_z_imag = characterization_utils.complex_impedance(LCR_spec_c1_obj, LCR_spec_c1_obj.freqs) #compute the complex impedance based on the experimental data
c1_z = c1_z_real - 1j*c1_z_imag

initial_guess = [80.45, 1, 100, 3320, 1e-3, 1e3, 1]
circuit = CustomCircuit(initial_guess=initial_guess,
                              circuit='R_0-p(Wo_1,CPE_1)-p(R_2,C_1)')

circuit.fit(LCR_spec_c1_obj.freqs, c1_z)
print(circuit)

opt_fit = circuit.predict(LCR_spec_c1_obj.freqs)
curr_MSE = MSE(c1_z, opt_fit) #compute the MSE (option - impedance.models.circuits.fitting.rmse)
print(curr_MSE)

#plot
plt.figure()
leg = []
plt.scatter(c1_z_real, c1_z_imag, marker='o', color="tab:blue")
leg.append('c1 measured')
plt.plot(opt_fit.real, -opt_fit.imag, color="tab:orange")
leg.append('c1 model (impedance.py)')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.scatter(np.log10(LCR_spec_c1_obj.freqs), np.abs(c1_z))
leg.append('c1 measured')
plt.plot(np.log10(LCR_spec_c1_obj.freqs), np.abs(opt_fit), color="tab:orange")
leg.append('c1 measured')
plt.ylabel("|Z|")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.scatter(np.log10(LCR_spec_c1_obj.freqs), -np.angle(c1_z.astype('complex')))
leg.append('c1 measured')
plt.plot(np.log10(LCR_spec_c1_obj.freqs), -np.angle(opt_fit), color="tab:orange")
leg.append('c1 measured')
plt.ylabel("-∠Z")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()
plt.show()
