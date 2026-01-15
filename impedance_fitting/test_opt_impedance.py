from framework import file_lcr, characterization_utils
import numpy as np
from impedance.models.circuits import CustomCircuit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def MSE(z_hat, z):
    SSE = np.sum((np.abs(z)-np.abs(z_hat))**2) #sum of squared errors
    return SSE/len(z_hat)

#PHOBOS spectroscopy acquisition
spec_air_obj = file_lcr.read('../data/testICE_12_12_25/c0.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_ice_obj = file_lcr.read('../data/testICE_12_12_25/cice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_12_12_25/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#dielectric parameters
h2o_eps_real, h2o_eps_imag = characterization_utils.dielectric_params_corrected(spec_h2o_obj, spec_air_obj, spec_h2o_obj.freqs) #compute the spectrum based on the experimental data
h2o_z_real, h2o_z_imag = characterization_utils.complex_impedance(spec_h2o_obj, spec_h2o_obj.freqs) #compute the complex impedance based on the experimental data
h2o_z = h2o_z_real - 1j*h2o_z_imag
ice_eps_real, ice_eps_imag = characterization_utils.dielectric_params_corrected(spec_ice_obj, spec_air_obj, spec_ice_obj.freqs) #compute the spectrum based on the experimental data
ice_z_real, ice_z_imag = characterization_utils.complex_impedance(spec_ice_obj, spec_ice_obj.freqs) #compute the complex impedance based on the experimental data
ice_z = ice_z_real - 1j*ice_z_imag

#Simple Rp/Cp circuit
R_candidates = [0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3, 0.6e-3, 0.7e-3, 0.8e-3, 0.9e-3, 1e-3,
                0.1e-2, 0.2e-2, 0.3e-2, 0.4e-2, 0.5e-2, 0.6e-2, 0.7e-2, 0.8e-2, 0.9e-2, 1e-2,
                0.1e-1, 0.2e-1, 0.3e-1, 0.4e-1, 0.5e-1, 0.6e-1, 0.7e-1, 0.8e-1, 0.9e-1, 1e-1,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                0.1e1, 0.2e1, 0.3e1, 0.4e1, 0.5e1, 0.6e1, 0.7e1, 0.8e1, 0.9e1, 1e1,
                0.1e2, 0.2e2, 0.3e2, 0.4e2, 0.5e2, 0.6e2, 0.7e2, 0.8e2, 0.9e2, 1e2,
                0.1e3, 0.2e3, 0.3e3, 0.4e3, 0.5e3, 0.6e3, 0.7e3, 0.8e3, 0.9e3, 1e3,
                0.1e4, 0.2e4, 0.3e4, 0.4e4, 0.5e4, 0.6e4, 0.7e4, 0.8e4, 0.9e4, 1e4]
C_candidates = [0.1e-8, 0.2e-8, 0.3e-8, 0.4e-8, 0.5e-8, 0.6e-8, 0.7e-8, 0.8e-8, 0.9e-8, 1e-8,
                0.1e-7, 0.2e-7, 0.3e-7, 0.4e-7, 0.5e-7, 0.6e-7, 0.7e-7, 0.8e-7, 0.9e-7, 1e-7,
                0.1e-6, 0.2e-6, 0.3e-6, 0.4e-6, 0.5e-6, 0.6e-6, 0.7e-6, 0.8e-6, 0.9e-6, 1e-6,
                0.1e-5, 0.2e-5, 0.3e-5, 0.4e-5, 0.5e-5, 0.6e-5, 0.7e-5, 0.8e-5, 0.9e-5, 1e-5,
                0.1e-4, 0.2e-4, 0.3e-4, 0.4e-4, 0.5e-4, 0.6e-4, 0.7e-4, 0.8e-4, 0.9e-4, 1e-4,
                0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3, 0.6e-3, 0.7e-3, 0.8e-3, 0.9e-3, 1e-3,
                0.1e-2, 0.2e-2, 0.3e-2, 0.4e-2, 0.5e-2, 0.6e-2, 0.7e-2, 0.8e-2, 0.9e-2, 1e-2,
                0.1e-1, 0.2e-1, 0.3e-1, 0.4e-1, 0.5e-1, 0.6e-1, 0.7e-1, 0.8e-1, 0.9e-1, 1e-1]
W_o10_candidates = [0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3, 0.6e-3, 0.7e-3, 0.8e-3, 0.9e-3, 1e-3,
                0.1e-2, 0.2e-2, 0.3e-2, 0.4e-2, 0.5e-2, 0.6e-2, 0.7e-2, 0.8e-2, 0.9e-2, 1e-2,
                0.1e-1, 0.2e-1, 0.3e-1, 0.4e-1, 0.5e-1, 0.6e-1, 0.7e-1, 0.8e-1, 0.9e-1, 1e-1,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                0.1e1, 0.2e1, 0.3e1, 0.4e1, 0.5e1, 0.6e1, 0.7e1, 0.8e1, 0.9e1, 1e1,
                0.1e2, 0.2e2, 0.3e2, 0.4e2, 0.5e2, 0.6e2, 0.7e2, 0.8e2, 0.9e2, 1e2,
                0.1e3, 0.2e3, 0.3e3, 0.4e3, 0.5e3, 0.6e3, 0.7e3, 0.8e3, 0.9e3, 1e3,
                0.1e4, 0.2e4, 0.3e4, 0.4e4, 0.5e4, 0.6e4, 0.7e4, 0.8e4, 0.9e4, 1e4]
W_o11_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                0.1e1, 0.2e1, 0.3e1, 0.4e1, 0.5e1, 0.6e1, 0.7e1, 0.8e1, 0.9e1, 1e1,
                0.1e2, 0.2e2, 0.3e2, 0.4e2, 0.5e2, 0.6e2, 0.7e2, 0.8e2, 0.9e2, 1e2,
                0.1e3, 0.2e3, 0.3e3, 0.4e3, 0.5e3, 0.6e3, 0.7e3, 0.8e3, 0.9e3, 1e3,
                0.1e4, 0.2e4, 0.3e4, 0.4e4, 0.5e4, 0.6e4, 0.7e4, 0.8e4, 0.9e4, 1e4]
C_pe_candidates = [0.1e-4, 0.2e-4, 0.3e-4, 0.4e-4, 0.5e-4, 0.6e-4, 0.7e-4, 0.8e-4, 0.9e-4, 1e-4,
                0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3, 0.6e-3, 0.7e-3, 0.8e-3, 0.9e-3, 1e-3,
                0.1e-2, 0.2e-2, 0.3e-2, 0.4e-2, 0.5e-2, 0.6e-2, 0.7e-2, 0.8e-2, 0.9e-2, 1e-2,
                0.1e-1, 0.2e-1, 0.3e-1, 0.4e-1, 0.5e-1, 0.6e-1, 0.7e-1, 0.8e-1, 0.9e-1, 1e-1,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

opt_MSE = np.inf #variable to store the optimal MSE value
opt_circuit = None #object of the optimal circuit

#grid search to optimize the hyperparameters
for i in range(len(R_candidates)):
    for j in range(len(C_candidates)):
        for k in range(len(R_candidates)):
            for l in range(len(W_o10_candidates)):
                for m in range(len(W_o11_candidates)):
                    for n in range(len(C_pe_candidates)):
                        for o in range(len(C_pe_candidates)):
                            curr_guess = [R_candidates[i], C_candidates[j], R_candidates[k], W_o10_candidates[l], W_o11_candidates[m], C_pe_candidates[n], C_pe_candidates[o]]
                            try:
                                circuit = CustomCircuit('p(R_1, C_0)-p(R_2-Wo_1,CPE_1)',
                                                initial_guess=curr_guess)
                                circuit.fit(spec_ice_obj.freqs, ice_z)
                                curr_fit = circuit.predict(spec_ice_obj.freqs)
                                curr_MSE = MSE(ice_z, curr_fit) #compute the MSE
                                if np.abs(curr_MSE) < opt_MSE:
                                    opt_MSE = curr_MSE #commute the variables
                                    #opt_args = curr_guess #update the optimal guess
                                    opt_circuit = circuit #update the optimal circuit
                            except Exception as e:
                                print(f'[test_opt_impedance.py] Skipped {curr_guess}: {e}')

#compute the fit for the optimal circuit
print(f'Optimal circuit = {opt_circuit}')
opt_circuit.fit(spec_ice_obj.freqs, ice_z)
opt_fit = opt_circuit.predict(spec_ice_obj.freqs)
curr_MSE = MSE(ice_z, opt_fit) #compute the MSE

#plot
plt.figure()
leg = []
plt.scatter(ice_z_real, ice_z_imag, marker='o', color="tab:blue")
leg.append('ice measured')
plt.plot(opt_fit.real, -opt_fit.imag, color="tab:orange")
leg.append('ice model (impedance.py)')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), np.abs(ice_z))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(opt_fit), color="tab:orange")
leg.append('ice measured')
plt.ylabel("|Z|")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), -np.angle(ice_z.astype('complex')))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(opt_fit), color="tab:orange")
leg.append('ice measured')
plt.ylabel("-∠Z")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()
plt.show()


