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
R_candidates = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
#R1_candidates = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
C_candidates = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
all_MSE = np.zeros((len(R_candidates), len(C_candidates)))
#all_MSE = np.zeros((len(R_candidates), len(R1_candidates), len(C_candidates)))
opt_MSE = np.inf #variable to store the optimal MSE value
opt_args = [None, None] #list to store the optimal electrical parameters
#opt_args = [None, None, None] #list to store the optimal electrical parameters
opt_circuit = None #object of the optimal circuit

#grid search to optimize the hyperparameters
# for i in range(len(R_candidates)):
#     for j in range(len(R1_candidates)):
#         for k in range(len(C_candidates)):
#             print(f'candidate: i = {i}, j = {j}, k = {k}')
#             curr_guess = [R_candidates[i], R1_candidates[j], C_candidates[k]]
#             circuit = CustomCircuit('R_0-p(R_1,C_0)',
#                             initial_guess=curr_guess)
#             circuit.fit(spec_ice_obj.freqs, ice_z)
#             curr_fit = circuit.predict(spec_ice_obj.freqs)
#             curr_MSE = MSE(ice_z, curr_fit) #compute the MSE
#             all_MSE[i,j,k] = np.abs(curr_MSE)
#             if np.abs(curr_MSE) < opt_MSE:
#                 opt_MSE = curr_MSE #commute the variables
#                 opt_args = curr_guess #update the optimal guess
#                 opt_circuit = circuit #update the optimal circuit

for i in range(len(R_candidates)):
    for j in range(len(C_candidates)):
        print(f'candidate: i = {i}, j = {j}')
        curr_guess = [R_candidates[i], C_candidates[j]]
        circuit = CustomCircuit('p(R_0,C_0)',
                        initial_guess=curr_guess)
        circuit.fit(spec_ice_obj.freqs, ice_z)
        curr_fit = circuit.predict(spec_ice_obj.freqs)
        curr_MSE = MSE(ice_z, curr_fit) #compute the MSE
        all_MSE[i,j] = np.abs(curr_MSE)
        if np.abs(curr_MSE) < opt_MSE:
            opt_MSE = curr_MSE #commute the variables
            opt_args = curr_guess #update the optimal guess
            opt_circuit = circuit #update the optimal circuit

#compute the fit for the optimal circuit
print(f'Optimal circuit = {opt_circuit}')
opt_fit = opt_circuit.predict(spec_ice_obj.freqs)

# initial_guess = [.1, .005, .1, .9, .005, .1, 200, .1, .9]
# opt_circuit = CustomCircuit('R_0-p(R_1,CPE_1)-p(R_2-Wo_1,CPE_2)',
#                         initial_guess=initial_guess)
# opt_circuit.fit(spec_ice_obj.freqs, ice_z)
# opt_fit = opt_circuit.predict(spec_ice_obj.freqs)
# curr_MSE = MSE(ice_z, opt_fit) #compute the MSE
# print(opt_circuit)

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

# plt.figure()
# plt.imshow(all_MSE,
#           aspect='auto',
#           extent=[C_candidates[0], C_candidates[-1], R_candidates[0], R_candidates[-1]])
# plt.xlabel("C_candidates")
# plt.ylabel("R_candidates")
# plt.colorbar()
# plt.show()

# import plotly.graph_objects as go
# x, y, z = np.indices((8, 8, 8))
# fig = go.Figure(data=go.Volume(
#     x=x.flatten(),
#     y=y.flatten(),
#     z=z.flatten(),
#     value=all_MSE.flatten(),
#     opacity=0.1,
#     surface_count=15,
# ))
#
# fig.show()

#for i in range(len(C_candidates)):
