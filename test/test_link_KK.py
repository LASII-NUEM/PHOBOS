from framework import file_lcr, fitting_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from impedance_fitting import script_to_overwrite_local_impedancedotpy_files
import time

def compute_mu(R):
    neg_R = R[R<0]
    pos_R = R[R>=0]
    return 1 - np.sum(np.abs(neg_R))/np.sum(np.abs(pos_R))

def compute_residues(z, z_hat):
    z = z.astype("complex")
    z_hat = z_hat.astype("complex")
    return (z.real*z_hat.real)/np.abs(z), (z.imag*z_hat.imag)/np.abs(z)

def complex_cost_fun(z, z_hat):
    z = z.astype("complex")
    z_hat = z_hat.astype("complex")
    real_mse = ((z.real-z_hat.real)/np.abs(z))**2
    imag_mse = ((z.imag-z_hat.imag)/np.abs(z))**2
    return np.sum(real_mse+imag_mse)

#PHOBOS spectroscopy acquisition
spec_ice_obj = file_lcr.read('../data/testICE_30_01_26/c_ice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#Impedance fitting
fit_obj = fitting_utils.EquivalentCircuit("Zurich2021", spec_ice_obj, spec_ice_obj.freqs) #quivalent circuit object

# #fit the circuit with lin-KK
c = 0.35
fit_type = 'complex'
omega = 2*np.pi*fit_obj.freqs #rad/s
#M, mu, Z_linKK, res_real, res_imag = linKK(fit_obj.freqs, fit_obj.z_meas.astype("complex"), c=.5, max_M=100, fit_type='complex', add_cap=True)

M_max = 100
add_capacitor = True

#Lin-KK
t_init = time.time()
M = 2 #starting M
mu = 1
while True:
    M += 1 #update M

    #distribution of time constants
    tau = np.zeros(shape=(M,)) #M time-constants
    tau_min = 1/np.max(omega) #the smallest time constant
    tau_max = 1/np.min(omega) #the largest time constant
    k_idx = np.arange(2, M, 1) #indexes of k to compute the remainder time constants
    tau[1:-1] = 10**(np.log10(tau_min) + ((k_idx-1)/(M-1))*np.log10(tau_max/tau_min))
    tau[0] = tau_min
    tau[-1] = tau_max

    #build the A matrix (real and imag)
    if add_capacitor:
        #Rk values plus the ohmic resistor, a capacitor, and an inductance
        A_re = np.zeros(shape=(len(omega), M+3))
        A_imag = np.zeros(shape=(len(omega), M+3))
    else:
        #Rk values plus the ohmic resistor and an inductance
        A_re = np.zeros(shape=(len(omega), M+2))
        A_imag = np.zeros(shape=(len(omega), M+2))

    #handle real components
    Z = fit_obj.z_meas
    Z = Z.astype("complex")
    A_re[:,0] = 1/np.abs(Z) #measured impedance
    A_imag[:,-2] = -1/(omega*np.abs(Z)) #inductance
    A_imag[:,-1] = omega/np.abs(Z) #capacitance
    # for i in range(len(tau)):
    #     K = 1/(1+1j*omega*tau[i])
    #     A_re[:,i+1] = K.real/np.abs(Z)
    #     A_imag[:,i+1] = K.imag/np.abs(Z)
    K = 1/(1+1j*omega[:, np.newaxis]@tau[:, np.newaxis].T)
    A_re[:,1:len(tau)+1] = K.real/np.abs(Z[:, np.newaxis])
    A_imag[:,1:len(tau)+1] = K.imag/np.abs(Z[:, np.newaxis])

    #fit the parameters via pseudo-inverse
    pi_first_half = np.linalg.inv(np.dot(A_re.T, A_re) + np.dot(A_imag.T, A_imag))
    pi_second_half = np.dot(A_re.T, Z.real/np.abs(Z)) + np.dot(A_imag.T, Z.imag/np.abs(Z))
    elements = pi_first_half@pi_second_half

    if add_capacitor:
        mu = compute_mu(elements[1:-2])
    else:
        mu = compute_mu(elements[1:-1])

    if mu <= c:
        break

    if M == M_max:
        break

print(f'[linKK] t elapsed = {time.time() - t_init} s')

#test the fitting
f = fit_obj.freqs
circuit_string = f"s([R({[elements[0]]},{f.tolist()}),"
for Rk, tk in zip(elements[1:], tau):
        circuit_string += f"K({[Rk, tk]},{f.tolist()}),"

circuit_string += f"L({[elements[-1]]},{f.tolist()}),"
if elements.size == (tau.size + 3):
    circuit_string += f"C({[1 / elements[-2]]},{f.tolist()}),"

circuit_string = circuit_string.strip(',')
circuit_string += '])'
Z_linKK = eval(circuit_string, script_to_overwrite_local_impedancedotpy_files.circuit_elements)
Z_linKK = Z_linKK.astype("complex")
fit_residues_real, fit_residues_imag = compute_residues(fit_obj.z_meas, Z_linKK)

#plot
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
leg.append('ice measured')
ax.plot(Z_linKK.real, -Z_linKK.imag, color="tab:orange")
leg.append('Linear KK')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
leg = []
plt.plot(spec_ice_obj.freqs, fit_residues_real, '-o')
leg.append('Δ_re')
plt.plot(spec_ice_obj.freqs, fit_residues_imag, '-o')
leg.append('Δ_imag')
plt.ylabel('Δ [%]')
plt.legend(leg)
plt.xlabel("Frequency [Hz]")
plt.xscale('log')
plt.grid()
plt.show()
