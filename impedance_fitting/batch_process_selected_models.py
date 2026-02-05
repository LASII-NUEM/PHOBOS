import numpy as np
from framework import file_lcr, characterization_utils, fitting_utils
import time
import pickle
import os

#.pkl save arguments
savedir = f'../data/batch_fit'
if not os.path.exists(savedir):
    os.makedirs(savedir, 0o777)

#read the files
filename_flange = '../data/testICE_30_01_26/c_test.csv'
spec_ice_obj = file_lcr.read(filename_flange, n_samples=1, sweeptype="cell", acquisition_mode="freq", aggregate=np.mean)
z_meas_real, z_meas_imag = characterization_utils.complex_impedance(spec_ice_obj, spec_ice_obj.freqs)
z_meas = z_meas_real-1j*z_meas_imag
fitting_handles = fitting_utils.function_handlers

#window the signals from 100 Hz
freqs_mask = spec_ice_obj.freqs > 100
spec_ice_obj.freqs = spec_ice_obj.freqs[freqs_mask]
z_meas_real = z_meas_real[:,:,freqs_mask]
z_meas_imag = z_meas_imag[:,:,freqs_mask]
z_meas = z_meas[:,:,freqs_mask]

#Longo2020
longo_z_hat_real = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
longo_z_hat_imag = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
longo_NMSE = np.zeros(shape=(len(longo_z_hat_real,)))
longo_chi_square = np.zeros(shape=(len(longo_z_hat_real,)))
longo_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["longo2020"]["n_params"]))

#Zurich2021
zurich_z_hat_real = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zurich_z_hat_imag = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zurich_NMSE = np.zeros(shape=(len(zurich_z_hat_real,)))
zurich_chi_square = np.zeros(shape=(len(zurich_z_hat_real,)))
zurich_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["zurich2021"]["n_params"]))

#Zhang2024
zhang_z_hat_real = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zhang_z_hat_imag = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zhang_NMSE = np.zeros(shape=(len(zhang_z_hat_real,)))
zhang_chi_square = np.zeros(shape=(len(zhang_z_hat_real,)))
zhang_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["zhang2024"]["n_params"]))

#Yang2025
yang_z_hat_real = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
yang_z_hat_imag = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
yang_NMSE = np.zeros(shape=(len(yang_z_hat_real,)))
yang_chi_square = np.zeros(shape=(len(yang_z_hat_real,)))
yang_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["yang2025"]["n_params"]))

#Fouquet2005
fouquet_z_hat_real = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
fouquet_z_hat_imag = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
fouquet_NMSE = np.zeros(shape=(len(fouquet_z_hat_real,)))
fouquet_chi_square = np.zeros(shape=(len(fouquet_z_hat_real,)))
fouquet_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["fouquet2005"]["n_params"]))

#Awayssa2025
awayssa_z_hat_real = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
awayssa_z_hat_imag = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
awayssa_NMSE = np.zeros(shape=(len(awayssa_z_hat_real,)))
awayssa_chi_square = np.zeros(shape=(len(awayssa_z_hat_real,)))
awayssa_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["awayssa2025"]["n_params"]))

t_init = time.time()
for i in range(len(z_meas_real)):
    #compute all the models
    longo_obj = fitting_utils.EquivalentCircuit("Longo2020", [z_meas_real[i,0,:], z_meas_imag[i,0,:]], spec_ice_obj.freqs)
    longo_params = longo_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="BFGS")

    zurich_obj = fitting_utils.EquivalentCircuit("Zurich2021", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    zurich_params = zurich_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e-7, 1, 1e5, 1e3, 1e-8]), method="BFGS")

    zhang_obj = fitting_utils.EquivalentCircuit("Zhang2024", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    zhang_params = zhang_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-8, 1, 1e7, 1e3, 1e2, 1e-8]), method="BFGS")

    yang_obj = fitting_utils.EquivalentCircuit("Yang2025", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    yang_params = yang_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e3, 1e4, 1e-8, 1, 1e4, 1e-8]), method="BFGS")

    fouquet_obj = fitting_utils.EquivalentCircuit("Fouquet2005", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    fouquet_params = fouquet_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-4]), method="BFGS")

    awayssa_obj = fitting_utils.EquivalentCircuit("Awayssa2025", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    awayssa_params = awayssa_obj.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-8, 1e-8]), method="BFGS")

    #store the modelled impedance
    longo_z_hat_real[i,0,:] = longo_params.opt_fit.real
    longo_z_hat_imag[i,0,:] = longo_params.opt_fit.imag
    longo_NMSE[i] = longo_params.nmse_score
    longo_chi_square[i] = longo_params.chi_square
    longo_fitted_params[i,:] = longo_params.opt_params_scaled

    zurich_z_hat_real[i,0,:] = zurich_params.opt_fit.real
    zurich_z_hat_imag[i,0,:] = zurich_params.opt_fit.imag
    zurich_NMSE[i] = zurich_params.nmse_score
    zurich_chi_square[i] = zurich_params.chi_square
    zurich_fitted_params[i,:] = zurich_params.opt_params_scaled

    zhang_z_hat_real[i,0,:] = zhang_params.opt_fit.real
    zhang_z_hat_imag[i,0,:] = zhang_params.opt_fit.imag
    zhang_NMSE[i] = zhang_params.nmse_score
    zhang_chi_square[i] = zhang_params.chi_square
    zhang_fitted_params[i,:] = zhang_params.opt_params_scaled

    yang_z_hat_real[i,0,:] = yang_params.opt_fit.real
    yang_z_hat_imag[i,0,:] = yang_params.opt_fit.imag
    yang_NMSE[i] = yang_params.nmse_score
    yang_chi_square[i] = yang_params.chi_square
    yang_fitted_params[i,:] = yang_params.opt_params_scaled

    fouquet_z_hat_real[i,0,:] = fouquet_params.opt_fit.real
    fouquet_z_hat_imag[i,0,:] = fouquet_params.opt_fit.imag
    fouquet_NMSE[i] = fouquet_params.nmse_score
    fouquet_chi_square[i] = fouquet_params.chi_square
    fouquet_fitted_params[i, :] = fouquet_params.opt_params_scaled

    awayssa_z_hat_real[i,0,:] = awayssa_params.opt_fit.real
    awayssa_z_hat_imag[i,0,:] = awayssa_params.opt_fit.imag
    awayssa_NMSE[i] = awayssa_params.nmse_score
    awayssa_chi_square[i] = awayssa_params.chi_square
    awayssa_fitted_params[i,:] = awayssa_params.opt_params_scaled

print(f'[impedance_fitting_freezing] finished computing all models in {time.time()-t_init}s')

#save the processed objects to a pickle file
processed_data = {
    "meas": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "freqs": spec_ice_obj.freqs, "timestamps": spec_ice_obj.human_timestamps},
    "longo": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "z_hat_real": longo_z_hat_real, "z_hat_imag": longo_z_hat_imag, "nmse": longo_NMSE, "chi_square": longo_chi_square, "params": longo_fitted_params},
    "zurich": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "z_hat_real": zurich_z_hat_real, "z_hat_imag": zurich_z_hat_imag, "nmse": zurich_NMSE, "chi_square": zurich_chi_square, "params": zurich_fitted_params},
    "zhang": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "z_hat_real": zhang_z_hat_real, "z_hat_imag": zhang_z_hat_imag, "nmse": zhang_NMSE, "chi_square": zhang_chi_square, "params": zhang_fitted_params},
    "yang": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "z_hat_real": yang_z_hat_real, "z_hat_imag": yang_z_hat_imag, "nmse": yang_NMSE, "chi_square": yang_chi_square, "params": yang_fitted_params},
    "fouquet": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "z_hat_real": fouquet_z_hat_real, "z_hat_imag": fouquet_z_hat_imag, "nmse": fouquet_NMSE, "chi_square": fouquet_chi_square, "params": fouquet_fitted_params},
    "awayssa": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "z_hat_real": awayssa_z_hat_real, "z_hat_imag": awayssa_z_hat_imag, "nmse": awayssa_NMSE, "chi_square": awayssa_chi_square, "params": awayssa_fitted_params}
}

filename = f"batch_{filename_flange.split('/')[-2]}.pkl"
with open(os.path.join(savedir, filename), 'wb') as handle:
    pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

