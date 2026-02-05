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
longo_z_hat_real_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
longo_z_hat_imag_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
longo_NMSE_BFGS = np.zeros(shape=(len(longo_z_hat_real_BFGS,)))
longo_chi_square_BFGS = np.zeros(shape=(len(longo_z_hat_real_BFGS,)))
longo_z_hat_real_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
longo_z_hat_imag_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
longo_NMSE_NLLS = np.zeros(shape=(len(longo_z_hat_real_NLLS,)))
longo_chi_square_NLLS = np.zeros(shape=(len(longo_z_hat_real_NLLS,)))
longo_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["longo2020"]["n_params"]))

#Zurich2021
zurich_z_hat_real_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zurich_z_hat_imag_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zurich_NMSE_BFGS = np.zeros(shape=(len(zurich_z_hat_real_BFGS,)))
zurich_chi_square_BFGS = np.zeros(shape=(len(zurich_z_hat_real_BFGS,)))
zurich_z_hat_real_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zurich_z_hat_imag_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zurich_NMSE_NLLS = np.zeros(shape=(len(zurich_z_hat_real_NLLS,)))
zurich_chi_square_NLLS = np.zeros(shape=(len(zurich_z_hat_real_NLLS,)))
zurich_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["zurich2021"]["n_params"]))

#Zhang2024
zhang_z_hat_real_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zhang_z_hat_imag_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zhang_NMSE_BFGS = np.zeros(shape=(len(zhang_z_hat_real_BFGS,)))
zhang_chi_square_BFGS = np.zeros(shape=(len(zhang_z_hat_real_BFGS,)))
zhang_z_hat_real_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zhang_z_hat_imag_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
zhang_NMSE_NLLS = np.zeros(shape=(len(zhang_z_hat_real_NLLS,)))
zhang_chi_square_NLLS = np.zeros(shape=(len(zhang_z_hat_real_NLLS,)))
zhang_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["zhang2024"]["n_params"]))

#Yang2025
yang_z_hat_real_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
yang_z_hat_imag_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
yang_NMSE_BFGS = np.zeros(shape=(len(yang_z_hat_real_BFGS,)))
yang_chi_square_BFGS = np.zeros(shape=(len(yang_z_hat_real_BFGS,)))
yang_z_hat_real_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
yang_z_hat_imag_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
yang_NMSE_NLLS = np.zeros(shape=(len(yang_z_hat_real_NLLS,)))
yang_chi_square_NLLS = np.zeros(shape=(len(yang_z_hat_real_NLLS,)))
yang_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["yang2025"]["n_params"]))

#Fouquet2005
fouquet_z_hat_real_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
fouquet_z_hat_imag_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
fouquet_NMSE_BFGS = np.zeros(shape=(len(fouquet_z_hat_real_BFGS,)))
fouquet_chi_square_BFGS = np.zeros(shape=(len(fouquet_z_hat_real_BFGS,)))
fouquet_z_hat_real_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
fouquet_z_hat_imag_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
fouquet_NMSE_NLLS = np.zeros(shape=(len(fouquet_z_hat_real_NLLS,)))
fouquet_chi_square_NLLS = np.zeros(shape=(len(fouquet_z_hat_real_NLLS,)))
fouquet_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["fouquet2005"]["n_params"]))

#Awayssa2025
awayssa_z_hat_real_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
awayssa_z_hat_imag_BFGS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
awayssa_NMSE_BFGS = np.zeros(shape=(len(awayssa_z_hat_real_BFGS,)))
awayssa_chi_square_BFGS = np.zeros(shape=(len(awayssa_z_hat_real_BFGS,)))
awayssa_z_hat_real_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
awayssa_z_hat_imag_NLLS = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
awayssa_NMSE_NLLS = np.zeros(shape=(len(awayssa_z_hat_real_NLLS,)))
awayssa_chi_square_NLLS = np.zeros(shape=(len(awayssa_z_hat_real_NLLS,)))
awayssa_fitted_params = np.zeros(shape=(len(z_meas_real),fitting_handles["awayssa2025"]["n_params"]))

t_init = time.time()
for i in range(len(z_meas_real)):
    #compute all the models
    longo_obj = fitting_utils.EquivalentCircuit("Longo2020", [z_meas_real[i,0,:], z_meas_imag[i,0,:]], spec_ice_obj.freqs)
    longo_params_BFGS = longo_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="BFGS")
    try:
        longo_params_NLLS = longo_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="NLLS")
    except:
        continue

    zurich_obj = fitting_utils.EquivalentCircuit("Zurich2021", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    zurich_params_BFGS = zurich_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e-7, 1, 1e5, 1e3, 1e-8]), method="BFGS")
    try:
        zurich_params_NLLS = zurich_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e1, 1e-6, 1, 1e5, 1e3, 1e-8]), method="NLLS")
    except:
        continue

    zhang_obj = fitting_utils.EquivalentCircuit("Zhang2024", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    zhang_params_BFGS = zhang_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-8, 1, 1e7, 1e3, 1e2, 1e-8]), method="BFGS")
    try:
        zhang_params_NLLS = zhang_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-8, 1, 1e7, 1e2, 1e1, 1e-8]), method="NLLS")
    except:
        continue

    yang_obj = fitting_utils.EquivalentCircuit("Yang2025", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    yang_params_BFGS = yang_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e3, 1e4, 1e-8, 1, 1e4, 1e-8]), method="BFGS")
    try:
        yang_params_NLLS = yang_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e3, 1e4, 1e-8, 1, 1e4, 1e-8]), method="NLLS")
    except:
        continue

    fouquet_obj = fitting_utils.EquivalentCircuit("Fouquet2005", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    fouquet_params_BFGS = fouquet_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-4]), method="BFGS")
    try:
        fouquet_params_NLLS = fouquet_obj.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e2, 1e3, 1e-8, 1, 1e4, 1e-4]), method="NLLS")
    except:
        continue

    awayssa_obj = fitting_utils.EquivalentCircuit("Awayssa2025", [z_meas_real[i, 0, :], z_meas_imag[i, 0, :]], spec_ice_obj.freqs)
    awayssa_params_BFGS = awayssa_obj.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-8, 1e-8]), method="BFGS")
    awayssa_params_NLLS = awayssa_obj.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e2, 1e3, 1e-5, 1e-8, 1e-8]), method="NLLS")

    #store the modelled impedance
    longo_z_hat_real_BFGS[i,0,:] = longo_params_BFGS.opt_fit.real
    longo_z_hat_imag_BFGS[i,0,:] = longo_params_BFGS.opt_fit.imag
    longo_NMSE_BFGS[i] = longo_params_BFGS.nmse_score
    longo_chi_square_BFGS[i] = longo_params_BFGS.chi_square
    longo_z_hat_real_NLLS[i, 0, :] = longo_params_NLLS.opt_fit.real
    longo_z_hat_imag_NLLS[i, 0, :] = longo_params_NLLS.opt_fit.imag
    longo_NMSE_NLLS[i] = longo_params_NLLS.nmse_score
    longo_chi_square_NLLS[i] = longo_params_NLLS.chi_square
    longo_fitted_params[i,:] = longo_params_BFGS.opt_params_scaled

    zurich_z_hat_real_BFGS[i,0,:] = zurich_params_BFGS.opt_fit.real
    zurich_z_hat_imag_BFGS[i,0,:] = zurich_params_BFGS.opt_fit.imag
    zurich_NMSE_BFGS[i] = zurich_params_BFGS.nmse_score
    zurich_chi_square_BFGS[i] = zurich_params_BFGS.chi_square
    zurich_z_hat_real_NLLS[i, 0, :] = zurich_params_NLLS.opt_fit.real
    zurich_z_hat_imag_NLLS[i, 0, :] = zurich_params_NLLS.opt_fit.imag
    zurich_NMSE_NLLS[i] = zurich_params_NLLS.nmse_score
    zurich_chi_square_NLLS[i] = zurich_params_NLLS.chi_square
    zurich_fitted_params[i,:] = zurich_params_BFGS.opt_params_scaled

    zhang_z_hat_real_BFGS[i,0,:] = zhang_params_BFGS.opt_fit.real
    zhang_z_hat_imag_BFGS[i,0,:] = zhang_params_BFGS.opt_fit.imag
    zhang_NMSE_BFGS[i] = zhang_params_BFGS.nmse_score
    zhang_chi_square_BFGS[i] = zhang_params_BFGS.chi_square
    zhang_z_hat_real_NLLS[i, 0, :] = zhang_params_NLLS.opt_fit.real
    zhang_z_hat_imag_NLLS[i, 0, :] = zhang_params_NLLS.opt_fit.imag
    zhang_NMSE_NLLS[i] = zhang_params_NLLS.nmse_score
    zhang_chi_square_NLLS[i] = zhang_params_NLLS.chi_square
    zhang_fitted_params[i,:] = zhang_params_BFGS.opt_params_scaled

    yang_z_hat_real_BFGS[i,0,:] = yang_params_BFGS.opt_fit.real
    yang_z_hat_imag_BFGS[i,0,:] = yang_params_BFGS.opt_fit.imag
    yang_NMSE_BFGS[i] = yang_params_BFGS.nmse_score
    yang_chi_square_BFGS[i] = yang_params_BFGS.chi_square
    yang_z_hat_real_NLLS[i, 0, :] = yang_params_NLLS.opt_fit.real
    yang_z_hat_imag_NLLS[i, 0, :] = yang_params_NLLS.opt_fit.imag
    yang_NMSE_NLLS[i] = yang_params_NLLS.nmse_score
    yang_chi_square_NLLS[i] = yang_params_NLLS.chi_square
    yang_fitted_params[i,:] = yang_params_BFGS.opt_params_scaled

    fouquet_z_hat_real_BFGS[i,0,:] = fouquet_params_BFGS.opt_fit.real
    fouquet_z_hat_imag_BFGS[i,0,:] = fouquet_params_BFGS.opt_fit.imag
    fouquet_NMSE_BFGS[i] = fouquet_params_BFGS.nmse_score
    fouquet_chi_square_BFGS[i] = fouquet_params_BFGS.chi_square
    fouquet_z_hat_real_NLLS[i, 0, :] = fouquet_params_NLLS.opt_fit.real
    fouquet_z_hat_imag_NLLS[i, 0, :] = fouquet_params_NLLS.opt_fit.imag
    fouquet_NMSE_NLLS[i] = fouquet_params_NLLS.nmse_score
    fouquet_chi_square_NLLS[i] = fouquet_params_NLLS.chi_square
    fouquet_fitted_params[i, :] = fouquet_params_BFGS.opt_params_scaled

    awayssa_z_hat_real_BFGS[i,0,:] = awayssa_params_BFGS.opt_fit.real
    awayssa_z_hat_imag_BFGS[i,0,:] = awayssa_params_BFGS.opt_fit.imag
    awayssa_NMSE_BFGS[i] = awayssa_params_BFGS.nmse_score
    awayssa_chi_square_BFGS[i] = awayssa_params_BFGS.chi_square
    awayssa_z_hat_real_NLLS[i, 0, :] = awayssa_params_NLLS.opt_fit.real
    awayssa_z_hat_imag_NLLS[i, 0, :] = awayssa_params_NLLS.opt_fit.imag
    awayssa_NMSE_NLLS[i] = awayssa_params_NLLS.nmse_score
    awayssa_chi_square_NLLS[i] = awayssa_params_NLLS.chi_square
    awayssa_fitted_params[i,:] = awayssa_params_BFGS.opt_params_scaled

print(f'[impedance_fitting_freezing] finished computing all models in {time.time()-t_init}s')

#save the processed objects to a pickle file
processed_data = {
    "meas": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag, "freqs": spec_ice_obj.freqs, "timestamps": spec_ice_obj.human_timestamps},
    "longo": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag,
              "z_hat_real": longo_z_hat_real_BFGS, "z_hat_imag": longo_z_hat_imag_BFGS, "z_hat_real_NLLS": longo_z_hat_real_NLLS, "z_hat_imag_NLLS": longo_z_hat_imag_NLLS,
              "nmse": longo_NMSE_BFGS, "chi_square": longo_chi_square_BFGS, "nmse_NLLS": longo_NMSE_NLLS, "chi_square_NLLS": longo_chi_square_NLLS,
              "params": longo_fitted_params},
    "zurich": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag,
               "z_hat_real": zurich_z_hat_real_BFGS, "z_hat_imag": zurich_z_hat_imag_BFGS, "z_hat_real_NLLS": zurich_z_hat_real_NLLS, "z_hat_imag_NLLS": zurich_z_hat_imag_NLLS,
               "nmse": zurich_NMSE_BFGS, "chi_square": zurich_chi_square_BFGS, "nmse_NLLS": zurich_NMSE_NLLS, "chi_square_NLLS": zurich_chi_square_NLLS,
               "params": zurich_fitted_params},
    "zhang": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag,
              "z_hat_real": zhang_z_hat_real_BFGS, "z_hat_imag": zhang_z_hat_imag_BFGS, "z_hat_real_NLLS": zhang_z_hat_real_NLLS, "z_hat_imag_NLLS": zhang_z_hat_imag_NLLS,
              "nmse": zhang_NMSE_BFGS, "chi_square": zhang_chi_square_BFGS, "nmse_NLLS": zhang_NMSE_NLLS, "chi_square_NLLS": zhang_chi_square_NLLS,
              "params": zhang_fitted_params},
    "yang": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag,
             "z_hat_real": yang_z_hat_real_BFGS, "z_hat_imag": yang_z_hat_imag_BFGS, "z_hat_real_NLLS": yang_z_hat_real_NLLS, "z_hat_imag_NLLS": yang_z_hat_imag_NLLS,
             "nmse": yang_NMSE_BFGS, "chi_square": yang_chi_square_BFGS, "nmse_NLLS": yang_NMSE_NLLS, "chi_square_NLLS": yang_chi_square_NLLS,
             "params": yang_fitted_params},
    "fouquet": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag,
                "z_hat_real": fouquet_z_hat_real_BFGS, "z_hat_imag": fouquet_z_hat_imag_BFGS, "z_hat_real_NLLS": fouquet_z_hat_real_NLLS, "z_hat_imag_NLLS": fouquet_z_hat_imag_NLLS,
                "nmse": fouquet_NMSE_BFGS, "chi_square": fouquet_chi_square_BFGS, "nmse_NLLS": fouquet_NMSE_NLLS, "chi_square_NLLS": fouquet_chi_square_NLLS,
                "params": fouquet_fitted_params},
    "awayssa": {"z_meas_real": z_meas_real, "z_meas_imag": z_meas_imag,
                "z_hat_real": awayssa_z_hat_real_BFGS, "z_hat_imag": awayssa_z_hat_imag_BFGS, "z_hat_real_NLLS": awayssa_z_hat_real_NLLS, "z_hat_imag_NLLS": awayssa_z_hat_imag_NLLS,
                "nmse": awayssa_NMSE_BFGS, "chi_square": awayssa_chi_square_BFGS, "nmse_NLLS": awayssa_NMSE_NLLS, "chi_square_NLLS": awayssa_chi_square_NLLS,
                "params": awayssa_fitted_params}
}

filename = f"batch_{filename_flange.split('/')[-2]}.pkl"
with open(os.path.join(savedir, filename), 'wb') as handle:
    pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

