import numpy as np
from framework import file_lcr, fitting_utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os

#read the media files
spec_ice_obj = file_lcr.read('../data/testICE_30_01_26/c_ice.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)
spec_h2o_obj = file_lcr.read('../data/testICE_30_01_26/c1.csv', n_samples=3, sweeptype="cell", acquisition_mode="spectrum", aggregate=np.mean)

#fit the impedance for the ice files
longo_obj_ice= fitting_utils.EquivalentCircuit("Longo2020", spec_ice_obj, spec_ice_obj.freqs)
longo_params_ice = longo_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="BFGS", )

zurich_obj_ice = fitting_utils.EquivalentCircuit("Zurich2021", spec_ice_obj, spec_ice_obj.freqs)
zurich_params_ice = zurich_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e-7, 1, 1e5, 1e3, 1e-8]), method="BFGS", )

zhang_obj_ice = fitting_utils.EquivalentCircuit("Zhang2024", spec_ice_obj, spec_ice_obj.freqs)
zhang_params_ice = zhang_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-8, 1, 1e7, 1e3, 1e2, 1e-8]), method="BFGS", )

yang_obj_ice = fitting_utils.EquivalentCircuit("Yang2025", spec_ice_obj, spec_ice_obj.freqs)
yang_params_ice = yang_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e3, 1e4, 1e-8, 1, 1e4, 1e-8]), method="BFGS", )

fouquet_obj_ice = fitting_utils.EquivalentCircuit("Fouquet2005", spec_ice_obj, spec_ice_obj.freqs)
fouquet_params_ice = fouquet_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-4]), method="BFGS", )

awayssa_obj_ice = fitting_utils.EquivalentCircuit("Awayssa2025", spec_ice_obj, spec_ice_obj.freqs)
awayssa_params_ice = awayssa_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-8, 1e-8]), method="BFGS", )

#fit the impedance for the water files
longo_obj_h2o= fitting_utils.EquivalentCircuit("Longo2020", spec_h2o_obj, spec_h2o_obj.freqs)
longo_params_h2o = longo_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="BFGS", )

zurich_obj_h2o = fitting_utils.EquivalentCircuit("Zurich2021", spec_h2o_obj, spec_h2o_obj.freqs)
zurich_params_h2o = zurich_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e-7, 1, 1e5, 1e3, 1e-8]), method="BFGS", )

zhang_obj_h2o = fitting_utils.EquivalentCircuit("Zhang2024", spec_h2o_obj, spec_h2o_obj.freqs)
zhang_params_h2o = zhang_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-8, 1, 1e7, 1e3, 1e2, 1e-8]), method="BFGS", )

yang_obj_h2o = fitting_utils.EquivalentCircuit("Yang2025", spec_h2o_obj, spec_h2o_obj.freqs)
yang_params_h2o = yang_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e3, 1e4, 1e-8, 1, 1e4, 1e-8]), method="BFGS", )

fouquet_obj_h2o = fitting_utils.EquivalentCircuit("Fouquet2005", spec_h2o_obj, spec_h2o_obj.freqs)
fouquet_params_h2o = fouquet_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-4]), method="BFGS", )

awayssa_obj_h2o = fitting_utils.EquivalentCircuit("Awayssa2025", spec_h2o_obj, spec_h2o_obj.freqs)
awayssa_params_h2o = awayssa_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-8, 1e-8]), method="BFGS", )

#load the processed files
#see: ../impedance_fitting/batch_process_selected_models.py to generate the required .pkl file
batch_data_path = f'../data/batch_fit/batch_testICE_30_01_26.pkl'

if not os.path.isfile(batch_data_path):
    raise FileNotFoundError(f'[batch_impedance_fitting_freezing] Batch data file {batch_data_path} not found!')

with open(batch_data_path, 'rb') as handle:
    batch_data = pickle.load(handle)

#freerun sweep
idx_freerun = -1 #define which instant will be used as the freerun example
freqs = batch_data["meas"]["freqs"]
human_timestamps = batch_data["meas"]["timestamps"][idx_freerun]

#Longo2020
z_meas_real = batch_data["longo"]["z_meas_real"][idx_freerun,0,:]
z_meas_imag = batch_data["longo"]["z_meas_imag"][idx_freerun,0,:]
longo_z_hat_real = batch_data["longo"]["z_hat_real"][idx_freerun,0,:]
longo_z_hat_imag = batch_data["longo"]["z_hat_imag"][idx_freerun,0,:]
longo_NMSE = batch_data["longo"]["nmse"][idx_freerun]
longo_chi_square = batch_data["longo"]["chi_square"][idx_freerun]
longo_fit_params = batch_data["longo"]["params"][idx_freerun,:]

#Zurich2021
zurich_z_hat_real = batch_data["zurich"]["z_hat_real"][idx_freerun,0,:]
zurich_z_hat_imag = batch_data["zurich"]["z_hat_imag"][idx_freerun,0,:]
zurich_NMSE = batch_data["zurich"]["nmse"][idx_freerun]
zurich_chi_square = batch_data["zurich"]["chi_square"][idx_freerun]
zurich_fit_params = batch_data["zurich"]["params"][idx_freerun,:]

#Zhang2024
zhang_z_hat_real = batch_data["zhang"]["z_hat_real"][idx_freerun,0,:]
zhang_z_hat_imag = batch_data["zhang"]["z_hat_imag"][idx_freerun,0,:]
zhang_NMSE = batch_data["zhang"]["nmse"][idx_freerun]
zhang_chi_square = batch_data["zhang"]["chi_square"][idx_freerun]
zhang_fit_params = batch_data["zhang"]["params"][idx_freerun,:]

#Yang2025
yang_z_hat_real = batch_data["yang"]["z_hat_real"][idx_freerun,0,:]
yang_z_hat_imag = batch_data["yang"]["z_hat_imag"][idx_freerun,0,:]
yang_NMSE = batch_data["yang"]["nmse"][idx_freerun]
yang_chi_square = batch_data["yang"]["chi_square"][idx_freerun]
yang_fit_params = batch_data["yang"]["params"][idx_freerun,:]

#Fouquet2005
fouquet_z_hat_real = batch_data["fouquet"]["z_hat_real"][idx_freerun,0,:]
fouquet_z_hat_imag = batch_data["fouquet"]["z_hat_imag"][idx_freerun,0,:]
fouquet_NMSE = batch_data["fouquet"]["nmse"][idx_freerun]
fouquet_chi_square = batch_data["fouquet"]["chi_square"][idx_freerun]
fouquet_fit_params = batch_data["fouquet"]["params"][idx_freerun,:]

#Awayssa2025
awayssa_z_hat_real = batch_data["awayssa"]["z_hat_real"][idx_freerun,0,:]
awayssa_z_hat_imag = batch_data["awayssa"]["z_hat_imag"][idx_freerun,0,:]
awayssa_NMSE = batch_data["awayssa"]["nmse"][idx_freerun]
awayssa_chi_square = batch_data["awayssa"]["chi_square"][idx_freerun]
awayssa_fit_params = batch_data["awayssa"]["params"][idx_freerun,:]

#plot frame data
#animated plots
fig, axs = plt.subplots(nrows=1, ncols=6)

#longo2020
axs[0].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[0].plot(longo_z_hat_real, -longo_z_hat_imag, color="tab:orange", linestyle="dotted", label="Longo2020")
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel("Z'")
axs[0].set_ylabel("Z''")
axs[0].set_title(f"NMSE = {longo_NMSE} \n"
                 f"x² = {longo_chi_square}", fontsize=10)

#zurich2021
axs[1].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[1].plot(zurich_z_hat_real, -zurich_z_hat_imag, color="tab:orange", linestyle="dotted", label="Zurich2021")
axs[1].legend()
axs[1].grid()
axs[1].set_xlabel("Z'")
axs[1].set_title(f"NMSE = {zurich_NMSE} \n"
                 f"x² = {zurich_chi_square}", fontsize=10)
#Zhang2024
axs[2].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[2].plot(zhang_z_hat_real, -zhang_z_hat_imag, color="tab:orange", linestyle="dotted", label="Zhang2024")
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel("Z'")
axs[2].set_title(f"NMSE = {zhang_NMSE} \n"
                 f"x² = {zhang_chi_square}", fontsize=10)
#Yang2025
axs[3].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[3].plot(yang_z_hat_real, -yang_z_hat_imag, color="tab:orange", linestyle="dotted", label="Yang2025")
axs[3].legend()
axs[3].grid()
axs[3].set_xlabel("Z'")
axs[3].set_title(f"NMSE = {yang_NMSE} \n"
                 f"x² = {yang_chi_square}", fontsize=10)
#Fouquet2005
axs[4].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[4].plot(fouquet_z_hat_real, -fouquet_z_hat_imag, color="tab:orange", linestyle="dotted", label="Fouquet2005")
axs[4].legend()
axs[4].grid()
axs[4].set_xlabel("Z'")
axs[4].set_title(f"NMSE = {fouquet_NMSE} \n"
                 f"x² = {fouquet_chi_square}", fontsize=10)
#Awayssa2025
axs[5].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[5].plot(awayssa_z_hat_real, -awayssa_z_hat_imag, color="tab:orange", linestyle="dotted", label="Awayssa2025")
axs[5].legend()
axs[5].grid()
axs[5].set_xlabel("Z'")
axs[5].set_title(f"NMSE = {awayssa_NMSE} \n"
                 f"x² = {awayssa_chi_square}", fontsize=10)


#plot static ice vs. water
fig1, axs1 = plt.subplots(nrows=6, ncols=2, figsize=(10,10))

#longo2020
axs1[0,0].plot(longo_obj_h2o.z_meas_real, longo_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[0,0].plot(longo_params_h2o.opt_fit.real, -longo_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Longo2020")
axs1[0,0].legend()
axs1[0,0].grid()
axs1[0,0].set_xlabel("Z'")
axs1[0,0].set_ylabel("Z''")
# axs1[0,0].set_title(f"NMSE = {longo_params_h2o.nmse_score} \n"
#                  f"x² = {longo_params_h2o.chi_square}", fontsize=10)

axs1[0,1].plot(longo_obj_ice.z_meas_real, longo_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[0,1].plot(longo_params_ice.opt_fit.real, -longo_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Longo2020")
axs1[0,1].legend()
axs1[0,1].grid()
axs1[0,1].set_xlabel("Z'")
axs1[0,1].set_ylabel("Z''")
# axs1[0,1].set_title(f"NMSE = {longo_params_ice.nmse_score} \n"
#                  f"x² = {longo_params_ice.chi_square}", fontsize=10)

#zurich2021
axs1[1,0].plot(zurich_obj_h2o.z_meas_real, zurich_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[1,0].plot(zurich_params_h2o.opt_fit.real, -zurich_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zurich2021")
axs1[1,0].legend()
axs1[1,0].grid()
axs1[1,0].set_xlabel("Z'")
axs1[1,0].set_ylabel("Z''")
# axs1[1,0].set_title(f"NMSE = {zurich_params_h2o.nmse_score} \n"
#                  f"x² = {zurich_params_h2o.chi_square}", fontsize=10)

axs1[1,1].plot(zurich_obj_ice.z_meas_real, zurich_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[1,1].plot(zurich_params_ice.opt_fit.real, -zurich_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zurich2021")
axs1[1,1].legend()
axs1[1,1].grid()
axs1[1,1].set_xlabel("Z'")
axs1[1,1].set_ylabel("Z''")
# axs1[1,1].set_title(f"NMSE = {zurich_params_ice.nmse_score} \n"
#                  f"x² = {zurich_params_ice.chi_square}", fontsize=10)

#zhang2024
axs1[2,0].plot(zhang_obj_h2o.z_meas_real, zhang_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[2,0].plot(zhang_params_h2o.opt_fit.real, -zhang_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zhang2024")
axs1[2,0].legend()
axs1[2,0].grid()
axs1[2,0].set_xlabel("Z'")
axs1[2,0].set_ylabel("Z''")
# axs1[2,0].set_title(f"NMSE = {longo_params_h2o.nmse_score} \n"
#                  f"x² = {longo_params_h2o.chi_square}", fontsize=10)

axs1[2,1].plot(zhang_obj_ice.z_meas_real, zhang_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[2,1].plot(zhang_params_ice.opt_fit.real, -zhang_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zhang2024")
axs1[2,1].legend()
axs1[2,1].grid()
axs1[2,1].set_xlabel("Z'")
axs1[2,1].set_ylabel("Z''")
# axs1[2,1].set_title(f"NMSE = {zhang_params_ice.nmse_score} \n"
#                  f"x² = {zhang_params_ice.chi_square}", fontsize=10)

#yang2025
axs1[3,0].plot(yang_obj_h2o.z_meas_real, yang_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[3,0].plot(yang_params_h2o.opt_fit.real, -yang_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Yang2025")
axs1[3,0].legend()
axs1[3,0].grid()
axs1[3,0].set_xlabel("Z'")
axs1[3,0].set_ylabel("Z''")
# axs1[3,0].set_title(f"NMSE = {longo_params_h2o.nmse_score} \n"
#                  f"x² = {longo_params_h2o.chi_square}", fontsize=10)

axs1[3,1].plot(yang_obj_ice.z_meas_real, yang_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[3,1].plot(yang_params_ice.opt_fit.real, -yang_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Yang2025")
axs1[3,1].legend()
axs1[3,1].grid()
axs1[3,1].set_xlabel("Z'")
axs1[3,1].set_ylabel("Z''")
# axs1[3,1].set_title(f"NMSE = {yang_params_ice.nmse_score} \n"
#                  f"x² = {yang_params_ice.chi_square}", fontsize=10)

#fouquet2005
axs1[4,0].plot(fouquet_obj_h2o.z_meas_real, fouquet_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[4,0].plot(fouquet_params_h2o.opt_fit.real, -fouquet_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Fouquet2005")
axs1[4,0].legend()
axs1[4,0].grid()
axs1[4,0].set_xlabel("Z'")
axs1[4,0].set_ylabel("Z''")
# axs1[4,0].set_title(f"NMSE = {longo_params_h2o.nmse_score} \n"
#                  f"x² = {longo_params_h2o.chi_square}", fontsize=10)

axs1[4,1].plot(fouquet_obj_ice.z_meas_real, fouquet_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[4,1].plot(fouquet_params_ice.opt_fit.real, -fouquet_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Fouquet2005")
axs1[4,1].legend()
axs1[4,1].grid()
axs1[4,1].set_xlabel("Z'")
axs1[4,1].set_ylabel("Z''")
# axs1[4,1].set_title(f"NMSE = {fouquet_params_ice.nmse_score} \n"
#                  f"x² = {fouquet_params_ice.chi_square}", fontsize=10)

#Awayssa2025
axs1[5,0].plot(awayssa_obj_h2o.z_meas_real, awayssa_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[5,0].plot(awayssa_params_h2o.opt_fit.real, -awayssa_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="awayssa2005")
axs1[5,0].legend()
axs1[5,0].grid()
axs1[5,0].set_xlabel("Z'")
axs1[5,0].set_ylabel("Z''")
# axs1[5,0].set_title(f"NMSE = {longo_params_h2o.nmse_score} \n"
#                  f"x² = {longo_params_h2o.chi_square}", fontsize=10)

axs1[5,1].plot(awayssa_obj_ice.z_meas_real, awayssa_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[5,1].plot(awayssa_params_ice.opt_fit.real, -awayssa_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="awayssa2005")
axs1[5,1].legend()
axs1[5,1].grid()
axs1[5,1].set_xlabel("Z'")
axs1[5,1].set_ylabel("Z''")
# axs1[5,1].set_title(f"NMSE = {awayssa_params_ice.nmse_score} \n"
#                  f"x² = {awayssa_params_ice.chi_square}", fontsize=10)