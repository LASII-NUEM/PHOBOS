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
longo_params_ice_NLLS = longo_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e5, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="NLLS")

zurich_obj_ice = fitting_utils.EquivalentCircuit("Zurich2021", spec_ice_obj, spec_ice_obj.freqs)
zurich_params_ice = zurich_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e5, 1e-7, 1, 1e5, 1e3, 1e-8]), method="BFGS", )
zurich_params_ice_NLLS = zurich_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e5, 1e-7, 1, 1e5, 1e3, 1e-8]), method="NLLS")

zhang_obj_ice = fitting_utils.EquivalentCircuit("Zhang2024", spec_ice_obj, spec_ice_obj.freqs)
zhang_params_ice = zhang_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-9, 1, 1e6, 1e2, 1e2, 1e-8]), method="BFGS", )
zhang_params_ice_NLLS = zhang_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-9, 1, 1e6, 1e2, 1e2, 1e-8]), method="NLLS")

yang_obj_ice = fitting_utils.EquivalentCircuit("Yang2025", spec_ice_obj, spec_ice_obj.freqs)
yang_params_ice = yang_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-8]), method="BFGS", )
yang_params_ice_NLLS = yang_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-8]), method="NLLS")

fouquet_obj_ice = fitting_utils.EquivalentCircuit("Fouquet2005", spec_ice_obj, spec_ice_obj.freqs)
fouquet_params_ice = fouquet_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e3, 1e-8, 1, 1e4, 1e-4]), method="BFGS", )
fouquet_params_ice_NLLS = fouquet_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e3, 1e-8, 1, 1e4, 1e-4]), method="NLLS")

awayssa_obj_ice = fitting_utils.EquivalentCircuit("Awayssa2025", spec_ice_obj, spec_ice_obj.freqs)
awayssa_params_ice = awayssa_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-9, 1e-8]), method="BFGS", )
awayssa_params_ice_NLLS = awayssa_obj_ice.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-9, 1e-8]), method="NLLS")

#fit the impedance for the water files
longo_obj_h2o= fitting_utils.EquivalentCircuit("Longo2020", spec_h2o_obj, spec_h2o_obj.freqs)
longo_params_h2o = longo_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="BFGS", )
longo_params_h2o_NLLS = longo_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]), method="NLLS")

zurich_obj_h2o = fitting_utils.EquivalentCircuit("Zurich2021", spec_h2o_obj, spec_h2o_obj.freqs)
zurich_params_h2o = zurich_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e5, 1e-7, 1, 1e5, 1e3, 1e-8]), method="BFGS", )
zurich_params_h2o_NLLS = zurich_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e5, 1e-7, 1, 1e5, 1e3, 1e-8]), method="NLLS")

zhang_obj_h2o = fitting_utils.EquivalentCircuit("Zhang2024", spec_h2o_obj, spec_h2o_obj.freqs)
zhang_params_h2o = zhang_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-9, 1, 1e6, 1e2, 1e2, 1e-8]), method="BFGS", )
zhang_params_h2o_NLLS = zhang_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e-9, 1, 1e6, 1e2, 1e2, 1e-8]), method="NLLS")

yang_obj_h2o = fitting_utils.EquivalentCircuit("Yang2025", spec_h2o_obj, spec_h2o_obj.freqs)
yang_params_h2o = yang_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-8]), method="BFGS", )
yang_params_h2o_NLLS = yang_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e4, 1e-8, 1, 1e4, 1e-8]), method="NLLS")

fouquet_obj_h2o = fitting_utils.EquivalentCircuit("Fouquet2005", spec_h2o_obj, spec_h2o_obj.freqs)
fouquet_params_h2o = fouquet_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e3, 1e-8, 1, 1e4, 1e-4]), method="BFGS", )
fouquet_params_h2o_NLLS = fouquet_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1, 1]), np.array([1e4, 1e3, 1e-8, 1, 1e4, 1e-4]), method="NLLS")

awayssa_obj_h2o = fitting_utils.EquivalentCircuit("Awayssa2025", spec_h2o_obj, spec_h2o_obj.freqs)
awayssa_params_h2o = awayssa_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-9, 1e-8]), method="BFGS", )
awayssa_params_h2o_NLLS = awayssa_obj_h2o.fit_circuit(np.array([1, 1, 1, 1, 1]), np.array([1e3, 1e3, 1e-5, 1e-9, 1e-8]), method="NLLS")

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
longo_z_hat_real_BFGS = batch_data["longo"]["z_hat_real"][idx_freerun,0,:]
longo_z_hat_imag_BFGS = batch_data["longo"]["z_hat_imag"][idx_freerun,0,:]
longo_z_hat_real_NLLS = batch_data["longo"]["z_hat_real_NLLS"][idx_freerun,0,:]
longo_z_hat_imag_NLLS = batch_data["longo"]["z_hat_imag_NLLS"][idx_freerun,0,:]
longo_NMSE = batch_data["longo"]["nmse"][idx_freerun]
longo_NMSE_NLLS = batch_data["longo"]["nmse_NLLS"][idx_freerun]


#Zurich2021
zurich_z_hat_real_BFGS = batch_data["zurich"]["z_hat_real"][idx_freerun,0,:]
zurich_z_hat_imag_BFGS = batch_data["zurich"]["z_hat_imag"][idx_freerun,0,:]
zurich_z_hat_real_NLLS = batch_data["zurich"]["z_hat_real_NLLS"][idx_freerun,0,:]
zurich_z_hat_imag_NLLS = batch_data["zurich"]["z_hat_imag_NLLS"][idx_freerun,0,:]
zurich_NMSE = batch_data["zurich"]["nmse"][idx_freerun]
zurich_NMSE_NLLS = batch_data["zurich"]["nmse_NLLS"][idx_freerun]

#Zhang2024
zhang_z_hat_real_BFGS = batch_data["zhang"]["z_hat_real"][idx_freerun,0,:]
zhang_z_hat_imag_BFGS = batch_data["zhang"]["z_hat_imag"][idx_freerun,0,:]
zhang_z_hat_real_NLLS = batch_data["zhang"]["z_hat_real_NLLS"][idx_freerun,0,:]
zhang_z_hat_imag_NLLS = batch_data["zhang"]["z_hat_imag_NLLS"][idx_freerun,0,:]
zhang_NMSE = batch_data["zhang"]["nmse"][idx_freerun]
zhang_NMSE_NLLS = batch_data["zhang"]["nmse_NLLS"][idx_freerun]

#Yang2025
yang_z_hat_real_BFGS = batch_data["yang"]["z_hat_real"][idx_freerun,0,:]
yang_z_hat_imag_BFGS = batch_data["yang"]["z_hat_imag"][idx_freerun,0,:]
yang_z_hat_real_NLLS = batch_data["yang"]["z_hat_real_NLLS"][idx_freerun,0,:]
yang_z_hat_imag_NLLS = batch_data["yang"]["z_hat_imag_NLLS"][idx_freerun,0,:]
yang_NMSE = batch_data["yang"]["nmse"][idx_freerun]
yang_NMSE_NLLS = batch_data["yang"]["nmse_NLLS"][idx_freerun]

#Fouquet2005
fouquet_z_hat_real_BFGS = batch_data["fouquet"]["z_hat_real"][idx_freerun,0,:]
fouquet_z_hat_imag_BFGS = batch_data["fouquet"]["z_hat_imag"][idx_freerun,0,:]
fouquet_z_hat_real_NLLS = batch_data["fouquet"]["z_hat_real_NLLS"][idx_freerun,0,:]
fouquet_z_hat_imag_NLLS = batch_data["fouquet"]["z_hat_imag_NLLS"][idx_freerun,0,:]
fouquet_NMSE = batch_data["fouquet"]["nmse"][idx_freerun]
fouquet_NMSE_NLLS = batch_data["fouquet"]["nmse_NLLS"][idx_freerun]

#Awayssa2025
awayssa_z_hat_real_BFGS = batch_data["awayssa"]["z_hat_real"][idx_freerun,0,:]
awayssa_z_hat_imag_BFGS = batch_data["awayssa"]["z_hat_imag"][idx_freerun,0,:]
awayssa_z_hat_real_NLLS = batch_data["awayssa"]["z_hat_real_NLLS"][idx_freerun,0,:]
awayssa_z_hat_imag_NLLS = batch_data["awayssa"]["z_hat_imag_NLLS"][idx_freerun,0,:]
awayssa_NMSE = batch_data["awayssa"]["nmse"][idx_freerun]
awayssa_NMSE_NLLS = batch_data["awayssa"]["nmse_NLLS"][idx_freerun]

#plot frame data
#animated plots
fig, axs = plt.subplots(nrows=1, ncols=6)

#longo2020
axs[0].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[0].plot(longo_z_hat_real_BFGS, -longo_z_hat_imag_BFGS, color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")
axs[0].plot(longo_z_hat_real_NLLS, -longo_z_hat_imag_NLLS, color="tab:green", linestyle="dashed", label="Longo2020 (NLLS)")
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel("Z'")
axs[0].set_ylabel("Z''")
axs[0].set_title(f"NMSE BFGS = {longo_NMSE} \n"
                 f"NMSE NLLS = {longo_NMSE_NLLS}", fontsize=8)

#zurich2021
axs[1].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[1].plot(zurich_z_hat_real_BFGS, -zurich_z_hat_imag_BFGS, color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")
axs[1].plot(zurich_z_hat_real_NLLS, -zurich_z_hat_imag_NLLS, color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")
axs[1].legend()
axs[1].grid()
axs[1].set_xlabel("Z'")
axs[1].set_title(f"NMSE BFGS = {zurich_NMSE} \n"
                 f"NMSE NLLS = {zhang_NMSE_NLLS}", fontsize=8)

#Zhang2024
axs[2].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[2].plot(zhang_z_hat_real_BFGS, -zhang_z_hat_imag_BFGS, color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")
axs[2].plot(zhang_z_hat_real_NLLS, -zhang_z_hat_imag_NLLS, color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel("Z'")
axs[2].set_title(f"NMSE BFGS = {zhang_NMSE} \n"
                 f"NMSE NLLS = {zhang_NMSE_NLLS}", fontsize=8)

#Yang2025
axs[3].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[3].plot(yang_z_hat_real_BFGS, -yang_z_hat_imag_BFGS, color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")
axs[3].plot(yang_z_hat_real_NLLS, -yang_z_hat_imag_NLLS, color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")
axs[3].legend()
axs[3].grid()
axs[3].set_xlabel("Z'")
axs[3].set_title(f"NMSE BFGS = {yang_NMSE} \n"
                 f"NMSE NLLS = {yang_NMSE_NLLS}", fontsize=8)

#Fouquet2005
axs[4].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[4].plot(fouquet_z_hat_real_BFGS, -fouquet_z_hat_imag_BFGS, color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")
axs[4].plot(fouquet_z_hat_real_NLLS, -fouquet_z_hat_imag_NLLS, color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")
axs[4].legend()
axs[4].grid()
axs[4].set_xlabel("Z'")
axs[4].set_title(f"NMSE BFGS = {fouquet_NMSE} \n"
                 f"NMSE NLLS = {fouquet_NMSE_NLLS}", fontsize=8)

#Awayssa2025
axs[5].plot(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[5].plot(awayssa_z_hat_real_BFGS, -awayssa_z_hat_imag_BFGS, color="tab:orange", linestyle="dotted", label="Awayssa2025 (BFGS)")
axs[5].plot(awayssa_z_hat_real_NLLS, -awayssa_z_hat_imag_NLLS, color="tab:green", linestyle="dashed", label="Awayssa2025 (NLLS)")
axs[5].legend()
axs[5].grid()
axs[5].set_xlabel("Z'")
axs[5].set_title(f"NMSE BFGS = {awayssa_NMSE} \n"
                 f"NMSE NLLS = {awayssa_NMSE_NLLS}", fontsize=8)


#plot static ice vs. water
fig1, axs1 = plt.subplots(nrows=6, ncols=2, figsize=(10,10))

#longo2020
axs1[0,0].plot(longo_obj_h2o.z_meas_real, longo_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[0,0].plot(longo_params_h2o.opt_fit.real, -longo_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")
axs1[0,0].plot(longo_params_h2o_NLLS.opt_fit.real, -longo_params_h2o_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Longo2020 (NLLS)")
axs1[0,0].legend()
axs1[0,0].grid()
axs1[0,0].set_xlabel("Z'")
axs1[0,0].set_ylabel("Z''")
axs1[0,0].set_ylim([0,300])

axs1[0,1].plot(longo_obj_ice.z_meas_real, longo_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[0,1].plot(longo_params_ice.opt_fit.real, -longo_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")
axs1[0,1].plot(longo_params_ice_NLLS.opt_fit.real, -longo_params_ice_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Longo2020 (BFGS)")
axs1[0,1].legend()
axs1[0,1].grid()
axs1[0,1].set_xlabel("Z'")
axs1[0,1].set_ylabel("Z''")

#zurich2021
axs1[1,0].plot(zurich_obj_h2o.z_meas_real, zurich_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[1,0].plot(zurich_params_h2o.opt_fit.real, -zurich_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")
axs1[1,0].plot(zurich_params_h2o_NLLS.opt_fit.real, -zurich_params_h2o_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")
axs1[1,0].legend()
axs1[1,0].grid()
axs1[1,0].set_xlabel("Z'")
axs1[1,0].set_ylabel("Z''")

axs1[1,1].plot(zurich_obj_ice.z_meas_real, zurich_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[1,1].plot(zurich_params_ice.opt_fit.real, -zurich_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")
axs1[1,1].plot(zurich_params_ice.opt_fit.real, -zurich_params_ice.opt_fit.imag, color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")
axs1[1,1].legend()
axs1[1,1].grid()
axs1[1,1].set_xlabel("Z'")
axs1[1,1].set_ylabel("Z''")

#zhang2024
axs1[2,0].plot(zhang_obj_h2o.z_meas_real, zhang_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[2,0].plot(zhang_params_h2o.opt_fit.real, -zhang_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")
axs1[2,0].plot(zhang_params_h2o_NLLS.opt_fit.real, -zhang_params_h2o_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")
axs1[2,0].legend()
axs1[2,0].grid()
axs1[2,0].set_xlabel("Z'")
axs1[2,0].set_ylabel("Z''")

axs1[2,1].plot(zhang_obj_ice.z_meas_real, zhang_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[2,1].plot(zhang_params_ice.opt_fit.real, -zhang_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")
axs1[2,1].plot(zhang_params_ice_NLLS.opt_fit.real, -zhang_params_ice_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")
axs1[2,1].legend()
axs1[2,1].grid()
axs1[2,1].set_xlabel("Z'")
axs1[2,1].set_ylabel("Z''")

#yang2025
axs1[3,0].plot(yang_obj_h2o.z_meas_real, yang_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[3,0].plot(yang_params_h2o.opt_fit.real, -yang_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")
axs1[3,0].plot(yang_params_h2o_NLLS.opt_fit.real, -yang_params_h2o_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")
axs1[3,0].legend()
axs1[3,0].grid()
axs1[3,0].set_xlabel("Z'")
axs1[3,0].set_ylabel("Z''")

axs1[3,1].plot(yang_obj_ice.z_meas_real, yang_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[3,1].plot(yang_params_ice.opt_fit.real, -yang_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")
axs1[3,1].plot(yang_params_ice_NLLS.opt_fit.real, -yang_params_ice_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")
axs1[3,1].legend()
axs1[3,1].grid()
axs1[3,1].set_xlabel("Z'")
axs1[3,1].set_ylabel("Z''")

#fouquet2005
axs1[4,0].plot(fouquet_obj_h2o.z_meas_real, fouquet_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[4,0].plot(fouquet_params_h2o.opt_fit.real, -fouquet_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")
axs1[4,0].plot(fouquet_params_h2o_NLLS.opt_fit.real, -fouquet_params_h2o_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")
axs1[4,0].legend()
axs1[4,0].grid()
axs1[4,0].set_xlabel("Z'")
axs1[4,0].set_ylabel("Z''")

axs1[4,1].plot(fouquet_obj_ice.z_meas_real, fouquet_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[4,1].plot(fouquet_params_ice.opt_fit.real, -fouquet_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")
axs1[4,1].plot(fouquet_params_ice_NLLS.opt_fit.real, -fouquet_params_ice_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")
axs1[4,1].legend()
axs1[4,1].grid()
axs1[4,1].set_xlabel("Z'")
axs1[4,1].set_ylabel("Z''")

#Awayssa2025
axs1[5,0].plot(awayssa_obj_h2o.z_meas_real, awayssa_obj_h2o.z_meas_imag, color="tab:blue", label="water measured")
axs1[5,0].plot(awayssa_params_h2o.opt_fit.real, -awayssa_params_h2o.opt_fit.imag, color="tab:orange", linestyle="dotted", label="awayssa2005 (BFGS)")
axs1[5,0].plot(awayssa_params_h2o_NLLS.opt_fit.real, -awayssa_params_h2o_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="awayssa2005 (NLLS)")
axs1[5,0].legend()
axs1[5,0].grid()
axs1[5,0].set_xlabel("Z'")
axs1[5,0].set_ylabel("Z''")

axs1[5,1].plot(awayssa_obj_ice.z_meas_real, awayssa_obj_ice.z_meas_imag, color="tab:blue", label="ice measured")
axs1[5,1].plot(awayssa_params_ice.opt_fit.real, -awayssa_params_ice.opt_fit.imag, color="tab:orange", linestyle="dotted", label="awayssa2005 (BFGS)")
axs1[5,1].plot(awayssa_params_ice_NLLS.opt_fit.real, -awayssa_params_ice_NLLS.opt_fit.imag, color="tab:green", linestyle="dashed", label="awayssa2005 (NLLS)")
axs1[5,1].legend()
axs1[5,1].grid()
axs1[5,1].set_xlabel("Z'")
axs1[5,1].set_ylabel("Z''")
