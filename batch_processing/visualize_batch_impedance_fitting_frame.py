import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
import time

#load the processed files
#see: ./dump_batch_impedance_fitting.py to generate the required .pkl file
batch_data_path = f'../data/testICE_30_01_26/fittingbatch_testICE_30_01_26.pkl'

if not os.path.isfile(batch_data_path):
    raise FileNotFoundError(f'[batch_impedance_fitting_freezing] Fit batch data file {batch_data_path} not found!')

with open(batch_data_path, 'rb') as handle:
    batch_data = pickle.load(handle)

#freerun sweep
freqs = batch_data.media_obj.freqs
human_timestamps = batch_data.media_obj.human_timestamps
idx_freerun = 10 #define which instant will be used as the freerun example

#Longo2020
z_meas_real = batch_data.fit_data["Longo2020"]["z_meas_real"][idx_freerun,:]
z_meas_imag = batch_data.fit_data["Longo2020"]["z_meas_imag"][idx_freerun,:]
longo_z_hat_real_BFGS = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,0]
longo_z_hat_imag_BFGS = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,0]
longo_z_hat_real_NLLS = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,1]
longo_z_hat_imag_NLLS = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,1]
longo_NMSE = batch_data.fit_data["Longo2020"]["nmse"][idx_freerun,0]
longo_NMSE_NLLS = batch_data.fit_data["Longo2020"]["nmse"][idx_freerun,1]


#Zurich2021
zurich_z_hat_real_BFGS = batch_data.fit_data["Zurich2021"]["z_hat_real"][idx_freerun,:,0]
zurich_z_hat_imag_BFGS = batch_data.fit_data["Zurich2021"]["z_hat_imag"][idx_freerun,:,0]
zurich_z_hat_real_NLLS = batch_data.fit_data["Zurich2021"]["z_hat_real"][idx_freerun,:,1]
zurich_z_hat_imag_NLLS = batch_data.fit_data["Zurich2021"]["z_hat_imag"][idx_freerun,:,1]
zurich_NMSE = batch_data.fit_data["Zurich2021"]["nmse"][idx_freerun,0]
zurich_NMSE_NLLS = batch_data.fit_data["Zurich2021"]["nmse"][idx_freerun,1]

#Zhang2024
zhang_z_hat_real_BFGS = batch_data.fit_data["Zhang2024"]["z_hat_real"][idx_freerun,:,0]
zhang_z_hat_imag_BFGS = batch_data.fit_data["Zhang2024"]["z_hat_imag"][idx_freerun,:,0]
zhang_z_hat_real_NLLS = batch_data.fit_data["Zhang2024"]["z_hat_real"][idx_freerun,:,1]
zhang_z_hat_imag_NLLS = batch_data.fit_data["Zhang2024"]["z_hat_imag"][idx_freerun,:,1]
zhang_NMSE = batch_data.fit_data["Zhang2024"]["nmse"][idx_freerun,0]
zhang_NMSE_NLLS = batch_data.fit_data["Zhang2024"]["nmse"][idx_freerun,1]

#Yang2025
yang_z_hat_real_BFGS = batch_data.fit_data["Yang2025"]["z_hat_real"][idx_freerun,:,0]
yang_z_hat_imag_BFGS = batch_data.fit_data["Yang2025"]["z_hat_imag"][idx_freerun,:,0]
yang_z_hat_real_NLLS = batch_data.fit_data["Yang2025"]["z_hat_real"][idx_freerun,:,1]
yang_z_hat_imag_NLLS = batch_data.fit_data["Yang2025"]["z_hat_imag"][idx_freerun,:,1]
yang_NMSE = batch_data.fit_data["Yang2025"]["nmse"][idx_freerun,0]
yang_NMSE_NLLS = batch_data.fit_data["Yang2025"]["nmse"][idx_freerun,1]

#Fouquet2005
fouquet_z_hat_real_BFGS = batch_data.fit_data["Fouquet2005"]["z_hat_real"][idx_freerun,:,0]
fouquet_z_hat_imag_BFGS = batch_data.fit_data["Fouquet2005"]["z_hat_imag"][idx_freerun,:,0]
fouquet_z_hat_real_NLLS = batch_data.fit_data["Fouquet2005"]["z_hat_real"][idx_freerun,:,1]
fouquet_z_hat_imag_NLLS = batch_data.fit_data["Fouquet2005"]["z_hat_imag"][idx_freerun,:,1]
fouquet_NMSE = batch_data.fit_data["Fouquet2005"]["nmse"][idx_freerun,0]
fouquet_NMSE_NLLS = batch_data.fit_data["Fouquet2005"]["nmse"][idx_freerun,1]

#Awayssa2025
awayssa_z_hat_real_BFGS = batch_data.fit_data["Awayssa2025"]["z_hat_real"][idx_freerun,:,0]
awayssa_z_hat_imag_BFGS = batch_data.fit_data["Awayssa2025"]["z_hat_imag"][idx_freerun,:,0]
awayssa_z_hat_real_NLLS = batch_data.fit_data["Awayssa2025"]["z_hat_real"][idx_freerun,:,1]
awayssa_z_hat_imag_NLLS = batch_data.fit_data["Awayssa2025"]["z_hat_imag"][idx_freerun,:,1]
awayssa_NMSE = batch_data.fit_data["Awayssa2025"]["nmse"][idx_freerun,0]
awayssa_NMSE_NLLS = batch_data.fit_data["Awayssa2025"]["nmse"][idx_freerun,1]

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