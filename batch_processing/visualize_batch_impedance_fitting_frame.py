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
batch_data_path = f'../data/testICE_30_01_26/batchfit_testICE_30_01_26.pkl'

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
longo_z_hat_real_DLS = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,2]
longo_z_hat_imag_DLS = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,2]
longo_z_hat_real_simplex = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,3]
longo_z_hat_imag_simplex = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,3]
longo_NMSE = batch_data.fit_data["Longo2020"]["nmse"][idx_freerun,0]
longo_NMSE_NLLS = batch_data.fit_data["Longo2020"]["nmse"][idx_freerun,1]


#Zurich2021
zurich_z_hat_real_BFGS = batch_data.fit_data["Zurich2021"]["z_hat_real"][idx_freerun,:,0]
zurich_z_hat_imag_BFGS = batch_data.fit_data["Zurich2021"]["z_hat_imag"][idx_freerun,:,0]
zurich_z_hat_real_NLLS = batch_data.fit_data["Zurich2021"]["z_hat_real"][idx_freerun,:,1]
zurich_z_hat_imag_NLLS = batch_data.fit_data["Zurich2021"]["z_hat_imag"][idx_freerun,:,1]
zurich_z_hat_real_DLS = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,2]
zurich_z_hat_imag_DLS = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,2]
zurich_z_hat_real_simplex = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,3]
zurich_z_hat_imag_simplex = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,3]
zurich_NMSE = batch_data.fit_data["Zurich2021"]["nmse"][idx_freerun,0]
zurich_NMSE_NLLS = batch_data.fit_data["Zurich2021"]["nmse"][idx_freerun,1]

#Zhang2024
zhang_z_hat_real_BFGS = batch_data.fit_data["Zhang2024"]["z_hat_real"][idx_freerun,:,0]
zhang_z_hat_imag_BFGS = batch_data.fit_data["Zhang2024"]["z_hat_imag"][idx_freerun,:,0]
zhang_z_hat_real_NLLS = batch_data.fit_data["Zhang2024"]["z_hat_real"][idx_freerun,:,1]
zhang_z_hat_imag_NLLS = batch_data.fit_data["Zhang2024"]["z_hat_imag"][idx_freerun,:,1]
zhang_z_hat_real_DLS = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,2]
zhang_z_hat_imag_DLS = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,2]
zhang_z_hat_real_simplex = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,3]
zhang_z_hat_imag_simplex = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,3]
zhang_NMSE = batch_data.fit_data["Zhang2024"]["nmse"][idx_freerun,0]
zhang_NMSE_NLLS = batch_data.fit_data["Zhang2024"]["nmse"][idx_freerun,1]

#Yang2025
yang_z_hat_real_BFGS = batch_data.fit_data["Yang2025"]["z_hat_real"][idx_freerun,:,0]
yang_z_hat_imag_BFGS = batch_data.fit_data["Yang2025"]["z_hat_imag"][idx_freerun,:,0]
yang_z_hat_real_NLLS = batch_data.fit_data["Yang2025"]["z_hat_real"][idx_freerun,:,1]
yang_z_hat_imag_NLLS = batch_data.fit_data["Yang2025"]["z_hat_imag"][idx_freerun,:,1]
yang_z_hat_real_DLS = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,2]
yang_z_hat_imag_DLS = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,2]
yang_z_hat_real_simplex = batch_data.fit_data["Longo2020"]["z_hat_real"][idx_freerun,:,3]
yang_z_hat_imag_simplex = batch_data.fit_data["Longo2020"]["z_hat_imag"][idx_freerun,:,3]
yang_NMSE = batch_data.fit_data["Yang2025"]["nmse"][idx_freerun,0]
yang_NMSE_NLLS = batch_data.fit_data["Yang2025"]["nmse"][idx_freerun,1]

#plot frame data
#animated plots
fig, axs = plt.subplots(nrows=1, ncols=4)

#longo2020
axs[0].scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[0].plot(longo_z_hat_real_BFGS, -longo_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axs[0].plot(longo_z_hat_real_NLLS, -longo_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axs[0].plot(longo_z_hat_real_DLS, -longo_z_hat_imag_DLS, color="tab:purple", label="DLS")
axs[0].plot(longo_z_hat_real_simplex, -longo_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = axs[0].inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axins.plot(longo_z_hat_real_BFGS, -longo_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axins.plot(longo_z_hat_real_NLLS, -longo_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axins.plot(longo_z_hat_real_DLS, -longo_z_hat_imag_DLS, color="tab:purple", label="DLS")
axins.plot(longo_z_hat_real_simplex, -longo_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
axs[0].indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel("Z'")
axs[0].set_ylabel("Z''")
axs[0].set_title("Longo2020")

#zurich2021
axs[1].scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[1].plot(zurich_z_hat_real_BFGS, -zurich_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axs[1].plot(zurich_z_hat_real_NLLS, -zurich_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axs[1].plot(zurich_z_hat_real_DLS, -zurich_z_hat_imag_DLS, color="tab:purple", label="DLS")
axs[1].plot(zurich_z_hat_real_simplex, -zurich_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = axs[1].inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axins.plot(zurich_z_hat_real_BFGS, -zurich_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axins.plot(zurich_z_hat_real_NLLS, -zurich_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axins.plot(zurich_z_hat_real_DLS, -zurich_z_hat_imag_DLS, color="tab:purple", label="DLS")
axins.plot(zurich_z_hat_real_simplex, -zurich_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
axs[1].indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
axs[1].legend()
axs[1].grid()
axs[1].set_xlabel("Z'")
axs[1].set_title("Zurich2021")

#Zhang2024
axs[2].scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[2].plot(zhang_z_hat_real_BFGS, -zhang_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axs[2].plot(zhang_z_hat_real_NLLS, -zhang_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axs[2].plot(zhang_z_hat_real_DLS, -zhang_z_hat_imag_DLS, color="tab:purple", label="DLS")
axs[2].plot(zhang_z_hat_real_simplex, -zhang_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = axs[2].inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axins.plot(zhang_z_hat_real_BFGS, -zhang_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axins.plot(zhang_z_hat_real_NLLS, -zhang_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axins.plot(zhang_z_hat_real_DLS, -zhang_z_hat_imag_DLS, color="tab:purple", label="DLS")
axins.plot(zhang_z_hat_real_simplex, -zhang_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
axs[2].indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel("Z'")
axs[2].set_title("Zhang2024")

#Yang2025
axs[3].scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axs[3].plot(yang_z_hat_real_BFGS, -yang_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axs[3].plot(yang_z_hat_real_NLLS, -yang_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axs[3].plot(yang_z_hat_real_DLS, -yang_z_hat_imag_DLS, color="tab:purple", label="DLS")
axs[3].plot(yang_z_hat_real_simplex, -yang_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
x1, x2, y1, y2 = -1000, 10000, 1000, 12000
axins = axs[3].inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(z_meas_real, z_meas_imag, color="tab:blue", label="measured")
axins.plot(yang_z_hat_real_BFGS, -yang_z_hat_imag_BFGS, color="tab:orange", label="BFGS")
axins.plot(yang_z_hat_real_NLLS, -yang_z_hat_imag_NLLS, color="tab:green", label="NLLS")
axins.plot(yang_z_hat_real_DLS, -yang_z_hat_imag_DLS, color="tab:purple", label="DLS")
axins.plot(yang_z_hat_real_simplex, -yang_z_hat_imag_simplex, color="tab:red", label="Nelder-Mead Simplex")
axs[3].indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
axs[3].legend()
axs[3].grid()
axs[3].set_xlabel("Z'")
axs[3].set_title("Yang2025")
