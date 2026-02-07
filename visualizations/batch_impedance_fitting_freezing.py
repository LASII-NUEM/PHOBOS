import numpy as np
from framework import file_lcr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os
import time

#load the processed files
#see: ../impedance_fitting/batch_process_selected_models.py to generate the required .pkl file
batch_data_path = f'../data/batch_fit/batch_testICE_30_01_26.pkl'

if not os.path.isfile(batch_data_path):
    raise FileNotFoundError(f'[batch_impedance_fitting_freezing] Batch data file {batch_data_path} not found!')

with open(batch_data_path, 'rb') as handle:
    batch_data = pickle.load(handle)

#freerun sweep
freqs = batch_data["meas"]["freqs"]
human_timestamps = batch_data["meas"]["timestamps"]

#Longo2020
z_meas_real = batch_data["longo"]["z_meas_real"]
z_meas_imag = batch_data["longo"]["z_meas_imag"]
longo_z_hat_real_BFGS = batch_data["longo"]["z_hat_real"]
longo_z_hat_imag_BFGS = batch_data["longo"]["z_hat_imag"]
longo_z_hat_real_NLLS = batch_data["longo"]["z_hat_real_NLLS"]
longo_z_hat_imag_NLLS = batch_data["longo"]["z_hat_imag_NLLS"]
longo_NMSE = batch_data["longo"]["nmse"]
longo_NMSE_NLLS = batch_data["longo"]["nmse_NLLS"]


#Zurich2021
zurich_z_hat_real_BFGS = batch_data["zurich"]["z_hat_real"]
zurich_z_hat_imag_BFGS = batch_data["zurich"]["z_hat_imag"]
zurich_z_hat_real_NLLS = batch_data["zurich"]["z_hat_real_NLLS"]
zurich_z_hat_imag_NLLS = batch_data["zurich"]["z_hat_imag_NLLS"]
zurich_NMSE = batch_data["zurich"]["nmse"]
zurich_NMSE_NLLS = batch_data["zurich"]["nmse_NLLS"]

#Zhang2024
zhang_z_hat_real_BFGS = batch_data["zhang"]["z_hat_real"]
zhang_z_hat_imag_BFGS = batch_data["zhang"]["z_hat_imag"]
zhang_z_hat_real_NLLS = batch_data["zhang"]["z_hat_real_NLLS"]
zhang_z_hat_imag_NLLS = batch_data["zhang"]["z_hat_imag_NLLS"]
zhang_NMSE = batch_data["zhang"]["nmse"]
zhang_NMSE_NLLS = batch_data["zhang"]["nmse_NLLS"]

#Yang2025
yang_z_hat_real_BFGS = batch_data["yang"]["z_hat_real"]
yang_z_hat_imag_BFGS = batch_data["yang"]["z_hat_imag"]
yang_z_hat_real_NLLS = batch_data["yang"]["z_hat_real_NLLS"]
yang_z_hat_imag_NLLS = batch_data["yang"]["z_hat_imag_NLLS"]
yang_NMSE = batch_data["yang"]["nmse"]
yang_NMSE_NLLS = batch_data["yang"]["nmse_NLLS"]

#Fouquet2005
fouquet_z_hat_real_BFGS = batch_data["fouquet"]["z_hat_real"]
fouquet_z_hat_imag_BFGS = batch_data["fouquet"]["z_hat_imag"]
fouquet_z_hat_real_NLLS = batch_data["fouquet"]["z_hat_real_NLLS"]
fouquet_z_hat_imag_NLLS = batch_data["fouquet"]["z_hat_imag_NLLS"]
fouquet_NMSE = batch_data["fouquet"]["nmse"]
fouquet_NMSE_NLLS = batch_data["fouquet"]["nmse_NLLS"]

#Awayssa2025
awayssa_z_hat_real_BFGS = batch_data["awayssa"]["z_hat_real"]
awayssa_z_hat_imag_BFGS = batch_data["awayssa"]["z_hat_imag"]
awayssa_z_hat_real_NLLS = batch_data["awayssa"]["z_hat_real_NLLS"]
awayssa_z_hat_imag_NLLS = batch_data["awayssa"]["z_hat_imag_NLLS"]
awayssa_NMSE = batch_data["awayssa"]["nmse"]
awayssa_NMSE_NLLS = batch_data["awayssa"]["nmse_NLLS"]

#animated plots
fig, axs = plt.subplots(nrows=6, ncols=2)

#longo2020
longo_meas = axs[0,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
longo_model = axs[0,0].plot(longo_z_hat_real_BFGS[0,0,:], -longo_z_hat_imag_BFGS[0,0,:], color="tab:orange", linestyle="dotted", label="Longo2020 (BFGS)")[0]
longo_model_NLLS = axs[0,0].plot(longo_z_hat_real_NLLS[0,0,:], -longo_z_hat_imag_NLLS[0,0,:], color="tab:green", linestyle="dashed", label="Longo2020 (NLLS)")[0]
axs[0,0].legend()
axs[0,0].grid()
axs[0,0].set_title("Nyquist plot")
axs[0,0].set_xlabel("Z'")
axs[0,0].set_ylabel("Z''")
longo_nmse = axs[0,1].plot(human_timestamps[0], longo_NMSE[0], color="tab:orange", label="BFGS")[0]
longo_nmse_NLLS = axs[0,1].plot(human_timestamps[0], longo_NMSE_NLLS[0], color="tab:green", label="NLLS")[0]
axs[0,1].legend()
axs[0,1].grid()
axs[0,1].set_xlabel("Timestamp")
axs[0,1].set_title("NMSE")

#zurich2021
zurich_meas = axs[1,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
zurich_model = axs[1,0].plot(zurich_z_hat_real_BFGS[0,0,:], -zurich_z_hat_imag_BFGS[0,0,:], color="tab:orange", linestyle="dotted", label="Zurich2021 (BFGS)")[0]
zurich_model_NLLS = axs[1,0].plot(zurich_z_hat_real_NLLS[0,0,:], -zurich_z_hat_imag_NLLS[0,0,:], color="tab:green", linestyle="dashed", label="Zurich2021 (NLLS)")[0]
axs[1,0].legend()
axs[1,0].grid()
axs[1,0].set_xlabel("Z'")
axs[1,0].set_ylabel("Z''")
zurich_nmse = axs[1,1].plot(human_timestamps[0], zurich_NMSE[0], color="tab:orange", label="BFGS")[0]
zurich_nmse_NLLS = axs[1,1].plot(human_timestamps[0], zurich_NMSE_NLLS[0], color="tab:green", label="NLLS")[0]
axs[1,1].legend()
axs[1,1].grid()
axs[1,1].set_xlabel("Timestamp")

#Zhang2024
zhang_meas = axs[2,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
zhang_model = axs[2,0].plot(zhang_z_hat_real_BFGS[0,0,:], -zhang_z_hat_imag_BFGS[0,0,:], color="tab:orange", linestyle="dotted", label="Zhang2024 (BFGS)")[0]
zhang_model_NLLS = axs[2,0].plot(zhang_z_hat_real_NLLS[0,0,:], -zhang_z_hat_imag_NLLS[0,0,:], color="tab:green", linestyle="dashed", label="Zhang2024 (NLLS)")[0]
axs[2,0].legend()
axs[2,0].grid()
axs[2,0].set_xlabel("Z'")
axs[2,0].set_ylabel("Z''")
zhang_nmse = axs[2,1].plot(human_timestamps[0], zhang_NMSE[0], color="tab:orange", label="BFGS")[0]
zhang_nmse_NLLS = axs[2,1].plot(human_timestamps[0], zhang_NMSE_NLLS[0], color="tab:green", label="NLLS")[0]
axs[2,1].legend()
axs[2,1].grid()
axs[2,1].set_xlabel("Timestamp")

#Yang2025
yang_meas = axs[3,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
yang_model = axs[3,0].plot(yang_z_hat_real_BFGS[0,0,:], -yang_z_hat_imag_BFGS[0,0,:], color="tab:orange", linestyle="dotted", label="Yang2025 (BFGS)")[0]
yang_model_NLLS = axs[3,0].plot(yang_z_hat_real_NLLS[0,0,:], -yang_z_hat_imag_NLLS[0,0,:], color="tab:green", linestyle="dashed", label="Yang2025 (NLLS)")[0]
axs[3,0].legend()
axs[3,0].grid()
axs[3,0].set_xlabel("Z'")
axs[3,0].set_ylabel("Z''")
yang_nmse = axs[3,1].plot(human_timestamps[0], yang_NMSE[0], color="tab:orange", label="BFGS")[0]
yang_nmse_NLLS = axs[3,1].plot(human_timestamps[0], yang_NMSE_NLLS[0], color="tab:green", label="NLLS")[0]
axs[3,1].legend()
axs[3,1].grid()
axs[3,1].set_xlabel("Timestamp")

#Fouquet2005
fouquet_meas = axs[4,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
fouquet_model = axs[4,0].plot(fouquet_z_hat_real_BFGS[0,0,:], -fouquet_z_hat_imag_BFGS[0,0,:], color="tab:orange", linestyle="dotted", label="Fouquet2005 (BFGS)")[0]
fouquet_model_NLLS = axs[4,0].plot(fouquet_z_hat_real_NLLS[0,0,:], -fouquet_z_hat_imag_NLLS[0,0,:], color="tab:green", linestyle="dashed", label="Fouquet2005 (NLLS)")[0]
axs[4,0].legend()
axs[4,0].grid()
axs[4,0].set_xlabel("Z'")
axs[4,0].set_ylabel("Z''")
fouquet_nmse = axs[4,1].plot(human_timestamps[0], fouquet_NMSE[0], color="tab:orange", label="BFGS")[0]
fouquet_nmse_NLLS = axs[4,1].plot(human_timestamps[0], fouquet_NMSE_NLLS[0], color="tab:green", label="NLLS")[0]
axs[4,1].legend()
axs[4,1].grid()
axs[4,1].set_xlabel("Timestamp")

#Awayssa2025
awayssa_meas = axs[5,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
awayssa_model = axs[5,0].plot(awayssa_z_hat_real_BFGS[0,0,:], -awayssa_z_hat_imag_BFGS[0,0,:], color="tab:orange", linestyle="dotted", label="Awayssa2025")[0]
awayssa_model_NLLS = axs[5,0].plot(awayssa_z_hat_real_NLLS[0,0,:], -awayssa_z_hat_imag_NLLS[0,0,:], color="tab:green", linestyle="dashed", label="Awayssa2025 (NLLS)")[0]
axs[5,0].legend()
axs[5,0].grid()
axs[5,0].set_xlabel("Z'")
axs[5,0].set_ylabel("Z''")
awayssa_nmse = axs[5,1].plot(human_timestamps[0], awayssa_NMSE[0], color="tab:orange", label="BFGS")[0]
awayssa_nmse_NLLS = axs[5,1].plot(human_timestamps[0], awayssa_NMSE_NLLS[0], color="tab:green", label="NLLS")[0]
axs[5,1].legend()
axs[5,1].grid()
axs[5,1].set_xlabel("Timestamp")

def update_nyquist(frame):
    #update plots
    frame += 1
    limits = 10
    window = 3
    if frame < 4:
        window = 0
    fig.suptitle(f'timestamp = {human_timestamps[frame]}')

    #Longo2020
    longo_meas.set_xdata(z_meas_real[frame,0,:])
    longo_meas.set_ydata(z_meas_imag[frame,0,:])
    longo_model.set_xdata(longo_z_hat_real_BFGS[frame,0,:])
    longo_model.set_ydata(-longo_z_hat_imag_BFGS[frame,0,:])
    longo_model_NLLS.set_xdata(longo_z_hat_real_NLLS[frame, 0, :])
    longo_model_NLLS.set_ydata(-longo_z_hat_imag_NLLS[frame, 0, :])
    longo_nmse.set_xdata(human_timestamps[:frame+1])
    longo_nmse.set_ydata(longo_NMSE[:frame+1])
    longo_nmse_NLLS.set_xdata(human_timestamps[:frame + 1])
    longo_nmse_NLLS.set_ydata(longo_NMSE_NLLS[:frame + 1])
    axs[0,0].set_xlim([np.min(z_meas_real[frame,0,:])-limits, np.max(z_meas_real[frame,0,:])+limits])
    axs[0,0].set_ylim([np.min(z_meas_imag[frame,0,:])-limits, np.max(z_meas_imag[frame,0,:])+limits])
    axs[0,1].set_xlim([human_timestamps[frame-window-1], human_timestamps[frame]])
    axs[0,1].set_ylim([np.min([np.min(longo_NMSE[frame-window-1:frame]),np.min(longo_NMSE_NLLS[frame-window-1:frame])]),
                       np.max([np.max(longo_NMSE[frame-window-1:frame]),np.max(longo_NMSE_NLLS[frame-window-1:frame])]) +
                       np.max([np.max(longo_NMSE[frame-window-1:frame]),np.max(longo_NMSE_NLLS[frame-window-1:frame])])])

    #Zurich2021
    zurich_meas.set_xdata(z_meas_real[frame, 0, :])
    zurich_meas.set_ydata(z_meas_imag[frame, 0, :])
    zurich_model.set_xdata(zurich_z_hat_real_BFGS[frame, 0, :])
    zurich_model.set_ydata(-zurich_z_hat_imag_BFGS[frame, 0, :])
    zurich_model_NLLS.set_xdata(zurich_z_hat_real_NLLS[frame, 0, :])
    zurich_model_NLLS.set_ydata(-zurich_z_hat_imag_NLLS[frame, 0, :])
    zurich_nmse.set_xdata(human_timestamps[:frame + 1])
    zurich_nmse.set_ydata(zurich_NMSE[:frame + 1])
    zurich_nmse_NLLS.set_xdata(human_timestamps[:frame + 1])
    zurich_nmse_NLLS.set_ydata(zurich_NMSE_NLLS[:frame + 1])
    axs[1, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[1, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[1, 1].set_xlim([human_timestamps[frame-window-1], human_timestamps[frame]])
    axs[1,1].set_ylim([np.min([np.min(zurich_NMSE[frame-window-1:frame]),np.min(zurich_NMSE_NLLS[frame-window-1:frame])]),
                       np.max([np.max(zurich_NMSE[frame-window-1:frame]),np.max(zurich_NMSE_NLLS[frame-window-1:frame])]) +
                       np.max([np.max(zurich_NMSE[frame-window-1:frame]),np.max(zurich_NMSE_NLLS[frame-window-1:frame])])])

    #Zhang2024
    zhang_meas.set_xdata(z_meas_real[frame, 0, :])
    zhang_meas.set_ydata(z_meas_imag[frame, 0, :])
    zhang_model.set_xdata(zhang_z_hat_real_BFGS[frame, 0, :])
    zhang_model.set_ydata(-zhang_z_hat_imag_BFGS[frame, 0, :])
    zhang_model_NLLS.set_xdata(zhang_z_hat_real_NLLS[frame, 0, :])
    zhang_model_NLLS.set_ydata(-zhang_z_hat_imag_NLLS[frame, 0, :])
    zhang_nmse.set_xdata(human_timestamps[:frame + 1])
    zhang_nmse.set_ydata(zhang_NMSE[:frame + 1])
    zhang_nmse_NLLS.set_xdata(human_timestamps[:frame + 1])
    zhang_nmse_NLLS.set_ydata(zhang_NMSE_NLLS[:frame + 1])
    axs[2, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[2, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[2, 1].set_xlim([human_timestamps[frame-window-1], human_timestamps[frame]])
    axs[2,1].set_ylim([np.min([np.min(zhang_NMSE[frame-window-1:frame]),np.min(zhang_NMSE_NLLS[frame-window-1:frame])]),
                       np.max([np.max(zhang_NMSE[frame-window-1:frame]),np.max(zhang_NMSE_NLLS[frame-window-1:frame])]) +
                       np.max([np.max(zhang_NMSE[frame-window-1:frame]),np.max(zhang_NMSE_NLLS[frame-window-1:frame])])])

    #Yang2025
    yang_meas.set_xdata(z_meas_real[frame, 0, :])
    yang_meas.set_ydata(z_meas_imag[frame, 0, :])
    yang_model.set_xdata(yang_z_hat_real_BFGS[frame, 0, :])
    yang_model.set_ydata(-yang_z_hat_imag_BFGS[frame, 0, :])
    yang_model_NLLS.set_xdata(yang_z_hat_real_NLLS[frame, 0, :])
    yang_model_NLLS.set_ydata(-yang_z_hat_imag_NLLS[frame, 0, :])
    yang_nmse.set_xdata(human_timestamps[:frame + 1])
    yang_nmse.set_ydata(yang_NMSE[:frame + 1])
    yang_nmse_NLLS.set_xdata(human_timestamps[:frame + 1])
    yang_nmse_NLLS.set_ydata(yang_NMSE_NLLS[:frame + 1])
    axs[3, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[3, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[3, 1].set_xlim([human_timestamps[frame-window-1], human_timestamps[frame]])
    axs[3, 1].set_ylim([np.min([np.min(yang_NMSE[frame-window-1:frame]),np.min(yang_NMSE_NLLS[frame-window-1:frame])]),
                        np.max([np.max(yang_NMSE[frame-window-1:frame]),np.max(yang_NMSE_NLLS[frame-window-1:frame])]) +
                        np.max([np.max(yang_NMSE[frame-window-1:frame]),np.max(yang_NMSE_NLLS[frame-window-1:frame])])])

    #Fouquet2005
    fouquet_meas.set_xdata(z_meas_real[frame, 0, :])
    fouquet_meas.set_ydata(z_meas_imag[frame, 0, :])
    fouquet_model.set_xdata(fouquet_z_hat_real_BFGS[frame, 0, :])
    fouquet_model.set_ydata(-fouquet_z_hat_imag_BFGS[frame, 0, :])
    fouquet_model_NLLS.set_xdata(fouquet_z_hat_real_NLLS[frame, 0, :])
    fouquet_model_NLLS.set_ydata(-fouquet_z_hat_imag_NLLS[frame, 0, :])
    fouquet_nmse.set_xdata(human_timestamps[:frame + 1])
    fouquet_nmse.set_ydata(fouquet_NMSE[:frame + 1])
    fouquet_nmse_NLLS.set_xdata(human_timestamps[:frame + 1])
    fouquet_nmse_NLLS.set_ydata(fouquet_NMSE_NLLS[:frame + 1])
    axs[4, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[4, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[4, 1].set_xlim([human_timestamps[frame-window-1], human_timestamps[frame]])
    axs[4, 1].set_ylim([np.min([np.min(fouquet_NMSE[frame-window-1:frame]),np.min(fouquet_NMSE_NLLS[frame-window-1:frame])]),
                        np.max([np.max(fouquet_NMSE[frame-window-1:frame]),np.max(fouquet_NMSE_NLLS[frame-window-1:frame])]) +
                        np.max([np.max(fouquet_NMSE[frame-window-1:frame]),np.max(fouquet_NMSE_NLLS[frame-window-1:frame])])])

    #Awayssa2005
    awayssa_meas.set_xdata(z_meas_real[frame, 0, :])
    awayssa_meas.set_ydata(z_meas_imag[frame, 0, :])
    awayssa_model.set_xdata(awayssa_z_hat_real_BFGS[frame, 0, :])
    awayssa_model.set_ydata(-awayssa_z_hat_imag_BFGS[frame, 0, :])
    awayssa_model_NLLS.set_xdata(awayssa_z_hat_real_NLLS[frame, 0, :])
    awayssa_model_NLLS.set_ydata(-awayssa_z_hat_imag_NLLS[frame, 0, :])
    awayssa_nmse.set_xdata(human_timestamps[:frame + 1])
    awayssa_nmse.set_ydata(awayssa_NMSE[:frame + 1])
    awayssa_nmse_NLLS.set_xdata(human_timestamps[:frame + 1])
    awayssa_nmse_NLLS.set_ydata(awayssa_NMSE_NLLS[:frame + 1])
    axs[5, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[5, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[5, 1].set_xlim([human_timestamps[frame-window-1], human_timestamps[frame]])
    axs[5, 1].set_ylim([np.min([np.min(awayssa_NMSE[frame-window-1:frame]),np.min(awayssa_NMSE_NLLS[frame-window-1:frame])]),
                        np.max([np.max(awayssa_NMSE[frame-window-1:frame]),np.max(awayssa_NMSE_NLLS[frame-window-1:frame])]) +
                        np.max([np.max(awayssa_NMSE[frame-window-1:frame]),np.max(awayssa_NMSE_NLLS[frame-window-1:frame])])])

    return (longo_meas, longo_model, longo_model_NLLS, longo_nmse, longo_nmse_NLLS,
            zurich_meas, zurich_model, zurich_model_NLLS, zurich_nmse, zurich_nmse_NLLS,
            zhang_meas, zhang_model, zhang_nmse, zhang_model_NLLS, zhang_nmse_NLLS,
            yang_meas, yang_model, yang_nmse, yang_model_NLLS, yang_nmse_NLLS,
            fouquet_meas, fouquet_model, fouquet_nmse, fouquet_model_NLLS, fouquet_nmse_NLLS,
            awayssa_meas, awayssa_model, awayssa_nmse, awayssa_model_NLLS, awayssa_nmse_NLLS)

time.sleep(2)
animate = animation.FuncAnimation(fig=fig, func=update_nyquist, frames=len(longo_z_hat_real_BFGS)-1, interval=500)
plt.show()
