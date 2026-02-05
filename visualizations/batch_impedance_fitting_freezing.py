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
longo_z_hat_real = batch_data["longo"]["z_hat_real"]
longo_z_hat_imag = batch_data["longo"]["z_hat_imag"]
longo_NMSE = batch_data["longo"]["nmse"]
longo_chi_square = batch_data["longo"]["chi_square"]

#Zurich2021
zurich_z_hat_real = batch_data["zurich"]["z_hat_real"]
zurich_z_hat_imag = batch_data["zurich"]["z_hat_imag"]
zurich_NMSE = batch_data["zurich"]["nmse"]
zurich_chi_square = batch_data["zurich"]["chi_square"]

#Zhang2024
zhang_z_hat_real = batch_data["zhang"]["z_hat_real"]
zhang_z_hat_imag = batch_data["zhang"]["z_hat_imag"]
zhang_NMSE = batch_data["zhang"]["nmse"]
zhang_chi_square = batch_data["zhang"]["chi_square"]

#Yang2025
yang_z_hat_real = batch_data["yang"]["z_hat_real"]
yang_z_hat_imag = batch_data["yang"]["z_hat_imag"]
yang_NMSE = batch_data["yang"]["nmse"]
yang_chi_square = batch_data["yang"]["chi_square"]

#Fouquet2005
fouquet_z_hat_real = batch_data["fouquet"]["z_hat_real"]
fouquet_z_hat_imag = batch_data["fouquet"]["z_hat_imag"]
fouquet_NMSE = batch_data["fouquet"]["nmse"]
fouquet_chi_square = batch_data["fouquet"]["chi_square"]

#Awayssa2025
awayssa_z_hat_real = batch_data["awayssa"]["z_hat_real"]
awayssa_z_hat_imag = batch_data["awayssa"]["z_hat_imag"]
awayssa_NMSE = batch_data["awayssa"]["nmse"]
awayssa_chi_square = batch_data["awayssa"]["chi_square"]

#animated plots
fig, axs = plt.subplots(nrows=6, ncols=3)

#longo2020
longo_meas = axs[0,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
longo_model = axs[0,0].plot(longo_z_hat_real[0,0,:], -longo_z_hat_imag[0,0,:], color="tab:orange", linestyle="dotted", label="Longo2020")[0]
axs[0,0].legend()
axs[0,0].grid()
axs[0,0].set_title("Nyquist plot")
axs[0,0].set_xlabel("Z'")
axs[0,0].set_ylabel("Z''")
longo_nmse = axs[0,1].plot(human_timestamps[0], longo_NMSE[0])[0]
axs[0,1].grid()
axs[0,1].set_xlabel("Timestamp")
axs[0,1].set_title("NMSE")
longo_chi = axs[0,2].plot(human_timestamps[0], longo_chi_square[0])[0]
axs[0,2].grid()
axs[0,2].set_xlabel("Timestamp")
axs[0,2].set_title("x²")

#zurich2021
zurich_meas = axs[1,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
zurich_model = axs[1,0].plot(zurich_z_hat_real[0,0,:], -zurich_z_hat_imag[0,0,:], color="tab:orange", linestyle="dotted", label="Zurich2021")[0]
axs[1,0].legend()
axs[1,0].grid()
axs[1,0].set_xlabel("Z'")
axs[1,0].set_ylabel("Z''")
zurich_nmse = axs[1,1].plot(human_timestamps[0], zurich_NMSE[0])[0]
axs[1,1].grid()
axs[1,1].set_xlabel("Timestamp")
zurich_chi = axs[1,2].plot(human_timestamps[0], zurich_chi_square[0])[0]
axs[1,2].grid()
axs[1,2].set_xlabel("Timestamp")

#Zhang2024
zhang_meas = axs[2,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
zhang_model = axs[2,0].plot(zhang_z_hat_real[0,0,:], -zhang_z_hat_imag[0,0,:], color="tab:orange", linestyle="dotted", label="Zhang2024")[0]
axs[2,0].legend()
axs[2,0].grid()
axs[2,0].set_xlabel("Z'")
axs[2,0].set_ylabel("Z''")
zhang_nmse = axs[2,1].plot(human_timestamps[0], zhang_NMSE[0])[0]
axs[2,1].grid()
axs[2,1].set_xlabel("Timestamp")
zhang_chi = axs[2,2].plot(human_timestamps[0], zhang_chi_square[0])[0]
axs[2,2].grid()
axs[2,2].set_xlabel("Timestamp")

#Yang2025
yang_meas = axs[3,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
yang_model = axs[3,0].plot(yang_z_hat_real[0,0,:], -yang_z_hat_imag[0,0,:], color="tab:orange", linestyle="dotted", label="Yang2025")[0]
axs[3,0].legend()
axs[3,0].grid()
axs[3,0].set_xlabel("Z'")
axs[3,0].set_ylabel("Z''")
yang_nmse = axs[3,1].plot(human_timestamps[0], yang_NMSE[0])[0]
axs[3,1].grid()
axs[3,1].set_xlabel("Timestamp")
yang_chi = axs[3,2].plot(human_timestamps[0], yang_chi_square[0])[0]
axs[3,2].grid()
axs[3,2].set_xlabel("Timestamp")

#Fouquet2005
fouquet_meas = axs[4,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
fouquet_model = axs[4,0].plot(fouquet_z_hat_real[0,0,:], -fouquet_z_hat_imag[0,0,:], color="tab:orange", linestyle="dotted", label="Fouquet2005")[0]
axs[4,0].legend()
axs[4,0].grid()
axs[4,0].set_xlabel("Z'")
axs[4,0].set_ylabel("Z''")
fouquet_nmse = axs[4,1].plot(human_timestamps[0], fouquet_NMSE[0])[0]
axs[4,1].grid()
axs[4,1].set_xlabel("Timestamp")
fouquet_chi = axs[4,2].plot(human_timestamps[0], fouquet_chi_square[0])[0]
axs[4,2].grid()
axs[4,2].set_xlabel("Timestamp")

#Awayssa2025
awayssa_meas = axs[5,0].plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
awayssa_model = axs[5,0].plot(awayssa_z_hat_real[0,0,:], -awayssa_z_hat_imag[0,0,:], color="tab:orange", linestyle="dotted", label="Awayssa2025")[0]
axs[5,0].legend()
axs[5,0].grid()
axs[5,0].set_xlabel("Z'")
axs[5,0].set_ylabel("Z''")
awayssa_nmse = axs[5,1].plot(human_timestamps[0], awayssa_NMSE[0])[0]
axs[5,1].grid()
axs[5,1].set_xlabel("Timestamp")
awayssa_chi = axs[5,2].plot(human_timestamps[0], awayssa_chi_square[0])[0]
axs[5,2].grid()
axs[5,2].set_xlabel("Timestamp")

def update_nyquist(frame):
    #update plots
    frame += 1
    limits = 10
    window = 0
    #if frame-window < 0:
    #    window = 0
    fig.suptitle(f'timestamp = {human_timestamps[frame]}')

    #Longo2020
    longo_meas.set_xdata(z_meas_real[frame,0,:])
    longo_meas.set_ydata(z_meas_imag[frame,0,:])
    longo_model.set_xdata(longo_z_hat_real[frame,0,:])
    longo_model.set_ydata(-longo_z_hat_imag[frame,0,:])
    longo_nmse.set_xdata(human_timestamps[:frame+1])
    longo_nmse.set_ydata(longo_NMSE[:frame+1])
    longo_chi.set_xdata(human_timestamps[:frame+1])
    longo_chi.set_ydata(longo_chi_square[:frame+1])
    axs[0,0].set_xlim([np.min(z_meas_real[frame,0,:])-limits, np.max(z_meas_real[frame,0,:])+limits])
    axs[0,0].set_ylim([np.min(z_meas_imag[frame,0,:])-limits, np.max(z_meas_imag[frame,0,:])+limits])
    axs[0,1].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[0,1].set_ylim([np.min(longo_NMSE), np.max(longo_NMSE)])
    axs[0,2].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[0,2].set_ylim([np.min(longo_chi_square), np.max(longo_chi_square)])

    #Zurich2021
    zurich_meas.set_xdata(z_meas_real[frame, 0, :])
    zurich_meas.set_ydata(z_meas_imag[frame, 0, :])
    zurich_model.set_xdata(zurich_z_hat_real[frame, 0, :])
    zurich_model.set_ydata(-zurich_z_hat_imag[frame, 0, :])
    zurich_nmse.set_xdata(human_timestamps[:frame + 1])
    zurich_nmse.set_ydata(zurich_NMSE[:frame + 1])
    zurich_chi.set_xdata(human_timestamps[:frame + 1])
    zurich_chi.set_ydata(zurich_chi_square[:frame + 1])
    axs[1, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[1, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[1, 1].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[1,1].set_ylim([np.min(zurich_NMSE), np.max(zurich_NMSE)])
    axs[1, 2].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[1, 2].set_ylim([np.min(zurich_chi_square), np.max(zurich_chi_square)])

    #Zhang2024
    zhang_meas.set_xdata(z_meas_real[frame, 0, :])
    zhang_meas.set_ydata(z_meas_imag[frame, 0, :])
    zhang_model.set_xdata(zhang_z_hat_real[frame, 0, :])
    zhang_model.set_ydata(-zhang_z_hat_imag[frame, 0, :])
    zhang_nmse.set_xdata(human_timestamps[:frame + 1])
    zhang_nmse.set_ydata(zhang_NMSE[:frame + 1])
    zhang_chi.set_xdata(human_timestamps[:frame + 1])
    zhang_chi.set_ydata(zhang_chi_square[:frame + 1])
    axs[2, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[2, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[2, 1].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[2,1].set_ylim([np.min(zhang_NMSE), np.max(zhang_NMSE)])
    axs[2, 2].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[2, 2].set_ylim([np.min(zhang_chi_square), np.max(zhang_chi_square)])

    #Yang2025
    yang_meas.set_xdata(z_meas_real[frame, 0, :])
    yang_meas.set_ydata(z_meas_imag[frame, 0, :])
    yang_model.set_xdata(yang_z_hat_real[frame, 0, :])
    yang_model.set_ydata(-yang_z_hat_imag[frame, 0, :])
    yang_nmse.set_xdata(human_timestamps[:frame + 1])
    yang_nmse.set_ydata(yang_NMSE[:frame + 1])
    yang_chi.set_xdata(human_timestamps[:frame + 1])
    yang_chi.set_ydata(yang_chi_square[:frame + 1])
    axs[3, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[3, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[3, 1].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[3, 1].set_ylim([np.min(yang_NMSE), np.max(yang_NMSE)])
    axs[3, 2].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[3, 2].set_ylim([np.min(yang_chi_square), np.max(yang_chi_square)])

    #Fouquet2005
    fouquet_meas.set_xdata(z_meas_real[frame, 0, :])
    fouquet_meas.set_ydata(z_meas_imag[frame, 0, :])
    fouquet_model.set_xdata(fouquet_z_hat_real[frame, 0, :])
    fouquet_model.set_ydata(-fouquet_z_hat_imag[frame, 0, :])
    fouquet_nmse.set_xdata(human_timestamps[:frame + 1])
    fouquet_nmse.set_ydata(fouquet_NMSE[:frame + 1])
    fouquet_chi.set_xdata(human_timestamps[:frame + 1])
    fouquet_chi.set_ydata(fouquet_chi_square[:frame + 1])
    axs[4, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[4, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[4, 1].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[4, 1].set_ylim([np.min(fouquet_NMSE), np.max(fouquet_NMSE)])
    axs[4, 2].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[4, 2].set_ylim([np.min(fouquet_chi_square), np.max(fouquet_chi_square)])

    #Awayssa2005
    awayssa_meas.set_xdata(z_meas_real[frame, 0, :])
    awayssa_meas.set_ydata(z_meas_imag[frame, 0, :])
    awayssa_model.set_xdata(awayssa_z_hat_real[frame, 0, :])
    awayssa_model.set_ydata(-awayssa_z_hat_imag[frame, 0, :])
    awayssa_nmse.set_xdata(human_timestamps[:frame + 1])
    awayssa_nmse.set_ydata(awayssa_NMSE[:frame + 1])
    awayssa_chi.set_xdata(human_timestamps[:frame + 1])
    awayssa_chi.set_ydata(awayssa_chi_square[:frame + 1])
    axs[5, 0].set_xlim([np.min(z_meas_real[frame, 0, :]) - limits, np.max(z_meas_real[frame, 0, :]) + limits])
    axs[5, 0].set_ylim([np.min(z_meas_imag[frame, 0, :]) - limits, np.max(z_meas_imag[frame, 0, :]) + limits])
    axs[5, 1].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[5, 1].set_ylim([np.min(awayssa_NMSE), np.max(awayssa_NMSE)])
    axs[5, 2].set_xlim([human_timestamps[0], human_timestamps[frame]])
    axs[5, 2].set_ylim([np.min(awayssa_chi_square), np.max(awayssa_chi_square)])

    return longo_meas, longo_model, longo_nmse, longo_chi, zurich_meas, zurich_model, zurich_nmse, zurich_chi, \
           zhang_meas, zhang_model, zhang_nmse, zhang_chi, yang_meas, yang_model, yang_nmse, yang_chi, \
           fouquet_meas, fouquet_model, fouquet_nmse, fouquet_chi, awayssa_meas, awayssa_model, awayssa_nmse, awayssa_chi

time.sleep(2)
animate = animation.FuncAnimation(fig=fig, func=update_nyquist, frames=len(longo_z_hat_real)-1, interval=500)
plt.show()
