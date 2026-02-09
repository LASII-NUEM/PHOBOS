import numpy as np
from framework import file_lcr, fitting_utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os
import time
import matplotlib.dates as mdates

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
longo_fit_params = batch_data["longo"]["params"]

#Zurich2021
zurich_z_hat_real = batch_data["zurich"]["z_hat_real"]
zurich_z_hat_imag = batch_data["zurich"]["z_hat_imag"]
zurich_NMSE = batch_data["zurich"]["nmse"]
zurich_chi_square = batch_data["zurich"]["chi_square"]
zurich_fit_params = batch_data["zurich"]["params"]

#Zhang2024
zhang_z_hat_real = batch_data["zhang"]["z_hat_real"]
zhang_z_hat_imag = batch_data["zhang"]["z_hat_imag"]
zhang_NMSE = batch_data["zhang"]["nmse"]
zhang_chi_square = batch_data["zhang"]["chi_square"]
zhang_fit_params = batch_data["zhang"]["params"]

#Yang2025
yang_z_hat_real = batch_data["yang"]["z_hat_real"]
yang_z_hat_imag = batch_data["yang"]["z_hat_imag"]
yang_NMSE = batch_data["yang"]["nmse"]
yang_chi_square = batch_data["yang"]["chi_square"]
yang_fit_params = batch_data["yang"]["params"]

#Fouquet2005
fouquet_z_hat_real = batch_data["fouquet"]["z_hat_real"]
fouquet_z_hat_imag = batch_data["fouquet"]["z_hat_imag"]
fouquet_NMSE = batch_data["fouquet"]["nmse"]
fouquet_chi_square = batch_data["fouquet"]["chi_square"]
fouquet_fit_params = batch_data["fouquet"]["params"]

#Awayssa2025
awayssa_z_hat_real = batch_data["awayssa"]["z_hat_real"]
awayssa_z_hat_imag = batch_data["awayssa"]["z_hat_imag"]
awayssa_NMSE = batch_data["awayssa"]["nmse"]
awayssa_chi_square = batch_data["awayssa"]["chi_square"]
awayssa_fit_params = batch_data["awayssa"]["params"]

#generate plots
model = "yang2025"
params = {
    "longo2020": longo_fit_params,
    "zurich2021": zurich_fit_params,
    "zhang2024": zhang_fit_params,
    "yang2025": yang_fit_params,
    "fouquet2005": fouquet_fit_params,
    "awayssa2025": awayssa_fit_params
}
fit_handlers = fitting_utils.function_handlers
model_params = params[model]
n_params = fit_handlers[model]["n_params"]
fig, axs = plt.subplots(nrows=1, ncols=n_params)
for i in range(n_params):
    axs[i].plot(human_timestamps, model_params[:,i], color="tab:blue", label=f"{fit_handlers[model]['fit_params'][i]}")
    axs[i].axvline(x=human_timestamps[10], color='red', label="interface", linestyle="dotted")
    axs[i].legend()
    axs[i].grid()
    axs[i].set_xlabel("Timestamp")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=3)
    axs[i].xaxis.set_major_locator(locator)
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))