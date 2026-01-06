import numpy as np
from framework import data_types, dsp_utils
import datetime
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#read the files
filename_flange = '../data/testICE_02_12_25/c_test.csv'
filename_temp = '../data/testICE_02_12_25/c_temp.lvm'
phobos_obj = data_types.PHOBOSData(filename_flange, filename_temperature=filename_temp, n_samples=1, sweeptype="flange", acquisition_mode="freq", aggregate=np.mean)

#compute the derivatives of each thermocouple
bdiff_thermo_readings = dsp_utils.first_order_backward_diff(phobos_obj.thermo_readings, eps=1) #backward difference
cdiff_thermo_readings = dsp_utils.first_order_central_diff(phobos_obj.thermo_readings, eps=1) #central difference

#find the peaks of each derivative
bdiff_max = np.max(bdiff_thermo_readings, axis=0) #backward diff. maximum for each thermocouple
bdiff_idx = np.where(bdiff_thermo_readings == bdiff_max) #max indexes
cdiff_max = np.max(cdiff_thermo_readings, axis=0) #central diff. maximum for each thermocouple
cdiff_idx = np.where(cdiff_thermo_readings == cdiff_max) #max indexes

plt.figure(1)
plt.subplot(2,1,1)
leg = []
colors_temp = [(0.85, 0.45, 0.20),
               (0.90, 0.55, 0.25),
               (0.95, 0.65, 0.35),
               (0.85, 0.60, 0.40)] #orange-like tones
linestyles_temp = ["-", "--", ":", "-."]
labels_temp = ['T1', 'T2', 'T3', 'T4']
for i in range(len(labels_temp)):
    plt.plot(phobos_obj.temp_human_timestamp, bdiff_thermo_readings[:,i],
             color=colors_temp[i], linestyle=linestyles_temp[i])
    leg.append(labels_temp[i])

plt.ylabel('Backward difference of the temperature')
plt.legend(leg)
plt.scatter(phobos_obj.temp_human_timestamp[bdiff_idx[0]], bdiff_thermo_readings[bdiff_idx],
            color="blue", marker="x")
for j in range(len(bdiff_idx[0])):
    thermo_stamp = pd.Timestamp(phobos_obj.temp_human_timestamp[bdiff_idx[0][j]])
    plt.text(phobos_obj.temp_human_timestamp[bdiff_idx[0][j]], bdiff_thermo_readings[bdiff_idx[0][j], bdiff_idx[1][j]],
             f'{thermo_stamp.hour}:{thermo_stamp.minute}:{thermo_stamp.second}',
             color="blue")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
delta = datetime.timedelta(seconds=10) #time delta of 10s
plt.xlim([phobos_obj.temp_human_timestamp[bdiff_idx[0][-1]]-delta, phobos_obj.temp_human_timestamp[bdiff_idx[0][0]]+delta])
plt.grid()

plt.subplot(2,1,2)
leg = []
colors_temp = [(0.85, 0.45, 0.20),
               (0.90, 0.55, 0.25),
               (0.95, 0.65, 0.35),
               (0.85, 0.60, 0.40)] #orange-like tones
linestyles_temp = ["-", "--", ":", "-."]
labels_temp = ['T1', 'T2', 'T3', 'T4']
diff_peaks = []
time_peaks = []
for i in range(len(labels_temp)):
    plt.plot(phobos_obj.temp_human_timestamp, cdiff_thermo_readings[:,i],
             color=colors_temp[i], linestyle=linestyles_temp[i])
    leg.append(labels_temp[i])
plt.ylabel('Central difference of the temperature')
plt.legend(leg)
plt.scatter(phobos_obj.temp_human_timestamp[cdiff_idx[0]], cdiff_thermo_readings[cdiff_idx],
            color="blue", marker="x")
for j in range(len(cdiff_idx[0])):
    thermo_stamp = pd.Timestamp(phobos_obj.temp_human_timestamp[cdiff_idx[0][j]])
    plt.text(phobos_obj.temp_human_timestamp[cdiff_idx[0][j]], cdiff_thermo_readings[cdiff_idx[0][j], cdiff_idx[1][j]],
             f'{thermo_stamp.hour}:{thermo_stamp.minute}:{thermo_stamp.second}',
             color="blue")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
delta = datetime.timedelta(seconds=10) #time delta of 10s
plt.xlim([phobos_obj.temp_human_timestamp[bdiff_idx[0][-1]]-delta, phobos_obj.temp_human_timestamp[bdiff_idx[0][0]]+delta])
plt.grid()
