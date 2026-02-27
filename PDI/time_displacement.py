import numpy as np
from framework import data_types
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

#read the files
filename_flange = '../data/testICE_02_12_25/c_test.csv'
filename_temp = '../data/testICE_02_12_25/c_temp.lvm'
phobos_obj = data_types.PHOBOSData(filename_flange, filename_temperature=filename_temp, n_samples=1, electrode="flange", acquisition_mode="freq", aggregate=np.mean)

#find where each temperature curve crosses zero
zero_cross_T1 = np.argmin(np.abs(phobos_obj.thermo_readings[:,0]-0))
zero_cross_T2 = np.argmin(np.abs(phobos_obj.thermo_readings[:,1]-0))
zero_cross_T3 = np.argmin(np.abs(phobos_obj.thermo_readings[:,2]-0))
zero_cross_T4 = np.argmin(np.abs(phobos_obj.thermo_readings[:,3]-0))

#define the time displacement between zero crossings
#reference to the T4 sensor
t_delta = phobos_obj.temp_human_timestamp[[zero_cross_T1, zero_cross_T2, zero_cross_T3, zero_cross_T4]][::-1] #reverse to consider T4 the first instant
t_delta = t_delta.astype("datetime64[ms]") - t_delta[0].astype("datetime64[ms]")
t_delta = np.abs(t_delta/np.timedelta64(1, 's'))

ratio = phobos_obj.temp_human_timestamp[[zero_cross_T1, zero_cross_T2, zero_cross_T3, zero_cross_T4]][::-1]
ratio = ratio.astype("datetime64[s]").astype("float")

#plot temperature vs. capacitance normalized
fig, ax1 = plt.subplots()
colors_temp = [(0.85, 0.45, 0.20),
               (0.90, 0.55, 0.25),
               (0.95, 0.65, 0.35),
               (0.85, 0.60, 0.40)] #orange-like tones
linestyles_temp = ["-", "--", ":", "-."]
labels_temp = ['T1', 'T2', 'T3', 'T4']
zero_cross = [zero_cross_T1, zero_cross_T2, zero_cross_T3, zero_cross_T4]
for thermo_idx in range(0,phobos_obj.n_thermosensors):
    ax1.plot(phobos_obj.temp_human_timestamp, phobos_obj.thermo_readings[:,thermo_idx],
             color=colors_temp[thermo_idx],
             linestyle=linestyles_temp[thermo_idx],
             linewidth=1,
             label=labels_temp[thermo_idx])
    ax1.scatter(phobos_obj.temp_human_timestamp[zero_cross[thermo_idx]], phobos_obj.thermo_readings[[zero_cross[thermo_idx]], thermo_idx], c="blue", marker='x')
ax1.set_ylabel('Temperature [°C]', color=colors_temp[0])
ax1.tick_params(axis='y', labelcolor=colors_temp[0])
ax1.grid()
lines2, labels2 = ax1.get_legend_handles_labels()
ax1.legend(lines2, labels2, loc=0)
fig.tight_layout()
plt.show()

plt.figure()
plt.plot((ratio[1:]), t_delta[1:], '-o')
plt.grid()
plt.ylabel('Time displacement [s]')
plt.xlabel('Unix Timestamps [s]')
plt.show()