from framework import file_lvm
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

temp_file_path = '../data/testICE_02_12_25/c_temp.lvm' #relative path to where the temperature file is stored
lvm_obj = file_lvm.read(temp_file_path, setup='freezer') #read the lvm file and store it as a TemperatureData structure

#some important attributes of the TemperatureData class
timestamps = lvm_obj.human_timestamp #human timestamps (%Y%M%D:%h%m%s)
rel_timestamps = lvm_obj.relative_timestamp #relative timestamps (t_init = 0s)
temp_data = lvm_obj.measured_temp #Nx4 matrix with the temperature readings
#OBS: [0] -> T1;
#     [1] -> T2;
#     [2] -> T3;
#     [3] -> T4

#plot temperature data
fig, ax1 = plt.subplots()
colors_temp = [(0.85, 0.45, 0.20),
               (0.90, 0.55, 0.25),
               (0.95, 0.65, 0.35),
               (0.85, 0.60, 0.40)] #orange-like tones
linestyles_temp = ["-", "--", ":", "-."]
labels_temp = ['T1', 'T2', 'T3', 'T4']
for thermo_idx in range(0,lvm_obj.n_sensors):
    ax1.plot(lvm_obj.human_timestamp, lvm_obj.measured_temp[:,thermo_idx],
             color=colors_temp[thermo_idx],
             linestyle=linestyles_temp[thermo_idx],
             linewidth=1,
             label=labels_temp[thermo_idx])
ax1.set_ylabel('Temperature [°C]', color=colors_temp[0])
ax1.tick_params(axis='y', labelcolor=colors_temp[0])
ax1.grid()
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc=0)
fig.tight_layout()
plt.show()
