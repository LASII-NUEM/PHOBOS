import numpy as np
from framework import data_types
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#read the files
filename_flange = '../data/testICE_02_12_25/c_test.csv'
filename_temp = '../data/testICE_02_12_25/c_temp.lvm'
phobos_obj = data_types.PHOBOSData(filename_flange, filename_temperature=filename_temp, n_samples=1, sweeptype="flange", acquisition_mode="freq", aggregate=np.mean)

#plot temperature vs. capacitance normalized
fig, ax1 = plt.subplots()
elec_colors = ["green", "blue"]
elec_labels = ['Avg. Cp norm. @ 10 kHz', 'Avg. Cp @ 1 MHz']
for freq_idx in range(0, phobos_obj.n_freqs):
    ax1.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Cp_norm[:,freq_idx],
             color=elec_colors[freq_idx],
             linestyle='dashed',
             label=elec_labels[freq_idx])
ax1.set_xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax1.set_ylabel('Capacitance [-]', color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2 = ax1.twinx() #instantiate a second Axes that shares the same x-axis
colors_temp = [(0.85, 0.45, 0.20),
               (0.90, 0.55, 0.25),
               (0.95, 0.65, 0.35),
               (0.85, 0.60, 0.40)] #orange-like tones
linestyles_temp = ["-", "--", ":", "-."]
labels_temp = ['T1', 'T2', 'T3', 'T4']
for thermo_idx in range(0,phobos_obj.n_thermosensors):
    ax2.plot(phobos_obj.temp_human_timestamp, phobos_obj.thermo_readings[:,thermo_idx],
             color=colors_temp[thermo_idx],
             linestyle=linestyles_temp[thermo_idx],
             linewidth=1,
             label=labels_temp[thermo_idx])
ax2.set_ylabel('Temperature [°C]', color=colors_temp[0])
ax2.tick_params(axis='y', labelcolor=colors_temp[0])
ax1.grid()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
fig.tight_layout()
plt.show()

#plot temperature vs. resistance normalized
fig, ax1 = plt.subplots()
elec_colors = ["green", "blue"]
elec_labels = ['Avg. Rp norm. @ 10 kHz', 'Avg. Rp norm. @ 1 MHz']
for freq_idx in range(0, phobos_obj.n_freqs):
    ax1.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Rp_norm[:, freq_idx],
             color=elec_colors[freq_idx],
             linestyle='dashed',
             label=elec_labels[freq_idx])
ax1.set_xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax1.set_ylabel('Resistance [-]', color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2 = ax1.twinx() #instantiate a second Axes that shares the same x-axis
colors_temp = [(0.85, 0.45, 0.20),
               (0.90, 0.55, 0.25),
               (0.95, 0.65, 0.35),
               (0.85, 0.60, 0.40)] #orange-like tones
linestyles_temp = ["-", "--", ":", "-."]
labels_temp = ['T1', 'T2', 'T3', 'T4']
for thermo_idx in range(0,phobos_obj.n_thermosensors):
    ax2.plot(phobos_obj.temp_human_timestamp, phobos_obj.thermo_readings[:,thermo_idx],
             color=colors_temp[thermo_idx],
             linestyle=linestyles_temp[thermo_idx],
             linewidth=1,
             label=labels_temp[thermo_idx])
ax2.set_ylabel('Temperature [°C]', color=colors_temp[0])
ax2.tick_params(axis='y', labelcolor=colors_temp[0])
ax1.grid()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
fig.tight_layout()
plt.show()

#plot avg. norm. capacitance vs. norm. capacitance per mode
plt.figure()
leg = []
avg_labels = ['Avg. Cp norm. @ 10 kHz', 'Avg. Cp norm. @ 1 MHz']
mode_labels = ['Cp norm. @ 10 kHz', 'Cp norm. @ 1 MHz']
for f in range(0, phobos_obj.n_freqs):
    plt.subplot(phobos_obj.n_freqs,1,f+1)
    leg = []
    for i in range(0, phobos_obj.n_modes):
        plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Cp_norm[:,i,f],
                 color=(0.7, 0.7, 0.7),
                 marker='o')
        if i == phobos_obj.n_modes-1:
            plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Cp_norm[:, i, f],
                     color=(0.7, 0.7, 0.7),
                     marker='o',
                     label=mode_labels[f])
    plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Cp_norm[:, f],
             color='tab:orange',
             label=avg_labels[f])
    plt.ylabel('Normalized capacitance [-]')
    plt.xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.legend()
    plt.grid()
plt.show()

#plot avg. norm. resistance vs. norm. resistance per mode
plt.figure()
leg = []
avg_labels = ['Avg. Rp norm. @ 10 kHz', 'Avg. Rp norm. @ 1 MHz']
mode_labels = ['Rp norm. @ 10 kHz', 'Rp norm. @ 1 MHz']
for f in range(0, phobos_obj.n_freqs):
    plt.subplot(phobos_obj.n_freqs,1,f+1)
    leg = []
    for i in range(0, phobos_obj.n_modes):
        plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Rp_norm[:,i,f],
                 color=(0.7, 0.7, 0.7),
                 marker='o')
        if i == phobos_obj.n_modes-1:
            plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Rp_norm[:, i, f],
                     color=(0.7, 0.7, 0.7),
                     marker='o',
                     label=mode_labels[f])
    plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Rp_norm[:, f],
             color='tab:orange',
             label=avg_labels[f])
    plt.ylabel('Normalized resistance [-]')
    plt.xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.legend()
    plt.grid()
plt.show()

#capacitance and resistance by mode as an image (heatmap)
plt.figure()
y_idx = np.arange(phobos_obj.n_modes)
x_idx = np.arange(len(phobos_obj.electrode_human_timestamps))
for f in range(0, phobos_obj.n_freqs):
    #capacitance plot
    plt.subplot(phobos_obj.n_freqs,2, 2*f+1)
    plt.imshow((phobos_obj.Cp[:,:,f]*1e12).astype('float').T,
               aspect='auto',
               cmap='jet')
    ax = plt.gca()
    ax.set_yticks(y_idx)
    ax.set_yticklabels(phobos_obj.modes)
    step = max(1, len(phobos_obj.electrode_human_timestamps) // 7)
    tick_idx = np.arange(0, len(phobos_obj.electrode_human_timestamps), step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([phobos_obj.electrode_human_timestamps[i].astype('datetime64[ms]').astype(object).strftime('%H:%M:%S') for i in tick_idx])
    plt.ylabel('Activation mode')
    plt.colorbar(label='Capacitance [pF]')

    #resistance plot
    plt.subplot(phobos_obj.n_freqs, 2, 2*f+2)
    plt.imshow((phobos_obj.Rp[:,:,f]).astype('float').T,
               aspect='auto',
               cmap='jet')
    ax = plt.gca()
    ax.set_yticks(y_idx)
    ax.set_yticklabels(phobos_obj.modes)
    step = max(1, len(phobos_obj.electrode_human_timestamps)//7)
    tick_idx = np.arange(0, len(phobos_obj.electrode_human_timestamps), step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([phobos_obj.electrode_human_timestamps[i].astype('datetime64[ms]').astype(object).strftime('%H:%M:%S') for i in tick_idx])
    plt.colorbar(label='Resistance [Ω]')
plt.show()