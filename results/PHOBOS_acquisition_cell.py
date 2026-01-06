import numpy as np
from framework import data_types
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#read the files
filename_flange = '../data/testICE_10_12_25/c_test.csv'
filename_temp = '../data/testICE_10_12_25/c_temp.lvm'
phobos_obj = data_types.PHOBOSData(filename_flange, n_samples=1, sweeptype="cell", acquisition_mode="freq", aggregate=np.mean)

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
ax1.grid()
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc=0)
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
ax1.grid()
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc=0)
fig.tight_layout()
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