import numpy as np
from framework import data_types
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#read the files
filename_flange = '../data/testICE_02_12_25/c_test.csv'
filename_temp = '../data/testICE_02_12_25/c_temp.lvm'
filename_ice_thickness = '../data/testICE_02_12_25/ice_thickness_vs_time.npy'

data_ice_thickness = np.load(filename_ice_thickness, allow_pickle=True)
thickness_frametime = data_ice_thickness[:, 0]
thickness_array = data_ice_thickness[:, 1]

phobos_obj = data_types.PHOBOSData(filename_flange, filename_temperature=filename_temp, n_samples=1, sweeptype="flange", acquisition_mode="freq", aggregate=np.mean)

# #plot temperature vs. capacitance normalized
# fig, ax1 = plt.subplots()
# elec_colors = ["green", "blue"]
# elec_labels = ['Avg. Cp norm. @ 10 kHz', 'Avg. Cp norm. @ 1 MHz']
# for freq_idx in range(0, phobos_obj.n_freqs):
#     ax1.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Cp_norm[:,freq_idx],
#              color=elec_colors[freq_idx],
#              linestyle='dashed',
#              label=elec_labels[freq_idx])
# ax1.set_xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
# ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax1.set_ylabel('Capacitance [-]', color="tab:blue")
# ax1.tick_params(axis='y', labelcolor="tab:blue")
# ax2 = ax1.twinx() #instantiate a second Axes that shares the same x-axis
# colors_temp = [(0.85, 0.45, 0.20),
#                (0.90, 0.55, 0.25),
#                (0.95, 0.65, 0.35),
#                (0.85, 0.60, 0.40)] #orange-like tones
# linestyles_temp = ["-", "--", ":", "-."]
# labels_temp = ['T1', 'T2', 'T3', 'T4']
# for thermo_idx in range(0,phobos_obj.n_thermosensors):
#     ax2.plot(phobos_obj.temp_human_timestamp, phobos_obj.thermo_readings[:,thermo_idx],
#              color=colors_temp[thermo_idx],
#              linestyle=linestyles_temp[thermo_idx],
#              linewidth=1,
#              label=labels_temp[thermo_idx])
# ax2.set_ylabel('Temperature [°C]', color=colors_temp[0])
# ax2.tick_params(axis='y', labelcolor=colors_temp[0])
# ax1.grid()
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc=0)
# fig.tight_layout()
# plt.show()

#plot temperature vs. resistance normalized
# fig, ax1 = plt.subplots()
# elec_colors = ["green", "blue"]
# elec_labels = ['Avg. Rp norm. @ 10 kHz', 'Avg. Rp norm. @ 1 MHz']
# # for freq_idx in range(0, phobos_obj.n_freqs):
# ax1.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Rp[:,0, 1],
#          color=elec_colors[1],
#          linestyle='dashed',
#          label=elec_labels[1])
# ax1.set_xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
# ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
# ax1.set_ylabel('Resistance [-]', color="tab:blue")
# ax1.tick_params(axis='y', labelcolor="tab:blue")
# ax2 = ax1.twinx() #instantiate a second Axes that shares the same x-axis
# colors_temp = [(0.85, 0.45, 0.20),
#                (0.90, 0.55, 0.25),
#                (0.95, 0.65, 0.35),
#                (0.85, 0.60, 0.40)] #orange-like tones
# linestyles_temp = ["-", "--", ":", "-."]
# labels_temp = ['T1', 'T2', 'T3', 'T4']
# for thermo_idx in range(0,phobos_obj.n_thermosensors):
#     ax2.plot(phobos_obj.temp_human_timestamp, phobos_obj.thermo_readings[:,thermo_idx],
#              color=colors_temp[thermo_idx],
#              linestyle=linestyles_temp[thermo_idx],
#              linewidth=1,
#              label=labels_temp[thermo_idx])
# ax2.set_ylabel('Temperature [°C]', color=colors_temp[0])
# ax2.tick_params(axis='y', labelcolor=colors_temp[0])
# ax1.grid()
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc=0)
# fig.tight_layout()
# plt.show()

# #plot avg. norm. capacitance vs. norm. capacitance per mode
# plt.figure()
# leg = []
# avg_labels = ['Avg. Cp norm. @ 10 kHz', 'Avg. Cp norm. @ 1 MHz']
# mode_labels = ['Cp norm. @ 10 kHz', 'Cp norm. @ 1 MHz']
# for f in range(0, phobos_obj.n_freqs):
#     plt.subplot(phobos_obj.n_freqs,1,f+1)
#     leg = []
#     for i in range(0, phobos_obj.n_modes):
#         plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Cp_norm[:,i,f],
#                  color=(0.7, 0.7, 0.7),
#                  marker='o')
#         if i == phobos_obj.n_modes-1:
#             plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Cp_norm[:, i, f],
#                      color=(0.7, 0.7, 0.7),
#                      marker='o',
#                      label=mode_labels[f])
#     plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Cp_norm[:, f],
#              color='tab:orange',
#              label=avg_labels[f])
#     plt.ylabel('Normalized capacitance [-]')
#     plt.xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     plt.legend()
#     plt.grid()
# plt.show()

# #plot avg. norm. resistance vs. norm. resistance per mode
# plt.figure()
# leg = []
# avg_labels = ['Avg. Rp norm. @ 10 kHz', 'Avg. Rp norm. @ 1 MHz']
# mode_labels = ['Rp norm. @ 10 kHz', 'Rp norm. @ 1 MHz']
# for f in range(0, phobos_obj.n_freqs):
#     plt.subplot(phobos_obj.n_freqs,1,f+1)
#     leg = []
#     for i in range(0, phobos_obj.n_modes):
#         plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Rp_norm[:,i,f],
#                  color=(0.7, 0.7, 0.7),
#                  marker='o')
#         if i == phobos_obj.n_modes-1:
#             plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.Rp_norm[:, i, f],
#                      color=(0.7, 0.7, 0.7),
#                      marker='o',
#                      label=mode_labels[f])
#     plt.plot(phobos_obj.electrode_human_timestamps, phobos_obj.agg_Rp_norm[:, f],
#              color='tab:orange',
#              label=avg_labels[f])
#     plt.ylabel('Normalized resistance [-]')
#     plt.xlim([phobos_obj.electrode_human_timestamps[1], phobos_obj.electrode_human_timestamps[-3]])
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     plt.legend()
#     plt.grid()
# plt.show()

# #capacitance and resistance by mode as an image (heatmap)
# plt.figure()
# y_idx = np.arange(phobos_obj.n_modes)
# x_idx = np.arange(len(phobos_obj.electrode_human_timestamps))
# for f in range(0, phobos_obj.n_freqs):
#     #capacitance plot
#     plt.subplot(phobos_obj.n_freqs,2, 2*f+1)
#     plt.imshow((phobos_obj.Cp[:,:,f]*1e12).astype('float').T,
#                aspect='auto',
#                cmap='jet')
#     ax = plt.gca()
#     ax.set_yticks(y_idx)
#     ax.set_yticklabels(phobos_obj.modes)
#     step = max(1, len(phobos_obj.electrode_human_timestamps) // 7)
#     tick_idx = np.arange(0, len(phobos_obj.electrode_human_timestamps), step)
#     ax.set_xticks(tick_idx)
#     ax.set_xticklabels([phobos_obj.electrode_human_timestamps[i].astype('datetime64[ms]').astype(object).strftime('%H:%M:%S') for i in tick_idx])
#     plt.ylabel('Activation mode')
#     plt.colorbar(label='Capacitance [pF]')
#
#     #resistance plot
#     plt.subplot(phobos_obj.n_freqs, 2, 2*f+2)
#     plt.imshow((phobos_obj.Rp[:,:,f]).astype('float').T,
#                aspect='auto',
#                cmap='jet')
#     ax = plt.gca()
#     ax.set_yticks(y_idx)
#     ax.set_yticklabels(phobos_obj.modes)
#     step = max(1, len(phobos_obj.electrode_human_timestamps)//7)
#     tick_idx = np.arange(0, len(phobos_obj.electrode_human_timestamps), step)
#     ax.set_xticks(tick_idx)
#     ax.set_xticklabels([phobos_obj.electrode_human_timestamps[i].astype('datetime64[ms]').astype(object).strftime('%H:%M:%S') for i in tick_idx])
#     plt.colorbar(label='Resistance [Ω]')
# plt.show()

# ----------------------------------------------------------------------
#plot Ice Thickness vs. capacitance normalized vs. resistance normalized

dt_obj = np.asarray(thickness_frametime)  # dtype=object
Thick_time = np.array([d.strftime("%H:%M:%S") for d in dt_obj], dtype="<U8")

ft_dt = np.asarray(thickness_frametime, dtype="datetime64[s]")
ts_dt = np.asarray(phobos_obj.electrode_human_timestamps, dtype="datetime64[s]")
temp_dt  = np.asarray(phobos_obj.temp_human_timestamp, dtype="datetime64[s]")

ft_sec = (ft_dt - ft_dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64) #thickness timestamp
ts_sec = (ts_dt - ts_dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64) #CpRp timestamp
temp_sec = (temp_dt - temp_dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64) #temperature timestamp

ts = np.asarray(ts_sec).ravel()
ft = np.asarray(ft_sec).ravel()
temps = np.asarray(temp_sec).ravel()

#idx CpRp
order = np.argsort(ts)
ts_s = ts[order]

pos = np.searchsorted(ts_s, ft)
pos = np.clip(pos, 1, len(ts_s) - 1)

left  = ts_s[pos - 1]
right = ts_s[pos]

idx_s = np.where((ft - left) <= (right - ft), pos - 1, pos)
idx_closest = order[idx_s]

# C = phobos_obj.Cp_norm
# C_clip = C[idx_closest]
# C_agg_norm = phobos_obj.agg_Cp_norm
# C_agg_norm_clip = C_agg_norm[idx_closest]

C_agg = phobos_obj.Cp_agg
C_agg_clip = C_agg[idx_closest]

# R = phobos_obj.Rp_norm
# R_clip = R[idx_closest]
# R_agg_norm = phobos_obj.agg_Rp_norm
# R_agg_norm_clip = R_agg_norm[idx_closest]

R_agg = phobos_obj.Rp_agg
R_agg_clip = R_agg[idx_closest]

#idx temperature
order_t = np.argsort(temps)
temps_s = temps[order_t]

pos_t = np.searchsorted(temps_s, ft)
pos_t = np.clip(pos_t, 1, len(temps_s) - 1)

left_t  = temps_s[pos_t - 1]
right_t = temps_s[pos_t]

idx_t = np.where((ft - left_t) <= (right_t - ft), pos_t - 1, pos_t)
idx_closest_t = order_t[idx_t]

thermo = phobos_obj.thermo_readings
thermo_clip = thermo[idx_closest_t]


#plot temperature vs. capacitance normalized
# side-by-side: left = Cp, right = Rp (shared x)
# fig, (ax_cp, ax_rp) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
# fig.suptitle("Cp vs. Rp vs. Ice thickness")
#
# Mode = ['d:1-2', 'd:2-3','d:3-4','d:4-5','d:5-6','d:6-7','d:7-8','d:8-9','d:9-10','d:10-1']
# linestyles = ["dashed", "solid"]
#
# # ---- Cp (left) ----
# for mode in range(phobos_obj.n_modes):
#     for freq_idx in range(phobos_obj.n_freqs):
#         ax_cp.plot(
#             thickness_array, C_clip[:, mode, freq_idx],
#             color="tab:blue",
#             linestyle=linestyles[freq_idx], marker="o",
#             label=f"{Mode[mode]} | f{freq_idx}"
#         )
#
# ax_cp.set_title("Cp")
# ax_cp.set_xlabel("Ice thickness [mm]")
# ax_cp.set_ylabel("Capacitance [-]")
# ax_cp.grid(True)
#
# # ---- Rp (right) ----
# for mode in range(phobos_obj.n_modes):
#     for freq_idx in range(phobos_obj.n_freqs):
#         ax_rp.plot(
#             thickness_array, R_clip[:, mode, freq_idx],
#             color="tab:red",
#             linestyle=linestyles[freq_idx], marker="v",
#             label=f"{Mode[mode]} | f{freq_idx}"
#         )
#
# ax_rp.set_title("Rp")
# ax_rp.set_xlabel("Ice thickness [mm]")
# ax_rp.set_ylabel("Resistance [-]")
# ax_rp.grid(True)
# # handles1, labels1 = ax_cp.get_legend_handles_labels()
# # handles2, labels2 = ax_rp.get_legend_handles_labels()
# fig.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
# plt.show()

# fig, ax1 = plt.subplots()
# fig.suptitle("Agg Cp vs. Agg Rp vs. Ice thickness")
# cap_labels = ['Avg. Cp norm. @ 10 kHz', 'Avg. Cp norm. @ 1 MHz']
# res_labels = ['Avg. Rp norm. @ 10 kHz', 'Avg. Rp norm. @ 1 MHz']
# linestyles = ["dashed", "solid"]
# for freq_idx in range(0, phobos_obj.n_freqs):
#     ax1.plot(thickness_array,C_agg_clip[:,freq_idx],
#              color="tab:blue",
#              linestyle="solid",marker ="o",
#              label=cap_labels[freq_idx])
# ax1.set_ylabel('Capacitance [-]', color="tab:blue")
# ax1.tick_params(axis='y', labelcolor="tab:blue")
# ax2 = ax1.twinx() #instantiate a second Axes that shares the same x-axis
# for freq_idx in range(0, phobos_obj.n_freqs):
#     ax2.plot(thickness_array,R_agg_clip[:,freq_idx],
#              color="tab:red",
#              linestyle="solid",marker ="v",
#              label=res_labels[freq_idx])
# ax2.set_ylabel('Resistance [--]', color="tab:red")
# ax2.tick_params(axis='y', labelcolor="tab:red")
# ax1.set_xlabel('Ice thickness [mm]')
# ax1.grid()
# ax3 = ax1.twinx()
# ax3.spines['right'].set_position(('outward', 60))
# colors_temp = [(0.85, 0.45, 0.20),
#                (0.90, 0.55, 0.25),
#                (0.95, 0.65, 0.35),
#                (0.85, 0.60, 0.40)] #orange-like tones
# linestyles_temp = ["-", "--", ":", "-."]
# labels_temp = ['T1', 'T2', 'T3', 'T4']
# for thermo_idx in range(0,phobos_obj.n_thermosensors):
#     ax3.plot(thickness_array, thermo_clip[:,thermo_idx],
#              color=colors_temp[thermo_idx],
#              linestyle=linestyles_temp[thermo_idx],
#              linewidth=1,
#              label=labels_temp[thermo_idx])
# ax3.set_ylabel('Temperature [°C]', color=colors_temp[0])
# ax3.tick_params(axis='y', labelcolor=colors_temp[0])
#
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# lines3, labels3 = ax3.get_legend_handles_labels()
# ax2.legend(lines + lines2+ lines3, labels + labels2 + labels3, loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, ncol=8)
# plt.show()

# fig, ax = plt.subplots(3,1, figsize = (15,10))
# fig.suptitle("Agg Cp vs. Agg Rp vs. Ice thickness")
# cap_labels = ['Avg. Cp norm. @ 10 kHz', 'Avg. Cp norm. @ 1 MHz']
# res_labels = ['Avg. Rp norm. @ 10 kHz', 'Avg. Rp norm. @ 1 MHz']
# marker = ["o", "v"]
# for freq_idx in range(0, phobos_obj.n_freqs):
#     ax[0].plot(thickness_array,C_agg_clip[:,freq_idx],
#              color="tab:blue",
#              linestyle="solid",marker =marker[freq_idx],
#              label=cap_labels[freq_idx])
# ax[0].set_ylabel('Capacitance [-]', color="tab:blue")
# ax[0].tick_params(axis='y', labelcolor="tab:blue")
# for freq_idx in range(0, phobos_obj.n_freqs):
#     ax[1].plot(thickness_array,R_agg_clip[:,freq_idx],
#              color="tab:red",
#              linestyle="solid",marker =marker[freq_idx],
#              label=res_labels[freq_idx])
# ax[1].set_ylabel('Resistance [--]', color="tab:red")
# ax[1].tick_params(axis='y', labelcolor="tab:red")
# ax[2].set_xlabel('Ice thickness [mm]')
# ax[0].grid()
# ax[1].grid()
# ax[2].grid()
#
# colors_temp = [(0.85, 0.45, 0.20),
#                (0.90, 0.55, 0.25),
#                (0.95, 0.65, 0.35),
#                (0.85, 0.60, 0.40)] #orange-like tones
# linestyles_temp = ["-", "--", ":", "-."]
# labels_temp = ['T1', 'T2', 'T3', 'T4']
# for thermo_idx in range(0,phobos_obj.n_thermosensors):
#     ax[2].plot(thickness_array, thermo_clip[:,thermo_idx],
#              color=colors_temp[thermo_idx],
#              linestyle=linestyles_temp[thermo_idx], marker = "x",
#              linewidth=1,
#              label=labels_temp[thermo_idx])
# ax[2].set_ylabel('Temperature [°C]', color=colors_temp[0])
# ax[2].tick_params(axis='y', labelcolor=colors_temp[0])
#
# lines, labels = ax[0].get_legend_handles_labels()
# lines2, labels2 = ax[1].get_legend_handles_labels()
# lines3, labels3 = ax[2].get_legend_handles_labels()
# ax[2].legend(lines + lines2+ lines3, labels + labels2 + labels3, loc='upper center', bbox_to_anchor=(0.5, -0.3),
#           fancybox=True, ncol=8)
# plt.show()

fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
fig.suptitle("Agg Cp vs. Agg Rp vs. Ice thickness")

cap_labels = ['Avg. Cp norm. @ 10 kHz', 'Avg. Cp norm. @ 1 MHz']
res_labels = ['Avg. Rp norm. @ 10 kHz', 'Avg. Rp norm. @ 1 MHz']
marker = ["o", "v"]

# --- Cp row (ax[0]) ---
ax0r = ax[0].twinx()
ax[0].plot(thickness_array, C_agg_clip[:, 0], linestyle="solid", color='tab:blue',marker = marker[0],
           label=cap_labels[0])
ax0r.plot(thickness_array, C_agg_clip[:, 1], linestyle="dotted", color='tab:blue',marker = marker[0],
          label=cap_labels[1])

ax[0].set_ylabel('Capacitance [F] @ 10 kHz')
ax0r.set_ylabel('Capacitance [F] @ 1 MHz')
ax[0].grid(True)

# --- Rp row (ax[1]) ---
ax1r = ax[1].twinx()
ax[1].plot(thickness_array, R_agg_clip[:, 0], linestyle="solid",color='tab:red',marker = marker[1],
           label=res_labels[0])
ax1r.plot(thickness_array, -R_agg_clip[:, 1], linestyle="dotted", color='tab:red',marker = marker[1],
          label=res_labels[1])

ax[1].set_ylabel(r'Resistance [$\Omega$] @ 10 kHz')
ax1r.set_ylabel(r'Resistance [$\Omega$] @ 1 MHz')
ax[1].grid(True)

# --- Temperature row (ax[2]) ---
colors_temp = [(0.85, 0.45, 0.20),
               (0.90, 0.55, 0.25),
               (0.95, 0.65, 0.35),
               (0.85, 0.60, 0.40)]
linestyles_temp = ["-", "--", ":", "-."]
labels_temp = ['T1', 'T2', 'T3', 'T4']

for thermo_idx in range(phobos_obj.n_thermosensors):
    ax[2].plot(thickness_array, thermo_clip[:, thermo_idx],
               color=colors_temp[thermo_idx],
               linestyle=linestyles_temp[thermo_idx],
               marker="x", linewidth=1,
               label=labels_temp[thermo_idx])

ax[2].set_ylabel('Temperature [°C]')
ax[2].set_xlabel('Ice thickness [mm]')
ax[2].grid(True)

# --- One legend for everything (collect handles from all axes) ---
handles, labels = [], []
for a in (ax[0], ax0r, ax[1], ax1r, ax[2]):
    h, l = a.get_legend_handles_labels()
    handles += h
    labels += l

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           fancybox=True, ncol=4)

plt.tight_layout()
plt.show()