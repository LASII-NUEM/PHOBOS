import numpy as np
from framework import file_lcr, characterization_utils, fitting_utils
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#read the files
filename_flange = '../data/testICE_30_01_26/c_test.csv'
spec_ice_obj = file_lcr.read(filename_flange, n_samples=1, electrode="cell", acquisition_mode="freq", aggregate=np.mean)
z_meas_real, z_meas_imag = characterization_utils.complex_impedance(spec_ice_obj, spec_ice_obj.freqs)
z_meas = z_meas_real-1j*z_meas_imag
freqs_mask = spec_ice_obj.freqs > 100
spec_ice_obj.freqs = spec_ice_obj.freqs[freqs_mask]
z_meas_real = z_meas_real[:,:,freqs_mask]
z_meas_imag = z_meas_imag[:,:,freqs_mask]
z_meas = z_meas[:,:,freqs_mask]
z_hat_real = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
z_hat_imag = np.zeros(shape=(len(z_meas_real), 1, len(spec_ice_obj.freqs)))
chi_sqr = np.zeros(shape=(len(z_hat_real,)))
M = np.zeros(shape=(len(z_hat_real,)))

#kramers-kronig algorithm
c = 0.5
max_iter = 100
for i in range(len(z_meas_real)):
    fit_obj = fitting_utils.LinearKramersKronig([z_meas_real[i,0,:], z_meas_imag[i,0,:]], spec_ice_obj.freqs, c=c, max_iter=max_iter, add_capacitor=True)
    z_hat_real[i,0,:] = fit_obj.z_hat_real
    z_hat_imag[i,0,:] = fit_obj.z_hat_imag
    chi_sqr[i] = fit_obj.chi_square
    M[i] = fit_obj.fit_components

fig, ax = plt.subplots()
nyquist_meas = ax.plot(z_meas_real[0,0,:], z_meas_imag[0,0,:], color="tab:blue", label="measured")[0]
linKK_meas = ax.plot(z_hat_real[0,0,:], -z_hat_imag[0,0,:], color="tab:orange", linestyle="dotted", label="linear kramers-kronig")[0]
ax.legend()
ax.grid()
ax.title.set_text(f"{spec_ice_obj.human_timestamps[0]}")
ax.set_xlabel("Z'")
ax.set_ylabel("Z''")

def update_nyquist(frame):
    frame += 1
    real_part = z_meas_real[frame,0,:]
    imag_part = z_meas_imag[frame,0,:]
    limits = 10
    nyquist_meas.set_xdata(real_part)
    nyquist_meas.set_ydata(imag_part)
    linKK_meas.set_xdata(z_hat_real[frame,0,:])
    linKK_meas.set_ydata(-z_hat_imag[frame,0,:])
    ax.title.set_text(f"{spec_ice_obj.human_timestamps[frame]} \n"
                      f"M = {M[frame]} \n"
                      f"x² = {chi_sqr[frame]}")
    plt.xlim([np.min(real_part)-limits, np.max(real_part)+limits])
    plt.ylim([np.min(imag_part)-limits, np.max(imag_part)+limits])

    return nyquist_meas

animate = animation.FuncAnimation(fig=fig, func=update_nyquist, frames=len(z_hat_real)-1, interval=300)
plt.show()
