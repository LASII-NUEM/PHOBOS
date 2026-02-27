import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import ndimage
from datetime import datetime
from sklearn.metrics import r2_score

def LinearRegression(X,Y):
    '''
    :param X: matrix with the regressors
    :param Y: array with the dependent variables
    :return: array with the parameters of the regression
    '''

    first_half = np.linalg.inv(X.T @ X) #(X.T @ X)⁻1 -> [n,m]@[m,n] = [n,n]
    seconds_half = X.T @ Y #(X.T @ Y) -> [n,m]@[m,1] = [n,1]
    return first_half@seconds_half #array of parameters [n,n]@[n,1] = [n,1]

data = loadmat('./Ice_thickness_trace/thickness_data.mat') #load .mat file
inner_radius = (data['inner_radius']/2).flatten()
inner_radius = inner_radius[2077:]

#generate timestamps
frame_init = datetime.fromisoformat("2025-12-01 12:01:21" + "+00:00").timestamp()
frame_end = datetime.fromisoformat("2025-12-01 13:33:35" + "+00:00").timestamp()
absolute_seconds = np.linspace(frame_init, frame_end, num=len(inner_radius))
rel_seconds = absolute_seconds-absolute_seconds[0]

#filtering
kernel_size = 25
med_filt_radius = ndimage.median_filter(inner_radius, size=kernel_size) #median filtering
smooth_filt_radius = ndimage.gaussian_filter1d(med_filt_radius, 180)

#liner regression
#s(t) = a0 + a1.t
X_st = np.ones(shape=(len(smooth_filt_radius),1))
X_st = np.concatenate((X_st, rel_seconds[:,np.newaxis]), axis=1) #a0 + a1*t
theta_st = LinearRegression(X_st, smooth_filt_radius[:, np.newaxis]) #s(t) x t regression parameters
y_hat_st = X_st@theta_st #compute the predictions

#s(t) = a0 + a1.√t
X_sqrt = np.ones(shape=(len(smooth_filt_radius),1))
X_sqrt = np.concatenate((X_sqrt, np.sqrt(rel_seconds[:,np.newaxis])), axis=1) #a0 + a1*√t
theta_sqrt = LinearRegression(X_sqrt, smooth_filt_radius[:, np.newaxis]) #s(t) x √t regression parameters
y_hat_sqrt = X_sqrt@theta_sqrt #compute the predictions

#log[s(t)] = a0 + a1.log(t)
X_log = np.ones(shape=(len(smooth_filt_radius[1:]),1))
X_log = np.concatenate((X_log, np.log10(rel_seconds[1:,np.newaxis]+1e-8)), axis=1) #a0 + a1*log(t)
theta_log = LinearRegression(X_log, np.log10(smooth_filt_radius[1:, np.newaxis]+1e-8)) #log[s(t)] x log(t) regression parameters
y_hat_log = X_log@theta_log #compute the predictions

plt.figure(1)
leg = []
plt.plot(inner_radius)
leg.append("Raw data")
plt.plot(rel_seconds, med_filt_radius, color="tab:orange")
leg.append("Med. filter")
plt.plot(rel_seconds, smooth_filt_radius, color="tab:green")
leg.append("Med. + Gauss. filter")
plt.grid()
plt.legend(leg)
plt.ylabel('Radial thickness [mm]')
plt.xlabel('Relative time [s]')
plt.suptitle('Thickness trace smoothing')
plt.show()

plt.figure(2)
leg = []
plt.subplot(3,1,1)
plt.scatter(rel_seconds, smooth_filt_radius)
leg.append('s(t)')
plt.plot(rel_seconds, y_hat_st, color="tab:orange")
leg.append(f's(t) = {theta_st[0].item()}+{theta_st[1].item()}.t \n'
           f'r² = {r2_score(smooth_filt_radius, y_hat_st)}')
plt.xlabel('t[s]')
plt.legend(leg)
plt.tight_layout()
plt.grid()

plt.subplot(3,1,2)
leg = []
plt.scatter(np.sqrt(rel_seconds), smooth_filt_radius)
leg.append('s(t)')
plt.plot(np.sqrt(rel_seconds), y_hat_sqrt, color="tab:orange")
leg.append(f's(t) = {theta_sqrt[0].item()}+{theta_sqrt[1].item()}.√t \n'
           f'r² = {r2_score(smooth_filt_radius, y_hat_sqrt)}')
plt.xlabel('√t')
plt.legend(leg)
plt.tight_layout()
plt.grid()

plt.subplot(3,1,3)
leg = []
plt.scatter(np.log10(rel_seconds[1:]+1e-8), np.log10(smooth_filt_radius[1:]+1e-8))
leg.append('log[s(t)]')
plt.plot(np.log10(rel_seconds[1:]+1e-8), y_hat_log, color="tab:orange")
leg.append(f'log[s(t)] = {theta_log[0].item()}+{theta_log[1].item()}.log(t) \n'
           f'r² = {r2_score(smooth_filt_radius[1:], y_hat_log)}')
plt.xlabel('log(t)')
plt.grid()
plt.legend(leg)
plt.tight_layout()
plt.show()