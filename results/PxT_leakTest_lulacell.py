from framework import file_lvm
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

filename = '../data/PxT_leaktest/test_pxt.lvm'
lvm_obj = file_lvm.read(filename, setup = "lulacell")

target = 3544.899256
idx =np.where(np.isclose(lvm_obj.relative_timestamp, target, rtol=0, atol=1e-9))[0]
idx = int(idx[0])

# teste PxT Lulacell
pressure = np.asarray(lvm_obj.measured_pressure, dtype=float).ravel()
temp = np.asarray(lvm_obj.measured_temp, dtype=float).ravel()
time = np.asarray(lvm_obj.human_timestamp)

pressure_mean = moving_average(pressure, 1000)
pressure_mean = pressure_mean[:-1]
temp_mean = moving_average(temp, 1000)
temp_mean = temp_mean[:-1]

t0 = time[0]
time_num = (time - t0) / np.timedelta64(1, "s")     # seconds as float
coefT = np.polyfit(time_num[idx:], temp[idx:], 5)
temp_fit = np.polyval(coefT, time_num)

plt.plot(time_num[idx:], temp[idx:], '.',color='blue', label="measured")
plt.plot(time_num[idx:], temp_fit[idx:], '-',color='red', label="poly fit (deg 5)")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("Temperature [°C]")
plt.show()

coefP = np.polyfit(time_num[idx:],pressure[idx:],1)
pressure_fit= np.polyval(coefP, time_num)

plt.plot(time_num[idx:], pressure[idx:], 'o',color='blue', label="measured")
plt.plot(time_num[idx:], pressure_fit[idx:], '-',color='red', label="poly fit (deg 1)")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("Pressure [bar]")
plt.show()

coefPT = np.polyfit(temp[idx:],pressure[idx:],1)
poly1d_fn = np.poly1d(coefPT)

plt.plot(temp[idx:],pressure[idx:], color='blue',linestyle='dashed')
plt.plot(temp_mean[idx:],pressure_mean[idx:], color='red',linestyle='dotted')
# plt.plot(temp[idx:],pressure[idx:], 'r.', temp[idx:], poly1d_fn(temp[idx:]), '--k')
plt.xlabel("Pressure [bar]")
plt.ylabel("Temperature [°C]")
plt.show()

plt.plot(temp,pressure, color='blue', linestyle='dashed')
plt.plot(temp_mean,pressure_mean, color='red', linestyle='dotted')
plt.xlabel("Pressure [bar]")
plt.ylabel("Temperature [°C]")
plt.show()
