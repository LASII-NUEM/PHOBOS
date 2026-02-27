from framework import file_lvm
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

filename = '../data/PxT_leaktest/test_pxt_26_02.lvm'
lvm_obj = file_lvm.read(filename, setup = "lulacell")

# teste PxT Lulacell
pressure = np.asarray(lvm_obj.measured_pressure, dtype=float).ravel()
pressure_std = np.std(pressure)
print("Standard deviation Pressure:",pressure_std, "bar")
pressure = pressure*100 #KPa to KPa

temp = np.asarray(lvm_obj.measured_temp, dtype=float).ravel()
temp_std = np.std(temp)
print("Standard deviation Temperature:",temp_std, "°C")
temp = temp+273.15 #K to K

time = np.asarray(lvm_obj.human_timestamp)

pressure_mean = moving_average(pressure, 1000)
pressure_mean = pressure_mean[:-1]
temp_mean = moving_average(temp, 1000)
temp_mean = temp_mean[:-1]

t0 = time[0]
time_num = (time - t0) / np.timedelta64(1, "s")     # seconds as float
time_py = time.astype('datetime64[ms]').astype(datetime)
idx1 = 20000
idx2 = 150000
idx3 = 625000
# Temperature plot
plt.plot(time_num, temp, '.',color='blue', label="measured")
plt.axvline(x=time_num[idx1], color='red', linestyle='dashed')
plt.text(time_num[idx1],
         temp[idx1]-0.1,
         f"{time_py[idx1].strftime('%H:%M:%S')}",
         fontsize=8,
         color='green')

plt.axvline(x=time_num[idx2], color='red', linestyle='dashed')
plt.text(time_num[idx2],
         temp[idx2]-0.1,
         f"{time_py[idx2].strftime('%H:%M:%S')}",
         fontsize=8,
         color='green')

plt.axvline(x=time_num[idx3], color='red', linestyle='dashed')
plt.text(time_num[idx3],
         temp[idx3]+0.35,
         f"{time_py[idx3].strftime('%H:%M:%S')}",
         fontsize=8,
         color='green')
plt.legend()
plt.title("Temperature leak test")
plt.xlabel("time [s]")
plt.ylabel("Temperature [K]")
plt.show()

# Pressure plot
plt.plot(time_num, pressure, 'o',color='blue', label="measured")
plt.axvline(x=time_num[idx1], color='red', linestyle='dashed')
plt.text(time_num[idx1],
         pressure[idx1]-100,
         f"{time_py[idx1].strftime('%H:%M:%S')}",
         fontsize=8,
         color='green')

plt.axvline(x=time_num[idx2], color='red', linestyle='dashed')
plt.text(time_num[idx2],
         pressure[idx2]-100,
         f"{time_py[idx2].strftime('%H:%M:%S')}",
         fontsize=8,
         color='green')

plt.axvline(x=time_num[idx3], color='red', linestyle='dashed')
plt.text(time_num[idx3],
         pressure[idx3]+100,
         f"{time_py[idx3].strftime('%H:%M:%S')}",
         fontsize=8,
         color='green')
plt.title("Pressure leak test - idx")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("Pressure [KPa]")
plt.show()

# PxT  plot
plt.plot(temp, pressure,color='blue', linestyle='dashed',label="measured")
plt.plot(temp_mean,pressure_mean, color='red', linestyle='dotted',label="mean")
plt.title("PxT leak test")
plt.legend()
plt.xlabel("Temperature [K]")
plt.ylabel("Pressure [KPa]")
plt.show()

# [PxT] x t plot
x = time_num
y = pressure / temp

# Regressão linear
a, b = np.polyfit(x, y, 1)
y_fit = a * x + b
# Plot original

plt.figure(figsize=(8,5))
plt.plot(x, y, color='blue', linestyle='dashed', label="Measured")
# Plot regressão
plt.plot(x, y_fit, color='red', linewidth=2,
         label=f"Fit: y = {a:.3e}x + {b:.3e}")

plt.title("[P/T] vs Time – Leak Test")
plt.ylabel("Pressure / Temperature [kPa/K]")
plt.xlabel("Time [s]")
plt.legend()
plt.grid(True)
plt.show()
