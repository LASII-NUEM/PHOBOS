import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

outFolder   = '../PDI/ice_thickness_trace/'
datapath = os.path.join(outFolder, "ice_thickness_vs_time.npy")
plotpath = os.path.join(outFolder, "ice_thickness_plot.pdf")
data = np.load(datapath, allow_pickle=True)

frametime = data[:, 0]
thickness_array = data[:, 1]

# frametime = [datetime.fromtimestamp(t) for t in frametime_sec]

plt.figure(figsize=(15,5))
plt.plot(frametime, thickness_array, 's-')
for i, value in enumerate(thickness_array):
    x = frametime[i] + timedelta(seconds=0.1)   # x-offset in time
    y = thickness_array[i] + 0.8                # y-offset in mm
    plt.text(x, y, f"{value:.2f}", fontsize=9, color='black')
plt.xlabel("Frame Time")
plt.ylabel("ICE Thickness [mm]")
plt.grid(True)
plt.gcf().autofmt_xdate()  # nicer datetime ticks
plt.savefig(plotpath, format="pdf", bbox_inches="tight")  # <- save here
plt.show()