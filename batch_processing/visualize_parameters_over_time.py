import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import os
from framework import fitting_utils
from scipy import ndimage

#load the processed files
#see: ./dump_batch_impedance_fitting.py to generate the required .pkl file
batch_data_path = f'../data/testICE_30_01_26/batchfit_testICE_30_01_26.pkl'

if not os.path.isfile(batch_data_path):
    raise FileNotFoundError(f'[visualize_parameters_over_time] Fit batch data file {batch_data_path} not found!')

with open(batch_data_path, 'rb') as handle:
    batch_data = pickle.load(handle)

#freerun sweep
freqs = batch_data.media_obj.freqs
human_timestamps = batch_data.media_obj.human_timestamps

#arrays to store the parameters
idx_model = 0
key_model = list(batch_data.fit_data.keys())[idx_model]
circuit_params = fitting_utils.function_handlers[key_model.lower()]["fit_params"]
data = batch_data.fit_data
params_model = data[key_model]["params"]
params_overtime = []
for i in range(len(params_model)):
    params_overtime.append(params_model[i][idx_model])
params_overtime = np.array(params_overtime) #convert to numpy array

#plot the parameters over time
plt.figure(1)
kernel_size = 9
for i in range(len(circuit_params)):
    plt.subplot(1, len(circuit_params), i+1)

    #extract C from tau if Longo2020
    if key_model == "Longo2020":
        params_overtime[1] = params_overtime[1]/params_overtime[0] #C1 = tau1/R1
        params_overtime[3] = params_overtime[3]/params_overtime[2] #C2 = tau2/R2
        params_overtime[5] = params_overtime[5]/params_overtime[4] #Q = tau3/R3
        circuit_params = ['R1','C1','R2','C2','R3','Q','n3','tau4']

    filtered_params = ndimage.median_filter(params_overtime[:,i], size=kernel_size)
    leg = []
    plt.plot(filtered_params)
    leg.append("predicted parameters")
    plt.vlines(10, ymin=np.min(filtered_params), ymax=np.max(filtered_params), colors='red', linestyles='dotted')
    leg.append("freezing instant")
    plt.title(circuit_params[i])
    plt.xlabel('samples')
    plt.legend(leg, prop={'size': 6})
    plt.grid()
plt.suptitle(key_model)
plt.show()