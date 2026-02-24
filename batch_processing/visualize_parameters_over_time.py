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

#plot the parameters over time
normalize = False
kernel_size = 20
key_model = "Longo2020"
algorithm = ["BFGS", "NLLS", "DLS", "Nelder-Mead Simplex"]
colors = ["tab:orange", "tab:green", "tab:purple", "tab:red"]
#arrays to store the parameters
circuit_params = fitting_utils.function_handlers[key_model.lower()]["fit_params"]
data = batch_data.fit_data
params_model = data[key_model]["params"]
params_overtime = np.zeros(shape=(len(params_model),len(params_model[0][0]),len(params_model[0])))
for i in range(len(params_model)):
    params_overtime[i,:,:] = np.array(params_model[i]).T

#extract C from tau if Longo2020
if key_model == "Longo2020":
    params_overtime[:,1,:] = params_overtime[:,1,:]/params_overtime[:,0,:] #C1 = tau1/R1
    params_overtime[:,3,:] = params_overtime[:,3,:]/params_overtime[:,2,:] #C2 = tau2/R2
    params_overtime[:,5,:] = params_overtime[:,5,:]/params_overtime[:,4,:] #Q = tau3/R3
    circuit_params = ['R1', 'C1', 'R2', 'C2', 'R3', 'Q', 'n3', 'tau4']

for i in range(len(circuit_params)):
    plt.subplot(1, len(circuit_params),i+1)
    leg = []
    y_max = 0
    y_min = np.inf
    for j in range(0,1):
        filtered_params = ndimage.median_filter(params_overtime[:,i,j], size=kernel_size)

        if not normalize:
            if np.max(filtered_params) > y_max:
                y_max = np.max(filtered_params)

            if np.min(filtered_params) < y_min:
                y_min = np.min(filtered_params)

            plt.plot(filtered_params, color=colors[j])
        else:
            plt.plot(filtered_params/np.max(filtered_params), color=colors[j])

        leg.append(f'{algorithm[j]}')

    if not normalize:
        plt.vlines(10, ymin=y_min, ymax=y_max, colors='black', linestyles='dotted')
    else:
        plt.vlines(10, ymin=0.5, ymax=1, colors='black', linestyles='dotted')

    leg.append("freezing instant")
    plt.title(circuit_params[i])
    plt.xlabel('samples')
    plt.legend(leg, prop={'size': 6})
    plt.grid()
    plt.suptitle(key_model)
    plt.show()

