import numpy as np
import sklearn
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def LinearRegression(X,Y):
    '''
    :param X: matrix with the regressors
    :param Y: array with the dependent variables
    :return: array with the parameters of the regression
    '''

    first_half = np.linalg.inv(X.T @ X) #(X.T @ X)⁻1 -> [n,m]@[m,n] = [n,n]
    seconds_half = X.T @ Y #(X.T @ Y) -> [n,m]@[m,1] = [n,1]
    return first_half@seconds_half #array of parameters [n,n]@[n,1] = [n,1]

#load the ice thickness data from
data = np.load("./Ice_thickness_trace/ice_thickness_vs_time.npy", allow_pickle=True)
timestamps = data[:,0] #slice the array to acquire the timestamps
seconds = (timestamps.astype('datetime64[s]')-timestamps.astype('datetime64[D]')).astype('timedelta64[s]').astype(int)
thick = data[:,1].astype("float") #slice the array to acquire the thickness
init_idx = np.argmin(np.abs(thick)) #find where thickness starts at 0 mm
thick = thick[init_idx+1:]
timestamps = timestamps[init_idx+1:]
rel_seconds = seconds[init_idx+1:]-seconds[init_idx+1] #relative timestamp

#the regression model proposed is s(t) = a0 + a1*t
#where t is the time in seconds
X = np.ones(shape=(len(rel_seconds),1))
X = np.concatenate((X, rel_seconds[:,np.newaxis]), axis=1) #a0 + a1*x
#X =  rel_seconds[:,np.newaxis]
Y = np.expand_dims(thick, axis=1)
theta = LinearRegression(X, Y) #regression parameters
y_hat = X@theta #predictions given the regression parameters (Y = X@theta)
r2 = r2_score(thick, y_hat)

plt.figure(1)
leg = []
plt.scatter(rel_seconds, thick)
leg.append("Measured thickness")
plt.plot(rel_seconds, y_hat, color="tab:orange")
leg.append("Linear regression")
plt.xlabel("Frame Time [s]")
plt.ylabel("ICE Thickness [mm]")
plt.legend(leg)
plt.grid(True)
plt.gcf().autofmt_xdate() #nicer datetime ticks
plt.show()
