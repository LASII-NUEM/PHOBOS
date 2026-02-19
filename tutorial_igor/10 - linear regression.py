import numpy as np
import datetime
from sklearn import linear_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Now, let's apply some of the concepts we learned prior and implement a Linear Regression algorithm from scratch
# (and then compare to a built-in method from scipy)

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
data = np.load("./ice_thickness_data/ice_thickness_vs_time.npy", allow_pickle=True)
timestamps = data[:,0] #slice the array to acquire the timestamps
timestamps = timestamps[5:] #slice from the 5th element
seconds = (timestamps.astype('datetime64[s]')-timestamps.astype('datetime64[D]')).astype('timedelta64[s]').astype(int)
rel_seconds = seconds-seconds[0] #relative seconds startings at the first timestamp
thick = data[:,1] #slice the array to acquire the thickness
thick = thick[5:] #slice from the 5th element

#the regression model proposed is s(t) = a0 + a1*t
#where t is the time in seconds
X = np.ones(shape=(len(rel_seconds),1))
X = np.concatenate((X, rel_seconds[:,np.newaxis]), axis=1) #a0 + a1*x
Y = np.expand_dims(thick, axis=1)
theta = LinearRegression(X, Y) #regression parameters
y_hat = X@theta #predictions given the regression parameters (Y = X@theta)

plt.figure(1)
leg = []
plt.scatter(timestamps, thick)
leg.append("Measured thickness")
for i, value in enumerate(thick):
    x = timestamps[i] + datetime.timedelta(seconds=0.1) #x-offset in time
    y = thick[i] + 0.8 #y-offset in mm
    plt.text(x, y, f"{value:.2f}", fontsize=9, color='black')
plt.plot(timestamps, y_hat, color="tab:orange")
leg.append("Linear regression")
plt.xlabel("Frame Time")
plt.ylabel("ICE Thickness [mm]")
plt.grid(True)
plt.gcf().autofmt_xdate() #nicer datetime ticks
plt.show()

