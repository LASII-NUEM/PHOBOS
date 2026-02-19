import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# Numpy is a powerful Python package for processing data vectorized. However, what does "vectorized" mean?
# In synthesis, a vector-oriented package allows you to compute a function over all the elements of an array/matrix at the same time, without iterating over each element at a time.
# For instance, given the arbitrary signal:

def generate_signal(fs, fc, t_init=0, t_end=1e-1):
    '''
    :param fs: sampling frequency in Hz
    :param fc: central frequency in Hz
    :param t_init: initial timestamp in s (optional argument, 0 by default)
    :param t_end: final timestamp in s (optional argument, 1.5e-6 by default)
    :return: the sinusoidal signal
    '''

    Ts = 1/fs #sampling period
    t = np.arange(t_init, t_end+Ts, Ts) #array with all the time samples
    signal = np.sin(2*np.pi*fc*t)

    return signal, t

f_samp = 10e3
f_central = 60
signal, t = generate_signal(f_samp, f_central, t_init=0, t_end=1) #create the same array for the signal but with a function
samples = np.arange(0, len(signal), 1) #create len(func_signal) equally spaced (by 1) samples

# Now, let's get our hands dirty! Create a function that computes the first-order central finite difference in the signal:
# x'[i] = (x[i+1] - x[i-1])/2
# That is, each sample is the difference between the next and the prior samples.
# Try both in a loop structure and vectorized, and compare the performance.

def central_fofd_loop(signal):
    '''
    :param signal: a signal that the finite difference will be computed over
    :return: the differentiated signal
    '''

    #iterate over every sample of the signal
    diff = np.zeros_like(signal) #array filled with zeros the size and type of 'signal'
    diff[0] = 0
    for i in range(1, len(signal)-1):
        diff[i] = signal[i+1] - signal[i-1] #x'[i] = x[i+1]-x[i-1]
    diff[-1] = 0

    return diff/2

def central_fofd_vectorized(signal):
    '''
    :param signal: a signal that the finite difference will be computed over
    :return: the differentiated signal
    '''

    advanced_signal = np.roll(signal, -1) #x[i+1]
    lagged_signal = np.roll(signal, 1) #x[i-1]
    diff = advanced_signal-lagged_signal #x'[i] = x[i+1]-x[i-1]
    diff[0] = 0
    diff[-1] = 0
    return diff/2

#FOFD with loop
t_init_loop = time.time()
diff_loop = central_fofd_loop(signal)
print(f'[fofd loop] t_elapsed = {time.time()-t_init_loop} s')

#FOFD with vectorizing
t_init_vect = time.time()
diff_vect = central_fofd_vectorized(signal)
print(f'[fofd vectorized] t_elapsed = {time.time()-t_init_vect} s')

res_diff = np.abs(diff_loop-diff_vect)
print(f'[loop vs. vectorized] are both arrays the same? {np.sum(res_diff) == 0}')

#now, let's plot the results!
plt.figure(1) #create a figure
plt.subplot(3,1,1) #a subplot with 3 lines and 1 column (subplot 1)
plt.plot(samples, signal)
plt.title("Signal")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3,1,2)
plt.plot(samples, diff_vect)
plt.title("FOFD")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

plt.subplot(3,1,3)
leg = []
plt.plot(samples, diff_vect/np.max(diff_vect))
leg.append("normalized FOFD")
plt.plot(samples, np.cos(2*np.pi*f_central*t))
leg.append("cos(2piwt)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend(leg)
plt.grid()
plt.show()
