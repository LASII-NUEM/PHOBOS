import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# Numpy is a powerful Python package for processing data vectorized. However, what does "vectorized" mean?
# In synthesis, a vector-oriented package allows you to compute a function over all the elements of an array/matrix at the same time, without iterating over each element at a time.
# For instance, given the arbitrary signal:

fs = 10e3 #sampling frequency in Hz
fc = 60 #central frequency of the signal
t_init = 0 #initial timestamp
t_end = 1 #final timestamp
Ts = 1/fs #sampling period
t = np.arange(t_init, t_end+Ts, Ts) #array with all the time samples
print(f"[first element] {t[0]} s")
print(f"[last element] {t[-1]} s")

# For now, how this sinusoidal pulse is generated is not really relevant, we'll get to it later ;)
signal = np.sin(2*np.pi*fc*t)

# Another *very* important feature of programming in general (this is not exclusive to Python!) is FUNCTIONS.
# They allow you to compute the same operation multiple times, subject to changes in the input arguments.
# In practice, this means only one line of code each time you need to repeat the computation.
# For example, if we wanted to create multiple Gaussian pulses, a function that takes the frequencies and timestamps as input would be much more interesting than repeating the 6 lines over and over!

def generate_signal(fs, fc, t_init=0, t_end=1e-2):
    '''
    :param fs: sampling frequency in Hz
    :param fc: central frequency in Hz
    :param t_init: initial timestamp in s (optional argument, 0 by default)
    :param t_end: final timestamp in s (optional argument, 1.5e-6 by default)
    :return: the Gaussian pulse signal
    '''

    Ts = 1/fs #sampling period
    t = np.arange(t_init, t_end+Ts, Ts) #array with all the time samples
    signal = np.sin(2*np.pi*fc*t)

    return signal

func_signal = generate_signal(fs, fc, t_init=t_init, t_end=t_end) #create the same array for the signal but with a function
samples = np.arange(0, len(func_signal), 1) #create len(func_signal) equally spaced (by 1) samples
res = np.abs(signal-func_signal) #|signal[i]-func_signal[i]|
print(f'[function vs. line of code] are both arrays the same? {np.sum(res)}')

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
    diff = np.zeros_like(signal)
    diff[0] = 0
    for i in range(1, len(signal)-1):
        diff[i] = signal[i+1] - signal[i-1] #x'[i] = x[i] - x[i-1]

    return diff/2

def central_fofd_vectorized(signal):
    '''
    :param signal: a signal that the finite difference will be computed over
    :return: the differentiated signal
    '''

    advanced_signal = np.roll(signal, -1) #x[i+1]
    lagged_signal = np.roll(signal, 1) #x[i-1]
    diff = advanced_signal-lagged_signal #x'[i] = x[i+1]-x[i-1]
    return diff/2

#FOFD with looping
t_init_loop = time.time()
diff_loop = central_fofd_loop(signal)
print(f'[fofd loop] t_elapsed = {time.time()-t_init_loop} s')

#FOFD with vectorizing
t_init_vect = time.time()
diff_vect = central_fofd_vectorized(signal)
print(f'[fofd vectorized] t_elapsed = {time.time()-t_init_vect} s')

res_diff = np.abs(diff_loop-diff_vect)
print(f'[loop vs. vectorized] are both arrays the same? {np.sum(res)}')

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
plt.plot(samples, np.cos(2*np.pi*fc*t))
leg.append("cos(2piwt)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend(leg)
plt.grid()
plt.show()