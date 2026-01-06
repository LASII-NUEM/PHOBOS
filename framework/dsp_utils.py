import numpy as np

def first_order_backward_diff(signal, eps=1):
    '''
    :param signal: signal to compute the first order term
    :param eps: epsilon (smallest change possible)
    :return: numerical first order backward derivative of the signal
    '''

    #validate inputs
    if not isinstance(signal, np.ndarray):
        raise TypeError(f'[first_order_backward_diff] Input signal must be a numpy array!')

    if len(signal) == 0:
        raise ValueError(f'[first_order_backward_diff] Input signal is an empty array!')

    if eps<=0:
        raise ValueError(f'[first_order_backward_diff] epsilon must be greater than 0!')

    #first order term
    backward = np.roll(signal, eps, axis=0) #f[i-eps]
    backward[0] = signal[0]
    return (signal-backward)/eps #(f[i]-f[i-1])/h

def first_order_central_diff(signal, eps=1):
    '''
    :param signal: signal to compute the first order term
    :param eps: epsilon (smallest change possible)
    :return: numerical first order central derivative of the signal
    '''

    #validate inputs
    if not isinstance(signal, np.ndarray):
        raise TypeError(f'[first_order_backward_diff] Input signal must be a numpy array!')

    if len(signal) == 0:
        raise ValueError(f'[first_order_backward_diff] Input signal is an empty array!')

    if eps<=0:
        raise ValueError(f'[first_order_backward_diff] epsilon must be greater than 0!')

    #first order term
    backward = np.roll(signal, eps, axis=0) #f[i-eps]
    backward[0] = signal[0]
    forward = np.roll(signal, -eps, axis=0) #f[i+eps]
    forward[-1] = signal[-1]
    return (forward-backward)/(2*eps) #(f[i+1]-f[i-1])/2h




