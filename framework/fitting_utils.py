from framework import characterization_utils, data_types, equivalent_circuits
import numpy as np
from scipy.optimize import curve_fit, minimize
import time
from functools import partial

#dictionary to handle function calls -> number of expected params and the function pointer
function_handlers = {
    "longo2020": {"n_params": 8, "function_ptr": equivalent_circuits.Longo2020, "partial_function_ptr": equivalent_circuits.Longo2020_partial, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,1), (0,np.inf)]},
    "fouquet2005": {"n_params": 6, "function_ptr": equivalent_circuits.Fouquet2005, "partial_function_ptr": equivalent_circuits.Fouquet2005_partial, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,1), (0,np.inf), (0,np.inf)]},
    "zurich2021": {"n_params": 6, "function_ptr": equivalent_circuits.Zurich2021, "partial_function_ptr": equivalent_circuits.Zurich2021_partial, "bounds": [(0,np.inf), (0,np.inf), (0,1), (0,np.inf), (0,np.inf), (0,np.inf)]},
    "awayssa2025": {"n_params": 5, "function_ptr": equivalent_circuits.Awayssa2025, "partial_function_ptr": equivalent_circuits.Awayssa2025_partial, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf)]},
    "hong2021": {"n_params": 7, "function_ptr": equivalent_circuits.Hong2021, "partial_function_ptr": equivalent_circuits.Hong2021_partial, "bounds": [(0,np.inf), (0,np.inf), (0,1), (0,np.inf), (0,np.inf), (0,1), (0,np.inf)]}
}

class OptimizerResults:
    def __init__(self, opt_params=None, opt_params_scaled=None, opt_cost=None, opt_fit=None, nmse_score=None, n_iter=None, t_elapsed=None):
        if opt_params is not None:
            self.opt_params = opt_params
        if opt_params_scaled is not None:
            self.opt_params_scaled = opt_params_scaled
        if opt_cost is not None:
            self.opt_cost = opt_cost
        if opt_fit is not None:
            self.opt_fit = opt_fit
        if nmse_score is not None:
            self.nmse_score = nmse_score
        if n_iter is not None:
            self.n_iter = n_iter
        if t_elapsed is not None:
            self.t_elapsed = t_elapsed

class EquivalentCircuit:
    def __init__(self, topology: str, data_medium:data_types.SpectroscopyData, freqs:np.ndarray):
        '''
        :param topology: The circuit topoly that outputs the modeled impedance of the expected circuit
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        '''

        #validate topology
        topology = topology.lower() #convert to lower case
        valid_topologies = list(function_handlers.keys()) #expected parameters for the available models
        if topology not in valid_topologies:
            raise ValueError(f'[EquivalentCircuit] {topology} not implemented! Try: {valid_topologies}')
        self.topology = topology
        self.circuit_impedance = None
        self.fit_method = None

        #validate data_medium
        if not isinstance(data_medium, data_types.SpectroscopyData):
            raise TypeError(f'[EquivalentCircuit] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
        self.data_medium = data_medium

        #validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs

        #compute the measured impedance from the SpectroscopyData objects
        z_meas_real, z_meas_imag = characterization_utils.complex_impedance(data_medium, freqs)
        self.z_meas_real = z_meas_real
        self.z_meas_imag = z_meas_imag
        self.z_meas = z_meas_real - 1j*z_meas_imag #complex impedance

    def CUMSE(self, theta, args):
        '''
        :param z_hat: the complex impedance computed from the fitted circuit
        :param z_meas: the complex impedance measured from the real system
        :return: the mean squared error between the measured and fitted impedance values
        '''

        z_hat = self.circuit_impedance(theta, [args[1], args[2]]) #compute the model for the arguments
        z_hat = z_hat.astype('complex')
        args[0] = args[0].astype('complex')
        SSE = np.sum(((args[0].real-z_hat.real)**2)+((args[0].imag-z_hat.imag)**2))
        #SSE = np.sum((np.abs(args[0]) - np.abs(z_hat))**2) #sum of squared errors
        return SSE / len(z_hat)

    def fit_circuit(self, initial_guess:np.ndarray, scaling_array:np.ndarray, method='BFGS'):
        '''
        :param initial_guess: the initial guess for the fit to run the iterative algorithms
        :param scaling_array: scale all the search parameters to avoid exploding gradients
        :param method: which optimization algorithm will be used to fit the circuit data
        :return: the parameters the best fit the expected equivalent circuit
        '''

        #validate "method"
        valid_methods = ['BFGS', 'NLLS']
        if method not in valid_methods:
            raise ValueError(f'[EquivalentCircuit] {method} not implemented! Try: {valid_methods}')
        self.fit_method = method

        #validate "initial_guess"
        if len(initial_guess) != function_handlers[self.topology]["n_params"]:
            raise ValueError(f'[EquivalentCircuit] The number of initial guess parameters do not match with the given model! Should be {function_handlers[self.topology]["n_params"]}.')

        if not isinstance(initial_guess, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "initial_guess" must be a Numpy Array! Curr. type = {type(initial_guess)}')

        #validate "scaling_array"
        if len(scaling_array) != len(initial_guess):
            raise ValueError(f'[EquivalentCircuit] Length of the scaling array do not match the length of the initial guess! {len(scaling_array)} != {len(initial_guess)}')

        if not isinstance(scaling_array, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "scaling_array" must be a Numpy Array! Curr. type = {type(scaling_array)}')

        #run the parameter search
        omega = 2*np.pi*self.freqs #Hz to rad/s
        bounds = function_handlers[self.topology]["bounds"] #optimization boundaries
        if self.fit_method == "BFGS":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()
            fit_obj = minimize(self.CUMSE, initial_guess, args=([self.z_meas, omega, scaling_array],), bounds=bounds, method='L-BFGS-B')
            t_elapsed = time.time() - t_init
            print(f'[EquivalentCircuit] BFGS fit elapsed time = {t_elapsed} s')
            opt_fit = self.circuit_impedance(fit_obj.x, [omega, scaling_array]) #compute the circuit for the optimal values
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            return OptimizerResults(opt_params=fit_obj.x, opt_params_scaled=fit_obj.x*scaling_array, opt_cost=fit_obj.fun,
                                    opt_fit=opt_fit, nmse_score=nmse, n_iter=fit_obj.nit, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == "NLLS":
            self.circuit_impedance = function_handlers[self.topology]["partial_function_ptr"]
            bounds = np.array(bounds) #convert the boundaries to numpy array
            bounds = ((bounds[:, 0]), (bounds[:, 1])) #curve_fit receives bounds as tuple

            #handle the real and imaginary parts separately
            circuit_impedance_real = partial(self.circuit_impedance, scaling=scaling_array, return_type="real") #set scaling and return_type to static inputs
            circuit_impedance_imag = partial(self.circuit_impedance, scaling=scaling_array, return_type="imag") #set scaling and return_type to static inputs
            t_init = time.time()
            fit_params_real, fit_cov_real = curve_fit(circuit_impedance_real, omega, self.z_meas_real, p0=initial_guess, bounds=bounds)
            fit_params_imag, fit_cov_imag = curve_fit(circuit_impedance_imag, omega, self.z_meas_imag, p0=initial_guess, bounds=bounds)
            t_elapsed = time.time() - t_init
            print(f'[EquivalentCircuit] NLLS fit elapsed time = {t_elapsed} s')
            opt_fit_real = function_handlers[self.topology]["function_ptr"](fit_params_real, [omega, scaling_array]) #compute the circuit real output for the optimal values
            opt_fit_imag = function_handlers[self.topology]["function_ptr"](fit_params_imag, [omega, scaling_array]) #compute the circuit imaginary output for the optimal values
            opt_fit = opt_fit_real + 1j*opt_fit_imag #complex impedance of the fit
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            return OptimizerResults(opt_params=[fit_params_real, fit_params_imag], opt_params_scaled=[fit_params_real*scaling_array, fit_params_imag]*scaling_array,
                                    opt_fit=opt_fit, nmse_score=nmse, t_elapsed=t_elapsed) #return the optimized parameters

        else:
            raise ValueError(f'[EquivalentCircuit] method = {method} not implemented! Try: {valid_methods}')


    def NMSE(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: NMSE of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        #normalized mean squared error
        SSE = np.sum(((z_hat.real-z.real)**2) + ((z_hat.imag-z.imag)**2)) #sum of squared errors
        SSO = np.sum((z.real**2) + (z.imag**2)) #sum of squared measurements

        return SSE/SSO





