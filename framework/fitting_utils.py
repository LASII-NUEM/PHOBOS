from framework import characterization_utils, data_types, equivalent_circuits
import numpy as np
from scipy.optimize import curve_fit, minimize
import time
from functools import partial
from impedance_fitting import script_to_overwrite_local_impedancedotpy_files

#dictionary to handle function calls -> number of expected params and the function pointer
function_handlers = {
    "longo2020": {"n_params": 8, "function_ptr": equivalent_circuits.Longo2020, "partial_function_ptr": equivalent_circuits.Longo2020_partial, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,1), (0,np.inf)]},
    "fouquet2005": {"n_params": 6, "function_ptr": equivalent_circuits.Fouquet2005, "partial_function_ptr": equivalent_circuits.Fouquet2005_partial, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,1), (0,np.inf), (0,np.inf)]},
    "zurich2021": {"n_params": 6, "function_ptr": equivalent_circuits.Zurich2021, "partial_function_ptr": equivalent_circuits.Zurich2021_partial, "bounds": [(0,np.inf), (0,np.inf), (0,1), (0,np.inf), (0,np.inf), (0,np.inf)]},
    "awayssa2025": {"n_params": 5, "function_ptr": equivalent_circuits.Awayssa2025, "partial_function_ptr": equivalent_circuits.Awayssa2025_partial, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf)]},
    "hong2021": {"n_params": 7, "function_ptr": equivalent_circuits.Hong2021, "partial_function_ptr": equivalent_circuits.Hong2021_partial, "bounds": [(0,np.inf), (0,np.inf), (0,1), (0,np.inf), (0,np.inf), (0,1), (0,np.inf)]},
    "yang2025": {"n_params": 6, "function_ptr": equivalent_circuits.Yang2025, "partial_function_ptr": equivalent_circuits.Yang2025_partial, "bounds": [(0, np.inf), (0, np.inf), (0, np.inf), (0, 1), (0, np.inf), (0, np.inf)]},
    "zhang2024": {"n_params": 6, "function_ptr": equivalent_circuits.Zhang2024, "partial_function_ptr": equivalent_circuits.Zhang2024_partial, "bounds": [(0, np.inf), (0, 1), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]}
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

class LinearKramersKronig:
    def __init__(self, data_medium:data_types.SpectroscopyData, freqs:np.ndarray, c=0.5, max_iter=100, add_capacitor=True):
        '''
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        :param c: threshold for overfitting criterion (mu)
        :param max_iter: maximum number of M RC pairs
        :param add_capacitor: flag to add a capacitor in series to the R_ohm+L+R_k block
        '''

        #validate data_medium
        if not isinstance(data_medium, data_types.SpectroscopyData):
            raise TypeError(f'[KramersKronig] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
        self.data_medium = data_medium

        #validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[KramersKronig] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs

        #compute the measured impedance from the SpectroscopyData objects
        z_meas_real, z_meas_imag = characterization_utils.complex_impedance(data_medium, freqs)
        self.z_meas_real = z_meas_real
        self.z_meas_imag = z_meas_imag
        self.z_meas = z_meas_real - 1j*z_meas_imag #complex impedance
        self.z_meas = self.z_meas.astype("complex")

        #fit routine
        self.tau = None #distribution of the time constants
        t_init = time.time()
        self.fit_params, self.fit_components = self.validate_data(self.z_meas, self.freqs, c=c, max_iter=max_iter, add_capacitor=add_capacitor) #run the LKK algorithm
        self.t_elapsed = time.time() - t_init #store the computation time of the algorithm
        self.z_hat_real, self.z_hat_imag = self.generate_MRCircuit(self.freqs, self.fit_params, self.tau) #compute the complex impedance for the fitted value
        self.z_hat = self.z_hat_real - 1j*self.z_hat_imag #complex impedance
        self.z_hat = self.z_hat.astype("complex")
        self.fit_residues_real, self.fit_residues_imag = self.compute_residues(self.z_meas, self. z_hat)

    def compute_residues(self, z, z_hat):
        '''
        :param z: the real measured complex impedance values
        :param z_hat: the fitted complex impedance values
        :return: the normalized residues for both real and imag parts
        '''

        z = z.astype("complex")
        z_hat = z_hat.astype("complex")

        return (z.real*z_hat.real)/np.abs(z), (z.imag*z_hat.imag)/np.abs(z)

    def compute_mu(self, Rk):
        '''
        :param Rk: the fitted Rk components from the linear kramers-kronig test
        :return: the overfitting criterion value
        '''

        neg_Rk = Rk[Rk<0] #negative signed ohmic components
        pos_Rk = Rk[Rk>=0] #positive signed ohmic components

        return 1 - np.sum(np.abs(neg_Rk))/np.sum(np.abs(pos_Rk))

    def validate_data(self, z_meas, freqs, c=0.5, max_iter=100, add_capacitor=True):
        '''
        :param z_meas: measured complex impedance values (real and imaginary)
        :param freqs: array with the swept frequencies
        :param c: threshold for overfitting criterion (mu)
        :param max_iter: maximum number of M RC pairs
        :param add_capacitor: flag to add a capacitor in series to the R_ohm+L+R_k block
        :return: the fitted elements of the M RC circuit
        '''

        M = 0 #variable to monitor the number of RC pairs added to the circuit
        z_meas = z_meas.astype("complex") #convert to complex object (extract real and imag value separately)
        omega = 2*np.pi*freqs #rad/s to Hz
        while True:
            M += 1 #update the M RC components number

            #distribution of time constants
            self.tau = np.zeros(shape=(M,)) #M time-constants
            tau_min = 1/np.max(omega)
            tau_max = 1/np.min(omega)
            k_idx = np.arange(2, M, 1) #indexes of k to compute the time constants
            self.tau[1:-1] = 10**(np.log10(tau_min) + ((k_idx-1)/(M-1))*np.log10(tau_max/tau_min)) #log distribution of the tau values
            self.tau[0] = tau_min
            self.tau[-1] = tau_max

            #build the A matrix (Ax=b) based on the existing components
            #by default: Ẑ = R_{ohm} + R_k + L + C
            #if add_capacitor=False -> Ẑ = R_{ohm} + R_k + L
            if add_capacitor:
                A_re = np.zeros(shape=(len(omega), M+3))
                A_imag = np.zeros(shape=(len(omega), M+3))
            else:
                A_re = np.zeros(shape=(len(omega), M+2))
                A_imag = np.zeros(shape=(len(omega), M+2))

            #add the components normalized by the absolute value
            A_re[:,0] = 1/np.abs(z_meas) #R_ohm
            A_imag[:,-2] = -1/(omega*np.abs(z_meas)) #inductance
            A_imag[:,-1] = omega/np.abs(z_meas) #capacitance

            #fill the contribution from each Rk component
            rk_den = 1/(1 + 1j*omega[:,np.newaxis]@self.tau[:,np.newaxis].T) #RC element
            A_re[:,1:len(self.tau)+1] = rk_den.real/np.abs(z_meas[:,np.newaxis])
            A_imag[:,1:len(self.tau)+1] = rk_den.imag/np.abs(z_meas[:,np.newaxis])

            #fit the parameters via pseudo-inverse: x = A⁻1@b
            pi_first_half = np.linalg.inv(A_re.T@A_re + A_imag.T@A_imag) #complex (A.T@A)⁻1
            pi_second_half = A_re.T@(z_meas.real/np.abs(z_meas)) + A_imag.T@(z_meas.imag/np.abs(z_meas)) #complex A.T@b
            params = pi_first_half@pi_second_half #(A.T@A)⁻1 @ (A.T@b)

            #compute the overfitting criterion
            if add_capacitor:
                mu = self.compute_mu(params[1:-2])
            else:
                mu = self.compute_mu(params[1:-1])

            #stop criteria
            if mu <= c:
                break

            if M == max_iter:
                break

        return params, M

    def generate_MRCircuit(self, freqs, params, tau):
        '''
        :param freqs: array with the swept frequencies
        :param params: the fitted parameters from the linear kramers-kronig algorithm
        :param tau: the distribution of time constants
        :return: the real and imaginary parts of the fitted circuit (computed with the help of impedance.py circuit solver)
        '''

        #Resistive parameters
        circuit_string = f"s([R({[params[0]]},{freqs.tolist()}),"
        for Rk, tk in zip(params[1:], tau):
            circuit_string += f"K({[Rk, tk]},{freqs.tolist()}),"

        #Inductance and Capacitance parameters
        circuit_string += f"L({[params[-1]]},{freqs.tolist()}),"
        if params.size == (tau.size + 3):
            circuit_string += f"C({[1 / params[-2]]},{freqs.tolist()}),"

        circuit_string = circuit_string.strip(',')
        circuit_string += '])'
        z_hat = eval(circuit_string, script_to_overwrite_local_impedancedotpy_files.circuit_elements) #compute the complex impedance

        return z_hat.real, z_hat.imag
