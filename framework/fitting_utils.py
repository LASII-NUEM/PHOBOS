from framework import characterization_utils, data_types
import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize
import time

def Longo2020(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R||C - C || (R - R||CPE) circuit
    '''

    #expand thetas into the components with scalling
    theta = np.array(theta) * args[1] #scaling
    R1 = theta[0]
    tau1 = theta[1]
    R2 = theta[2]
    tau2 = theta[3]
    R3 = theta[4]
    tau3 = theta[5]
    n3 = theta[6]
    tau4 = theta[7]

    #impedance computation
    omega = args[0] #rad/s
    Z_b1 = R1/(1+1j*omega*tau1) #p(R1,C1) block
    Z_b2n = R2 + (R3/(1+(1j*omega*tau3)**n3)) #num of the p(C2, R2-p(R3, CPE)) block
    Z_b2d = 1 + 1j*omega*tau2+(1j*omega*tau4)/(1+(1j*omega*tau3)**n3) #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = Z_b2n/Z_b2d #p(C2, R2-p(R3, CPE)) block

    return Z_b1 + Z_b2

def Randles1947(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R - R||C circuit
    '''

    #expand thetas into the components with scalling
    theta = np.array(theta)*args[1] #scaling
    R1 = theta[0]
    R2 = theta[1]
    tau2 = theta[2]

    #impedance computation
    omega = args[0] #rad/s
    Z_b2d = 1 + 1j*omega*tau2 #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = R2/Z_b2d #p(C2, R2-p(R3, CPE)) block
    return R1 + Z_b2

#dictionary to handle function calls -> number of expected params and the function pointer
function_handlers = {
    "longo2020": {"n_params": 8, "function_ptr": Longo2020, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,np.inf), (0,1), (0,np.inf)]},
    "randles1947": {"n_params": 3, "function_ptr": Randles1947, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf)]}
}

class OptimizerResults:
    def __init__(self, opt_params=None, opt_params_scaled=None, opt_cost=None, opt_fit=None, r2_score_real=None, r2_score_imag=None, n_iter=None, t_elapsed=None):
        if opt_params is not None:
            self.opt_params = opt_params
        if opt_params_scaled is not None:
            self.opt_params_scaled = opt_params_scaled
        if opt_cost is not None:
            self.opt_fun = opt_cost
        if opt_fit is not None:
            self.opt_fit = opt_fit
        if r2_score_real is not None:
            self.r2_score_real = r2_score_real
        if r2_score_imag is not None:
            self.r2_score_imag = r2_score_imag
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
        self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
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

    def complex_MSE(self, theta, args):
        '''
        :param z_hat: the complex impedance computed from the fitted circuit
        :param z_meas: the complex impedance measured from the real system
        :return: the mean squared error between the measured and fitted impedance values
        '''

        z_hat = self.circuit_impedance(theta, [args[1], args[2]]) #compute the model for the arguments
        z_hat = z_hat.astype('complex')
        args[0] = args[0].astype('complex')
        SSE = np.sum(np.abs((args[0].real-z_hat.real)*(args[0].imag-z_hat.imag)))
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
            t_init = time.time()
            fit_obj = minimize(self.complex_MSE, initial_guess, args=([self.z_meas, omega, scaling_array],), bounds=bounds, method='L-BFGS-B')
            t_elapsed = time.time() - t_init
            opt_fit = self.circuit_impedance(fit_obj.x, [omega, scaling_array]) #compute the circuit for the optimal values
            r2_real, r2_imag = self.complex_r2_score(self.z_meas_real, opt_fit.real, self.z_meas_imag, -opt_fit.imag) #r2 score for both complex parts
            return OptimizerResults(opt_params=fit_obj.x, opt_params_scaled=fit_obj.x*scaling_array, opt_cost=fit_obj.fun,
                                    opt_fit=opt_fit, r2_score_real=r2_real, r2_score_imag=r2_imag, n_iter=fit_obj.nit, t_elapsed=t_elapsed) #return the optimized parameters
        else:
            print('...')

    def r2_score(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: R2 score of the fit (coefficient of determination)
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

        #coefficient of determination
        z_mean = np.mean(z) #mean of the observed values
        SSE = np.sum((z-z_hat)**2) #sum of the squared errors
        SSTot = np.sum((z-z_mean)**2) #sum of squares

        return 1 - (SSE/SSTot)

    def complex_r2_score(self, z_real: np.ndarray, z_hat_real: np.ndarray, z_imag: np.ndarray, z_hat_imag: np.ndarray):
        '''
        :param z_real: real part of the observed values (real measurements)
        :param z_hat_real: real part of the predicted values from the fitted circuit
        :param z_imag: real part of the observed values (real measurements)
        :param z_hat_imag: real part of the predicted values from the fitted circuit
        :return: R2 score of the fit (coefficient of determination) for both parts
        '''

        #validate 'z_real'
        if not isinstance(z_real, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_real" must be a Numpy Array! Curr. type = {type(z_real)}')

        #validate 'z_hat_real'
        if not isinstance(z_hat_real, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat_real" must be a Numpy Array! Curr. type = {type(z_hat_real)}')

        #validate 'z_imag'
        if not isinstance(z_imag, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_imag" must be a Numpy Array! Curr. type = {type(z_imag)}')

        #validate 'z_hat_imag'
        if not isinstance(z_hat_imag, np.ndarray):
            raise TypeError(
                f'[EquivalentCircuit] "z_hat_imag" must be a Numpy Array! Curr. type = {type(z_hat_imag)}')

        #validate real shape
        if len(z_real) != len(z_hat_real):
            raise ValueError(f'[EquivalentCircuit] "z_real" and "z_hat_real" must match in length!')

        #validate imaginary shape
        if len(z_imag) != len(z_hat_imag):
            raise ValueError(f'[EquivalentCircuit] "z_imag" and "z_hat_imag" must match in length!')

        #validate complex shape
        if len(z_real) != len(z_imag):
            raise ValueError(f'[EquivalentCircuit] real and imaginary parts must match in length!]')

        #compute the r2 score for both complex parts
        r2_real = self.r2_score(z_real, z_hat_real) #r2 score of the real part
        r2_imag = self.r2_score(z_imag, z_hat_imag) #r2 score of the imaginary part

        return r2_real, r2_imag




