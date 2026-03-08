import numpy as np
from framework import fitting_utils, data_types
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class MonteCarloFit:
    def __init__(self, topology:str, n_spectra:int, noise_var:float, theta:np.ndarray, freqs:np.ndarray):
        '''
        :param topology: the circuit topoly that outputs the modeled impedance of the expected circuit
        :param n_spectra: number of simulated impedance spectra
        :param noise_var: noise variance of the gaussian noise
        :param theta: true parameters used to compute the impedance
        :param freqs: array with the swept frequencies
        '''

        #validate 'topology'
        topology = topology.lower() #convert to lower case
        valid_topologies = list(fitting_utils.function_handlers.keys()) #expected parameters for the available models
        if topology not in valid_topologies:
            raise ValueError(f'[MonteCarloFit] {topology} not implemented! Try: {valid_topologies}')
        self.topology = topology
        self.theta = theta
        self.circuit_impedance = fitting_utils.function_handlers[topology.lower()]["function_ptr"]
        self.fit_method = None

        #validate 'n_spectra'
        if n_spectra<=0:
            raise ValueError('[MonteCarloFit] n_spectra must be greater than 0!')
        self.n_spectra = int(n_spectra)
        self.noise_var = noise_var

        #validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[MonteCarloFit] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs

        #simulate the circuit with the true parameters
        omega = 2*np.pi*freqs #Hz to rad/s
        self.Z_simu = self.circuit_impedance(self.theta, args=[omega, np.ones_like(self.theta)]) #compute the true impedance based on theta
        self.Z_simu = self.Z_simu.astype("complex")

        #generate N impedance spectra with gaussian noise for the Monte Carlo Simulation

        #Note: the Monte Carlo Simulations follow the proposed in:
        # Francesco Santoni, Alessio De Angelis, Antonio Moschitta, Paolo Carbone, Matteo Galeotti, Lucio Cinà, Corrado Giammanco, Aldo Di Carlo,
        # A guide to equivalent circuit fitting for impedance analysis and battery state estimation,
        # Journal of Energy Storage. 2024; 82:110389. https://doi.org/10.1016/j.est.2023.110389

        #zero-mean uncorrelated circularly-symmetric Gaussian noise
        noise_real_part = np.random.normal(0, self.noise_var/1, size=(self.n_spectra, len(self.Z_simu)))
        noise_imag_part = np.random.normal(0, self.noise_var/1, size=(self.n_spectra, len(self.Z_simu)))
        self.Z_noise = (self.Z_simu.real+noise_real_part)-1j*(self.Z_simu.imag+noise_imag_part) #Gaussian-noise-corrupted samples
        self.Z_noise = self.Z_noise.astype("complex")
        #self.CramerRao = CramerRaoBound(self.circuit_impedance, self.theta, )

    def fit_simulations(self, initial_guess:np.ndarray, scaling_array:np.ndarray, method:str, verbose=False, plot=False):
        '''
        :param initial_guess: the initial guess for the fit to run the iterative algorithms
        :param scaling_array: scale all the search parameters to avoid exploding gradients
        :param method: which optimization algorithm will be used to fit the circuit data
        :param verbose: flag to print the statistics in the terminal
        :param plot: flag to plot the simulated fittings over the simulated data
        :return the fitted impedance and the optimal parameters for each simulated spectrum
        '''

        #validate "method"
        valid_methods = ['BFGS', 'NLLS', 'DLS', 'Nelder-Mead', 'PSO']
        if method not in valid_methods:
            raise ValueError(f'[MonteCarloFit] {method} not implemented! Try: {valid_methods}')
        self.fit_method = method #update fit method

        #fit each corrupted spectra separately
        theta_fit = np.zeros(shape=(self.n_spectra, len(initial_guess))) #matrix to store the optimal values computed per iteration
        Z_fit = np.zeros(shape=np.shape(self.Z_noise), dtype="complex") #matrix to store the fittings per iteration
        for i in range(0, self.n_spectra):
            if verbose:
                print(f'[MonteCarloFit] Spectra {i+1}/{self.n_spectra}')

            curr_obj = fitting_utils.EquivalentCircuit(self.topology,[self.Z_noise.real[i,:], self.Z_noise.imag[i,:]], self.freqs) #equivalent circuit object
            fit_params = curr_obj.fit_circuit(initial_guess, scaling_array, method=method, verbose=verbose) #optimal parameter search
            Z_fit[i,:] = fit_params.opt_fit.astype("complex") #store the fit of the current iteration
            theta_fit[i,:] = fit_params.opt_params_scaled #store the optimal parameters of the current iteration

        if plot:
            plt.figure()
            leg = []
            for i in range(0, self.n_spectra):
                plt.plot(Z_fit.real[i, :], -Z_fit.imag[i, :], color='red', linewidth=1, zorder=2)
                plt.scatter(self.Z_noise.real[i, :], self.Z_noise.imag[i, :], c='black', s=8, zorder=1)
                if i == self.n_spectra-1:
                    leg.append('Fit')
                    leg.append('Simulated Spectra')
            plt.xlabel("Z'")
            plt.ylabel("Z''")
            plt.legend(leg)
            plt.grid()
            plt.show()

        return Z_fit.astype("complex"), theta_fit

    def compute_statistics(self, Z_fit:np.ndarray, theta_fit:np.ndarray, verbose=False):
        '''
        :param Z_fit: the fitted impedance for each simulated spectrum
        :param theta_fit: the optimal parameters for each simulated spectrum
        :param verbose: flag to print the statistics in the terminal
        :return:
        '''

        #compute the average and standard error for each parameter
        theta_mean = np.mean(theta_fit, axis=0) #average values for the impedance spectra
        theta_sse = np.sum((self.theta-theta_fit)**2, axis=0) #sum of squared errors
        theta_var = (1/(self.n_spectra*(self.n_spectra-1)))*theta_sse #variance
        theta_std = np.sqrt(theta_var) #standard deviation

        #compute the mean absolute percentage error
        Z_fit = Z_fit.astype("complex")
        p_error_real = np.abs(self.Z_noise.real-Z_fit.real)/np.abs(self.Z_noise.real)
        mape_real = 100*np.mean(np.mean(p_error_real, axis=1)) #mean of the mean percentage error between fits
        p_error_imag= np.abs(self.Z_noise.imag-np.abs(Z_fit.imag))/np.abs(self.Z_noise.imag)
        mape_imag = 100*np.mean(np.mean(p_error_imag, axis=1)) #mean of the mean percentage error between fits

        if verbose:
            params = fitting_utils.function_handlers[self.topology.lower()]["fit_params"]
            for i in range(0, len(self.theta)):
                print(f'[{params[i]}] estimated: {theta_mean[i]}±{theta_std[i]} / true: {self.theta[i]}')

        print(f'[MAPE real] {mape_real}')
        print(f'[MAPE imag] {mape_imag}')

        return theta_mean, mape_imag

def compute_Jacobian(z_fun, theta:np.ndarray, y_hat:np.ndarray, delta=1e-3, args=()):
    '''
    :param z_fun: the function to compute the impedance of the target circuit of the fitting
    :param theta: current guess for the levenberg-marquardt algorithm
    :param y_hat: predicted samples of the current iteration
    :param delta: numerical derivative step
    :param args: list with parameters that won't be minimized but are required to compute the impedance
    :return: the Jacobian matrix computed by the central finite difference
    '''

    m = len(y_hat) #number of samples to be fitted
    n = len(theta) #number of candidate values
    J = np.zeros(shape=(m,n)) #Jacobian matrix
    delta_params = delta*np.ones(len(theta)) #vector of the derivative step

    for i in range(n):
        placeholder_theta = theta[i] #store the current theta before applying central difference
        curr_step = delta_params[i]*(1+np.abs(placeholder_theta)) #step for the central difference
        if curr_step!=0:
            theta[i] = placeholder_theta+curr_step #x[i+h]
            forward_y = z_fun(theta,args) #evaluate the cost at the i+h sample
            forward_y = forward_y.astype("complex").real
            theta[i] = placeholder_theta-curr_step #x[i-h]
            backwards_y = z_fun(theta, args) #evaluate the cost at the i-h sample
            backwards_y = backwards_y.astype("complex").real
            J[:,i] = (forward_y-backwards_y)/(2*curr_step) #(x[i+h]-x[i-h])/2h

        theta[i] = placeholder_theta #rebuild the parameters array

    return J

def compute_NoiseVar(y:np.ndarray, y_hat:np.ndarray, theta:np.ndarray):
    '''
    :param theta: circuit parameters
    :param y: measured samples
    :param y_hat: predicted samples
    :return: the estimated noise variance
    '''

    SSE = np.sum((y.real-y_hat.real)**2 + (y.imag-y_hat.imag)**2) #sum of the squared errors
    return SSE/(2*len(y)-len(theta))

def CramerRaoBound(z_fun, theta:np.ndarray, y:np.ndarray, y_hat:np.ndarray, delta=1e-3, args=()):
    '''
    :param z_fun: the function to compute the impedance of the target circuit of the fitting
    :param theta: circuit parameters
    :param y: measured samples
    :param y_hat: predicted samples
    :param delta: numerical derivative step
    :param args: list with parameters that won't be minimized but are required to compute the impedance
    :return: the lower bound acceptable for fitting variance
    '''

    J = compute_Jacobian(z_fun, theta, y_hat, delta=delta, args=args) #compute the Jacobian
    H = J.T@J #Hessian matrix
    H_inv = np.linalg.inv(H) #inverse of the Fisher information matrix
    noise_std = np.sqrt(compute_NoiseVar(y, y_hat, theta)) #compute the noise variance

    return noise_std*np.sqrt(np.diag(H_inv))
