from framework import characterization_utils, data_types
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def permittivity_by_freq_logx(data_medium: list[data_types.SpectroscopyData], data_air:data_types.SpectroscopyData, freqs: np.ndarray, eps_func=characterization_utils.dielectric_params_generic, medium = "water", artemov = False,  labels=None, title=None, yaxis_scale=1e5):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :param eps_func: function used to compute the permittivity
    :param medium: the medium to be characterized
    :param artemov : bool used to define artemov reference plot will or not be applied
    :param labels: list with the labels of the measured media
    :param title: title of the figure
    :param yaxis_scale: y-axis plot scale
    :return the computed permittivity (real and imaginary) in a dictionary structure
    '''

    #validate data_medium
    if not isinstance(data_medium, list):
        if isinstance(data_medium, data_types.SpectroscopyData):
            data_medium = [data_medium] #generate a list
        else:
            raise TypeError(f'[permittivity_by_freq] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
    else:
        for medium_elem in data_medium:
            if not isinstance(medium_elem, data_types.SpectroscopyData):
                raise TypeError(f'[permittivity_by_freq] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    if type(data_air) != data_types.SpectroscopyData:
        raise TypeError(f'[permittivity_by_freq] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs:
    if len(freqs) != len(data_medium[0].freqs):
        raise ValueError(f'[permittivity_by_freq] "freqs" and the swept frequencies from "data_medium" do not match!')

    #validate labels:
    if labels is not None:
        generate_labels = True #flag to monitor if a legend will be added or not
        if isinstance(labels, list):
            if len(labels) != len(data_medium):
                raise ValueError(f'[permittivity_by_freq] The length of "labels" and "data_medium" do not match!')
        else:
            labels = [f'{labels}']
    else:
        generate_labels = False #flag to monitor if a legend will be added or not

    #validate yaxis_scale
    if yaxis_scale < 1:
        raise ValueError(f'[permittivity_by_freq] "yaxis_scale" must be greater than 0')

        # validate medium
    medium = medium.lower()  # convert to lowercase
    valid_media = ["water", "ice"]
    if medium not in valid_media:
        raise ValueError(f'[dielectric_params_Artemov2013] "medium" = {medium} not implemented! Try: {valid_media}')
        
    #compute Artemov's reference permittivity
    eps_line_ideal, eps_2line_ideal = characterization_utils.dielectric_params_Artemov2013(freqs, medium)

    #compute all permittivity prior to plotting
    computed_eps = {} #dictionary to append all computed permittivity
    for i in range(len(data_medium)):
        curr_data_medium = data_medium[i] #extract the SpectroscopyObject
        curr_eps_real, curr_eps_imag = eps_func(curr_data_medium, data_air, curr_data_medium.freqs) #compute the dielectric parameters given "eps_func"
        computed_eps[f'data_{i}'] = {
            "real": curr_eps_real,
            "imag": curr_eps_imag
        } #append the computed data

    #plot the curves
    plt.figure()

    if generate_labels:
        leg_real = [] #list to append all real labels
        leg_imag = [] #list to append all imaginary labels
    else:
        leg_real = None
        leg_imag = None

    #real permittivity plot
    plt.subplot(2, 1, 1)

    for j in range(len(list(computed_eps.keys()))):
        plt.plot(np.log10(freqs), computed_eps[f'data_{j}']["real"]/yaxis_scale)
        if generate_labels:
            leg_real.append(labels[j])

    if artemov == True:
        plt.plot(np.log10(freqs), eps_line_ideal / yaxis_scale, color="black")
        leg_imag.append("Artemov")
    plt.ylabel(f"ε' x {yaxis_scale}")
    if generate_labels:
        plt.legend(leg_real)
    plt.grid()

    #imaginary permittivity plot
    plt.subplot(2, 1, 2)
    for j in range(len(list(computed_eps.keys()))):
        plt.plot(np.log10(freqs), computed_eps[f'data_{j}']["imag"]/yaxis_scale)
        if generate_labels:
            leg_imag.append(labels[j])
    if artemov == True:
        plt.plot(np.log10(freqs), eps_2line_ideal / yaxis_scale, color="black")
        leg_imag.append("Artemov")
    plt.ylabel(f"ε'' x {yaxis_scale}")
    if generate_labels:
        plt.legend(leg_imag)
    plt.grid()

    plt.xlabel("log(frequency)")
    if title is not None:
        plt.suptitle(f'{title}')
    plt.tight_layout()
    plt.show()

    return computed_eps

def tan_delta_logx(data_medium: list[data_types.SpectroscopyData], data_air:data_types.SpectroscopyData, freqs: np.ndarray, eps_func=characterization_utils.dielectric_params_generic, labels=None, title=None):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :param eps_func: function used to compute the permittivity
    :param labels: list with the labels of the measured media
    :param title: title of the figure
    :return the computed tan delta (eps_imag/eps_real) in a dictionary structure
    '''

    #validate data_medium
    if not isinstance(data_medium, list):
        if isinstance(data_medium, data_types.SpectroscopyData):
            data_medium = [data_medium] #generate a list
        else:
            raise TypeError(f'[tan_delta] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
    else:
        for medium_elem in data_medium:
            if not isinstance(medium_elem, data_types.SpectroscopyData):
                raise TypeError(f'[tan_delta] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    if type(data_air) != data_types.SpectroscopyData:
        raise TypeError(f'[tan_delta] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs:
    if len(freqs) != len(data_medium[0].freqs):
        raise ValueError(f'[tan_delta] "freqs" and the swept frequencies from "data_medium" do not match!')

    #validate labels:
    if labels is not None:
        generate_labels = True #flag to monitor if a legend will be added or not
        if isinstance(labels, list):
            if len(labels) != len(data_medium):
                raise ValueError(f'[tan_delta] The length of "labels" and "data_medium" do not match!')
        else:
            labels = [f'{labels}']
    else:
        generate_labels = False #flag to monitor if a legend will be added or not

    #compute all tan_delta prior to plotting
    computed_tandelta = {} #dictionary to append all computed tan_delta
    for i in range(len(data_medium)):
        curr_data_medium = data_medium[i] #extract the SpectroscopyObject
        curr_eps_real, curr_eps_imag = eps_func(curr_data_medium, data_air, curr_data_medium.freqs) #compute the dielectric parameters given "eps_func"
        curr_tan_delta = curr_eps_imag/curr_eps_real #tan_delta = eps''/eps'
        computed_tandelta[f'data_{i}'] = {
            "val": curr_tan_delta
        } #append the computed data

    #plot the curves
    plt.figure()

    if generate_labels:
        leg = [] #list to append all real labels
    else:
        leg = None

    #tan delta plot
    for j in range(len(list(computed_tandelta.keys()))):
        plt.plot(np.log10(freqs), computed_tandelta[f'data_{j}']["val"])
        if generate_labels:
            leg.append(labels[j])
    plt.xlabel("log(frequency)")
    plt.ylabel("tanδ")
    if generate_labels:
        plt.legend(leg)
    plt.grid()
    if title is not None:
        plt.title(f'{title}')
    plt.tight_layout()
    plt.show()

    return computed_tandelta

def nyquist(data_medium: list[data_types.SpectroscopyData], freqs: np.ndarray, labels=None, title=None):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param freqs: array with the swept frequencies
    :param labels: list with the labels of the measured media
    :param title: title of the figure
    :return the computed impedance (real and imaginary) in a dictionary structure
    '''

    #validate data_medium
    if not isinstance(data_medium, list):
        if isinstance(data_medium, data_types.SpectroscopyData):
            data_medium = [data_medium] #generate a list
        else:
            raise TypeError(f'[nyquist] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
    else:
        for medium_elem in data_medium:
            if not isinstance(medium_elem, data_types.SpectroscopyData):
                raise TypeError(f'[nyquist] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate freqs:
    if len(freqs) != len(data_medium[0].freqs):
        raise ValueError(f'[nyquist] "freqs" and the swept frequencies from "data_medium" do not match!')

    #validate labels:
    if labels is not None:
        generate_labels = True #flag to monitor if a legend will be added or not
        if isinstance(labels, list):
            if len(labels) != len(data_medium):
                raise ValueError(f'[nyquist] The length of "labels" and "data_medium" do not match!')
        else:
            labels = [f'{labels}']
    else:
        generate_labels = False #flag to monitor if a legend will be added or not

    #compute all permittivity prior to plotting
    computed_Z = {} #dictionary to append all computed permittivity
    for i in range(len(data_medium)):
        curr_data_medium = data_medium[i] #extract the SpectroscopyObject
        curr_z_real, curr_z_imag = characterization_utils.complex_impedance(curr_data_medium, curr_data_medium.freqs) #compute the complex impedance
        computed_Z[f'data_{i}'] = {
            "real": curr_z_real,
            "imag": curr_z_imag
        } #append the computed data

    #plot the curves
    plt.figure()

    if generate_labels:
        leg = [] #list to append all real labels
    else:
        leg = None

    #nyquist plot Z' x Z''
    for j in range(len(list(computed_Z.keys()))):
        plt.plot(computed_Z[f'data_{j}']["real"], computed_Z[f'data_{j}']["imag"])
        if generate_labels:
            leg.append(labels[j])
    plt.xlabel(f"Z'")
    plt.ylabel(f"Z''")
    if generate_labels:
        plt.legend(leg)
    plt.grid()
    if title is not None:
        plt.title(f'{title}')
    plt.tight_layout()
    plt.show()

    return computed_Z

def conductivity_by_freq_logx(data_medium: list[data_types.SpectroscopyData], data_air:data_types.SpectroscopyData, freqs: np.ndarray, eps_func=characterization_utils.dielectric_params_generic, labels=None, title=None):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :param eps_func: function used to compute the permittivity
    :param labels: list with the labels of the measured media
    :param title: title of the figure
    :return the computed conductivity (real and imaginary) in a dictionary structure
    '''

    #validate data_medium
    if not isinstance(data_medium, list):
        if isinstance(data_medium, data_types.SpectroscopyData):
            data_medium = [data_medium] #generate a list
        else:
            raise TypeError(f'[conductivity_by_freq] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
    else:
        for medium_elem in data_medium:
            if not isinstance(medium_elem, data_types.SpectroscopyData):
                raise TypeError(f'[conductivity_by_freq] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    if type(data_air) != data_types.SpectroscopyData:
        raise TypeError(f'[conductivity_by_freq] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs:
    if len(freqs) != len(data_medium[0].freqs):
        raise ValueError(f'[conductivity_by_freq] "freqs" and the swept frequencies from "data_medium" do not match!')

    #validate labels:
    if labels is not None:
        generate_labels = True #flag to monitor if a legend will be added or not
        if isinstance(labels, list):
            if len(labels) != len(data_medium):
                raise ValueError(f'[conductivity_by_freq] The length of "labels" and "data_medium" do not match!')
        else:
            labels = [f'{labels}']
    else:
        generate_labels = False #flag to monitor if a legend will be added or not

    #compute all conductivity prior to plotting
    computed_sigma = {} #dictionary to append all computed permittivity
    for i in range(len(data_medium)):
        curr_data_medium = data_medium[i] #extract the SpectroscopyObject
        curr_sigma_real, curr_sigma_imag = characterization_utils.complex_conductivity(curr_data_medium, data_air, curr_data_medium.freqs, eps_func=eps_func) #compute the conductivity given "eps_func"
        computed_sigma[f'data_{i}'] = {
            "real": curr_sigma_real,
            "imag": curr_sigma_imag
        } #append the computed data

    #plot the curves
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"] #up to 5 different scenarios, shouldn't require more for now

    for j in range(len(list(computed_sigma.keys()))):
        if generate_labels:
            ax1.plot(np.log10(freqs), computed_sigma[f'data_{j}']["real"], label=f"σ' {labels[j]}", color=colors[j])
            ax2.plot(np.log10(freqs), computed_sigma[f'data_{j}']["imag"], label=f"σ'' {labels[j]}", linestyle="dotted",
                        color=colors[j])
        else:
            ax1.plot(np.log10(freqs), computed_sigma[f'data_{j}']["real"], color=colors[j])
            ax2.plot(np.log10(freqs), computed_sigma[f'data_{j}']["imag"], linestyle="dotted", color=colors[j])

    ax1.set_ylabel("σ'")
    ax1.set_xlabel("log(frequency)")
    ax1.tick_params(axis='y')
    ax2.set_ylabel("σ''")
    ax2.tick_params(axis='y')
    if generate_labels:
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax1.grid()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return computed_sigma

def cole_cole_permittivity(data_medium: list[data_types.SpectroscopyData], data_air:data_types.SpectroscopyData, freqs: np.ndarray, eps_func=characterization_utils.dielectric_params_generic, labels=None, title=None):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :param eps_func: function used to compute the permittivity
    :param labels: list with the labels of the measured media
    :param title: title of the figure
    :return the computed permittivity (real and imaginary) in a dictionary structure
    '''

    #validate data_medium
    if not isinstance(data_medium, list):
        if isinstance(data_medium, data_types.SpectroscopyData):
            data_medium = [data_medium] #generate a list
        else:
            raise TypeError(f'[cole_cole_permittivity] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
    else:
        for medium_elem in data_medium:
            if not isinstance(medium_elem, data_types.SpectroscopyData):
                raise TypeError(f'[cole_cole_permittivity] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    if type(data_air) != data_types.SpectroscopyData:
        raise TypeError(f'[cole_cole_permittivity] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs:
    if len(freqs) != len(data_medium[0].freqs):
        raise ValueError(f'[cole_cole_permittivity] "freqs" and the swept frequencies from "data_medium" do not match!')

    #validate labels:
    if labels is not None:
        generate_labels = True #flag to monitor if a legend will be added or not
        if isinstance(labels, list):
            if len(labels) != len(data_medium):
                raise ValueError(f'[cole_cole_permittivity] The length of "labels" and "data_medium" do not match!')
        else:
            labels = [f'{labels}']
    else:
        generate_labels = False #flag to monitor if a legend will be added or not

    #compute all conductivity prior to plotting
    computed_eps = {} #dictionary to append all computed permittivity
    for i in range(len(data_medium)):
        curr_data_medium = data_medium[i] #extract the SpectroscopyObject
        curr_eps_real, curr_eps_imag = eps_func(curr_data_medium, data_air, curr_data_medium.freqs) #compute the conductivity given "eps_func"
        computed_eps[f'data_{i}'] = {
            "real": curr_eps_real,
            "imag": curr_eps_imag
        } #append the computed data

    #cole-cole conductivity plot ε' x ε''
    plt.figure()

    if generate_labels:
        leg = [] #list to append all real labels
    else:
        leg = None

    for j in range(len(list(computed_eps.keys()))):
        plt.plot(computed_eps[f'data_{j}']["real"], computed_eps[f'data_{j}']["imag"])
        if generate_labels:
            leg.append(labels[j])
    plt.xlabel(f"ε'")
    plt.ylabel(f"ε''")
    if generate_labels:
        plt.legend(leg)
    plt.grid()
    if title is not None:
        plt.title(f'{title}')
    plt.tight_layout()
    plt.show()

    return computed_eps

def cole_cole_conductivity(data_medium: list[data_types.SpectroscopyData], data_air:data_types.SpectroscopyData, freqs: np.ndarray, eps_func=characterization_utils.dielectric_params_generic, labels=None, title=None):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :param eps_func: function used to compute the permittivity
    :param labels: list with the labels of the measured media
    :param title: title of the figure
    :return the computed conductivity (real and imaginary) in a dictionary structure
    '''

    #validate data_medium
    if not isinstance(data_medium, list):
        if isinstance(data_medium, data_types.SpectroscopyData):
            data_medium = [data_medium] #generate a list
        else:
            raise TypeError(f'[cole_cole_conductivity] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
    else:
        for medium_elem in data_medium:
            if not isinstance(medium_elem, data_types.SpectroscopyData):
                raise TypeError(f'[cole_cole_conductivity] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    if type(data_air) != data_types.SpectroscopyData:
        raise TypeError(f'[cole_cole_conductivity] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs:
    if len(freqs) != len(data_medium[0].freqs):
        raise ValueError(f'[cole_cole_conductivity] "freqs" and the swept frequencies from "data_medium" do not match!')

    #validate labels:
    if labels is not None:
        generate_labels = True #flag to monitor if a legend will be added or not
        if isinstance(labels, list):
            if len(labels) != len(data_medium):
                raise ValueError(f'[cole_cole_conductivity] The length of "labels" and "data_medium" do not match!')
        else:
            labels = [f'{labels}']
    else:
        generate_labels = False #flag to monitor if a legend will be added or not

    #compute all conductivity prior to plotting
    computed_sigma = {} #dictionary to append all computed permittivity
    for i in range(len(data_medium)):
        curr_data_medium = data_medium[i] #extract the SpectroscopyObject
        curr_sigma_real, curr_sigma_imag = characterization_utils.complex_conductivity(curr_data_medium, data_air, curr_data_medium.freqs, eps_func=eps_func) #compute the conductivity given "eps_func"
        computed_sigma[f'data_{i}'] = {
            "real": curr_sigma_real,
            "imag": curr_sigma_imag
        } #append the computed data

    #cole-cole conductivity plot σ' x σ''
    plt.figure()

    if generate_labels:
        leg = [] #list to append all real labels
    else:
        leg = None

    for j in range(len(list(computed_sigma.keys()))):
        plt.plot(computed_sigma[f'data_{j}']["real"], computed_sigma[f'data_{j}']["imag"])
        if generate_labels:
            leg.append(labels[j])
    plt.xlabel(f"σ'")
    plt.ylabel(f"σ''")
    if generate_labels:
        plt.legend(leg)
    plt.grid()
    if title is not None:
        plt.title(f'{title}')
    plt.tight_layout()
    plt.show()

    return computed_sigma

def cole_cole_modulus(data_medium: list[data_types.SpectroscopyData], data_air:data_types.SpectroscopyData, freqs: np.ndarray, eps_func=characterization_utils.dielectric_params_generic, labels=None, title=None):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :param eps_func: function used to compute the permittivity
    :param labels: list with the labels of the measured media
    :param title: title of the figure
    :return the computed dielectric modulus (real and imaginary) in a dictionary structure
    '''

    #validate data_medium
    if not isinstance(data_medium, list):
        if isinstance(data_medium, data_types.SpectroscopyData):
            data_medium = [data_medium] #generate a list
        else:
            raise TypeError(f'[cole_cole_modulus] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')
    else:
        for medium_elem in data_medium:
            if not isinstance(medium_elem, data_types.SpectroscopyData):
                raise TypeError(f'[cole_cole_modulus] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    if type(data_air) != data_types.SpectroscopyData:
        raise TypeError(f'[cole_cole_modulus] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs:
    if len(freqs) != len(data_medium[0].freqs):
        raise ValueError(f'[cole_cole_modulus] "freqs" and the swept frequencies from "data_medium" do not match!')

    #validate labels:
    if labels is not None:
        generate_labels = True #flag to monitor if a legend will be added or not
        if isinstance(labels, list):
            if len(labels) != len(data_medium):
                raise ValueError(f'[cole_cole_modulus] The length of "labels" and "data_medium" do not match!')
        else:
            labels = [f'{labels}']
    else:
        generate_labels = False #flag to monitor if a legend will be added or not

    #compute all conductivity prior to plotting
    computed_M = {} #dictionary to append all computed permittivity
    for i in range(len(data_medium)):
        curr_data_medium = data_medium[i] #extract the SpectroscopyObject
        curr_M_real, curr_M_imag = characterization_utils.dielectric_modulus(curr_data_medium, data_air, curr_data_medium.freqs, eps_func=eps_func) #compute the conductivity given "eps_func"
        computed_M[f'data_{i}'] = {
            "real": curr_M_real,
            "imag": curr_M_imag
        } #append the computed data

    #cole-cole conductivity plot M' x M''
    plt.figure()

    if generate_labels:
        leg = [] #list to append all real labels
    else:
        leg = None

    for j in range(len(list(computed_M.keys()))):
        plt.plot(computed_M[f'data_{j}']["real"], computed_M[f'data_{j}']["imag"])
        if generate_labels:
            leg.append(labels[j])
    plt.xlabel(f"M'")
    plt.ylabel(f"M''")
    if generate_labels:
        plt.legend(leg)
    plt.grid()
    if title is not None:
        plt.title(f'{title}')
    plt.tight_layout()
    plt.show()

    return computed_M