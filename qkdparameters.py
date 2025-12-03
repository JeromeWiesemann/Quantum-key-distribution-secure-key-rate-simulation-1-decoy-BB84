"""
QKD Parameters
**************

About
-----

This class contains all the parameters used to run the QKDSimulator. 
"""

import json

class QKDParameters:

    ### Basis choice ###
    P_X_alice = 0.9 #: Probability that Alice chooses the X basis
    P_Z_alice = 1 - P_X_alice #: Probability that Alice chooses the Z basis
    P_X_bob = 0.9 #: Probability that Bob chooses the X basis
    P_Z_bob = 1 - P_X_bob #: Probability that Bob chooses the Z basis

    ### Intensity ###
    mu_1 = 0.5 #: Decoy intensity (see Poisson distribution)
    mu_2 = 0.15 #: Signal intensity (see Poisson distribution)
    P_mu_1 = 0.9 #: Probability to send a decoy mu_1
    P_mu_2 = 1 - P_mu_1 #: Probability to send a signal mu_2

    ### State preparation ###
    R_0 = 625e6 #:  [bit/s] Transmission rate (i.e. bits prepared by Alice)
    N = 1e10 # [bit] Number of signals Alice sends

    ### Attenuation ###
    eta_bob = 1 #: Bob detector efficiency
    alpha = 0.2 #: [dB / km] losses in the fiber
    L = 100 #: [km] Fiber length
    eta_ch = pow(10, -alpha * L / 10) #: Channel attenuation

    eta_sys = eta_ch * eta_bob #: Total attenuation of the system

    ### Epsilon parameters ###
    epsilon_cor = pow(10, -12) #: Correctness parameter
    epsilon_sec = pow(10, -12) #: Secrecy parameter
    epsilon_0 = epsilon_sec / 15 #: Value to which all the terms in the security analysis are set to

    epsilon_1 = epsilon_0 #: Epsilon parameter for the Hoeffding delta of the number of photon events
    epsilon_2 = epsilon_0 #: Epsilon parameter for the Hoeffding delta of the number of photon errors

    ### Detector parameters ###
    DCR = 200 # [Hz] Dark count rate
    P_err = 0.01 # Detection error due to the light being guided to the wrong detector/timebin

    ### Mathematical model
    concentration_inequalities_method = "Hoeffding" # Method used to compute the concentration inequalities. Can be "Hoeffding" or "Azuma".

    def __init__(self, pathToParameterJSON = None):
        """
        Constructor initializes qkd_parameters with the path provided.

        Parameters
        ----------
        pathToParameterJSON : String
           Path to the JSON file containing all the parameters required to initialize ``qkd_parameters``. If :code:`None`, then the default parameters are used.
        """
        
        # Read parameters from file if a path was specified
        if pathToParameterJSON != None:
            try:
                f = open(pathToParameterJSON, "r")
            except IOError:
                input(f"Could not open file at {pathToParameterJSON}!")
            
            data = json.load(f)

            self.P_X_alice = data["P_X_alice"]
            self.P_Z_alice = 1 - self.P_X_alice
            self.P_X_bob = data["P_X_bob"]
            self.P_Z_bob = 1 - self.P_X_bob

            self.mu_1 = data["mu_1"]
            self.mu_2 = data["mu_2"]
            self.P_mu_1 = data["P_mu_1"]
            self.P_mu_2 = 1 - self.P_mu_1

            self.R_0 = data["R_0"]
            self.N = data["N"]

            self.eta_bob = data["eta_bob"]
            self.alpha = data["alpha"]
            self.L = data["L"]
            self.eta_ch = pow(10, -self.alpha * self.L / 10)

            self.eta_sys = self.eta_ch * self.eta_bob

            self.epsilon_cor = data["epsilon_cor"]
            self.epsilon_sec = data["epsilon_sec"]
            self.epsilon_0 = self.epsilon_sec / 15

            self.epsilon_1 = self.epsilon_0
            self.epsilon_2 = self.epsilon_0

            self.DCR = data["DCR"]
            self.P_err = data["P_err"]

            self.concentration_inequalities_method = data["concentration_inequalities_method"]

            f.close()

    """
    ---------- SETTER FUNCTIONS ----------
    """

    def set_channel_length(self, L):
        """Sets the channel length L and adjusts other parameters depending on L (e.g. the attenuation). 

        Parameters
        ----------
        L : float
            Channel length to set.
        """
        self.L = L
        self.eta_ch = pow(10, -self.alpha * self.L / 10)
        self.eta_sys = self.eta_ch * self.eta_bob

    def set_security_params(self, new_epsilon_cor, new_epsilon_sec):

        self.epsilon_cor = new_epsilon_cor #: Correctness parameter
        self.epsilon_sec = new_epsilon_sec #: Secrecy parameter
        self.epsilon_0 = self.epsilon_sec / 15 #: Value to which all the terms in the security analysis are set to

        self.epsilon_1 = self.epsilon_0 #: Epsilon parameter for the Hoeffding delta of the number of photon events
        self.epsilon_2 = self.epsilon_0 #: Epsilon parameter for the Hoeffding delta of the number of photon errors

    
    