"""
QKD Simulator
*************

About
=====

This class simulates a quantum key distribution (QKD) system using the decoy state method.
"""

from qkdparameters import QKDParameters
import numpy as np
from math import factorial


def clip(a, b, x):
    """
    Clip the value `x` when it is outside of interval `[a, b]`.
    """
    assert a < b
    x_new = max(a, min(b, x))
    return x_new


class QKDSimulator:

    qkd_parameters = QKDParameters() #: Class containing all the parameters used to run the QKDSimulator

    with_vacuum = False # Add vacuum events to final secure key length
    using_improved_serfling = False # If False, using original Serfling inequality, otherwise, using the improved version by Fung

    def __init__(self, pathToParameterJSON = None):
        """
        Constructor initializes qkd_parameters with the path provided.

        Parameters
        ----------
        pathToParameterJSON : String
           Path to the JSON file containing all the parameters required to initialize ``qkd_parameters``. If :code:`None`, then the default parameters are used.
        """
        
        self.qkd_parameters = QKDParameters(pathToParameterJSON)


    def calculate_skr(self):
        """
        This function computes the secret key length for the 1-decoy state protocol and for the parameters set in ``qkd_parameters``. 

        .. math::

            l_\\text{max} = s_{X,0}^- + s_{X,1}^-(1 - h(\\Lambda_X^+)) - \\text{leak}_{EC} - \\log_2{\\frac{2}{\\epsilon_\\text{cor}}} - 6\\log_2{\\frac{19}{\\epsilon_\\text{sec}}}

        Returns
        -------
        float
            Secret key rate
        """

        """
        ---------- ASSIGNEMENT TO LOCAL VARIABLES ----------
        TODO: this can probably be better optimizes. Copying is very inefficient but writing self for all variables is not convenient.
        """

        ### Basis choice ###
        P_X_alice = self.qkd_parameters.P_X_alice # Probability that Alice chooses the X basis
        P_Z_alice = self.qkd_parameters.P_Z_alice # Probability that Alice chooses the Z basis
        P_X_bob = self.qkd_parameters.P_X_bob # Probability that Bob chooses the X basis
        P_Z_bob = self.qkd_parameters.P_Z_bob # Probability that Bob chooses the Z basis

        ### Intensity ###
        mu_1 = self.qkd_parameters.mu_1 # Mu_1 intensity (see Poisson distribution)
        mu_2 = self.qkd_parameters.mu_2 # Mu_2 intensity (see Poisson distribution)
        P_mu_1 = self.qkd_parameters.P_mu_1 # Probability to send a decoy mu_1
        P_mu_2 = self.qkd_parameters.P_mu_2 # Probability to send a decoy mu_2

        ### State preparation ###
        R_0 = self.qkd_parameters.R_0 #  [bit/s] Transmission rate (i.e. bits prepared by Alice)
        N = self.qkd_parameters.N # [bit] Number of signals Alice sends
        asymptotic = self.qkd_parameters.asymptotic # If true, then asymptotic key rate is computed (N is then ignored)

        ### Attenuation ###
        eta_ch = self.qkd_parameters.eta_ch # Channel attenuation
        eta_sys = self.qkd_parameters.eta_sys # Total attenuation of the system

        ### Epsilon parameters ###
        epsilon_cor = self.qkd_parameters.epsilon_cor # Correctness parameter
        epsilon_sec = self.qkd_parameters.epsilon_sec # Secrecy parameter
        epsilon_0 = epsilon_sec / 15

        epsilon_1 = epsilon_0 # Epsilon parameter for the Hoeffding delta of the number of photon events
        epsilon_2 = epsilon_0 # Epsilon parameter for the Hoeffding delta of the number of photon events

        ### Detectors ###
        DCR = self.qkd_parameters.DCR # [Hz] Dark count rate
        P_err = self.qkd_parameters.P_err # Detection error due to the light being guided to the wrong detector/timebin

        """
        ---------- PRIOR CHECKS ----------
        """
        debug = False

        if debug:

            # Some parts of the proof assume that mu_1 > mu_2
            if mu_1 <= mu_2:
                raise Exception("mu_1 should be strictly greater than mu_2!")

            if eta_ch <= 0:
                raise Exception("Negative or zero total attenuation!")


        """
        ---------- MEASURED DETECTIONS AND ERRORS ----------
        - Determine the a priori detections and errors without considering the dead time of the detectors
        - We then "correct" these a priori detections by considering the dead time of the detectors (see https://en.wikipedia.org/wiki/Dead_time or https://arxiv.org/pdf/2507.10361 ;))

        All detection rates are given in [bit/s]
        """

        if mu_1 <= mu_2: # This is to avoid constraints when optimizing the parameters (using scipy.minimize() for example) as the security proof assumes that mu_1 > mu_2
            return 0

        P_ZZ = P_Z_alice * P_Z_bob
        P_XX = P_X_alice * P_X_bob

        P_DC = DCR / R_0 # Dark count probability per qubit

        P_det_mu_1_aprio = (1 - np.exp(-mu_1 * eta_sys) * (1 - P_DC)) # A priori propability of detecting a click given the intensity mu_1
        P_det_mu_2_aprio = (1 - np.exp(-mu_2 * eta_sys) * (1 - P_DC)) # A priori propability of detecting a click given the intensity mu_2

        ### X Basis ###
        # Detection rates
        R_X_mu_1 = R_0 * P_XX * P_mu_1 * P_det_mu_1_aprio # Detection rate in X basis with intensity mu_1
        R_X_mu_2 = R_0 * P_XX * P_mu_2 * P_det_mu_2_aprio # Detection rate in X basis with intensity mu_2
        R_X_tot = R_X_mu_1 + R_X_mu_2

        # Nb. of detections
        integration_time = N / R_X_tot # Time required for Bob to receive N signals
        n_X_mu_1 = integration_time * R_X_mu_1 # Nb. of detections in the X basis with intensity mu_1
        n_X_mu_2 = integration_time * R_X_mu_2 # Nb. of detections in the X basis with intensity mu_2
        n_X = n_X_mu_1 + n_X_mu_2

        # Errors
        # P_err_mu_1 = (1 - np.exp(-mu_1 * eta_sys))* P_err + P_DC * np.exp(-mu_1 * eta_sys) / 2 # Probability of an error occuring given a detection with intensity mu_1
        P_err_mu_1 = (1 - np.exp(-mu_1 * eta_sys))* P_err + P_DC / 2 # Probability of an error occuring given a detection with intensity mu_1
        c_X_mu_1 = integration_time * R_0 * P_XX * P_mu_1 * P_err_mu_1 # Number of errors for detections with intensity mu_1

        # P_err_mu_2 = (1 - np.exp(-mu_2 * eta_sys))* P_err + P_DC * np.exp(-mu_2 * eta_sys) / 2 # Probability of an error occuring given a detection with intensity mu_2
        P_err_mu_2 = (1 - np.exp(-mu_2 * eta_sys))* P_err + P_DC / 2 # Probability of an error occuring given a detection with intensity mu_2
        c_X_mu_2 = integration_time * R_0 * P_XX * P_mu_2 * P_err_mu_2 # Number of errors for detections with intensity mu_2

        c_X = c_X_mu_1 + c_X_mu_2 # Total number of errors in the X basis (excluding discarded bits)

        ### Z Basis ###
        # Detection rates (without considering detector dead times)
        R_Z_mu_1 = R_0 * P_ZZ * P_mu_1 * P_det_mu_1_aprio # Detection rate in Z basis with intensity mu_1
        R_Z_mu_2 = R_0 * P_ZZ * P_mu_2 * P_det_mu_2_aprio # Detection rate in Z basis with intensity mu_2

        # Nb. of detections
        n_Z_mu_1 = integration_time * R_Z_mu_1 # Nb. of detections in the Z basis with intensity mu_1
        n_Z_mu_2 = integration_time * R_Z_mu_2 # Nb. of detections in the Z basis with intensity mu_2

        n_Z = n_Z_mu_1 + n_Z_mu_2
        
        # Errors
        c_Z_mu_1 = integration_time * R_0 * P_ZZ * P_mu_1 * P_err_mu_1 # Number of errors for detections with intensity mu_1
        c_Z_mu_2 = integration_time * R_0 * P_ZZ * P_mu_2 * P_err_mu_2 # Number of errors for detections with intensity mu_2

        c_Z = c_Z_mu_1 + c_Z_mu_2 # Total nb. of errors in the Z basis

        """
        ---------- BOUNDS ON NUMBER OF DETECTIONS AND ERRORS ----------
        Section 4.1: Hoeffding inequalities
        """

        ### X Basis ###
        # Detections
        n_X_mu_1_plus = clip(0, n_X, n_X_mu_1 + self.concentration_ineq_deviation(n_X, epsilon_1))
        
        n_X_mu_2_minus = clip(0, n_X, n_X_mu_2 - self.concentration_ineq_deviation(n_X, epsilon_1))

        # Errors
        c_X_mu_1_plus = clip(0, c_X, c_X_mu_1 + self.concentration_ineq_deviation(c_X, epsilon_2))

        c_X_mu_2_plus = clip(0, c_X, c_X_mu_2 + self.concentration_ineq_deviation(c_X, epsilon_2))
        c_X_mu_2_minus = clip(0, c_X, c_X_mu_2 - self.concentration_ineq_deviation(c_X, epsilon_2))

        ### Z Basis ###
        # Detections
        n_Z_mu_1_plus = clip(0, n_Z, n_Z_mu_1 + self.concentration_ineq_deviation(n_Z, epsilon_1))
        
        n_Z_mu_2_minus = clip(0, n_Z, n_Z_mu_2 - self.concentration_ineq_deviation(n_Z, epsilon_1))

        # Errors
        c_Z_mu_1_plus = clip(0, c_Z, c_Z_mu_1 + self.concentration_ineq_deviation(c_Z, epsilon_2))

        c_Z_mu_2_plus = clip(0, c_Z, c_Z_mu_2 + self.concentration_ineq_deviation(c_Z, epsilon_2))

        c_Z_mu_2_minus = clip(0, c_Z, c_Z_mu_2 - self.concentration_ineq_deviation(c_Z, epsilon_2))

        """
        ---------- BOUNDS ON PHOTON EVENTS AND ERRORS ----------
        Sections 4.1.x and 4.2 
        """

        ### Lower bound on the nb. of vacuum events ###
        s_X_0_minus = (self.tau(0) / (mu_1 - mu_2)) * (mu_1 * np.exp(mu_2) * n_X_mu_2_minus / P_mu_2 - mu_2 * np.exp(mu_1) * n_X_mu_1_plus / P_mu_1) # X basis
        s_X_0_minus = clip(0, n_X, s_X_0_minus)
        
        ### Upper bound on the nb. of vaccum events ###
        # REVIEW: Here we can plug in the intensity mu that gives the highest secret key rate
        s_X_0_plus_mu_1 = 2 * (self.tau(0) * (np.exp(mu_1) / P_mu_1) * c_X_mu_1_plus + self.concentration_ineq_deviation(n_X, epsilon_1)) 
        s_X_0_plus_mu_2 = 2 * (self.tau(0) * (np.exp(mu_2) / P_mu_2) * c_X_mu_2_plus + self.concentration_ineq_deviation(n_X, epsilon_1))
        s_X_0_plus = min(s_X_0_plus_mu_1, s_X_0_plus_mu_2) # X basis
        s_X_0_plus = clip(0, n_X, s_X_0_plus)

        s_Z_0_plus_mu_1 = 2 * (self.tau(0) * (np.exp(mu_1) / P_mu_1) * c_Z_mu_1_plus + self.concentration_ineq_deviation(n_Z, epsilon_1))
        s_Z_0_plus_mu_2 = 2 * (self.tau(0) * (np.exp(mu_2) / P_mu_2) * c_Z_mu_2_plus + self.concentration_ineq_deviation(n_Z, epsilon_1))
        s_Z_0_plus = min(s_Z_0_plus_mu_1, s_Z_0_plus_mu_2) # Z basis
        s_Z_0_plus = clip(0, n_Z, s_Z_0_plus)

        ### Lower bound on the nb. of single photon events ###
        s_X_1_minus = (mu_1 * self.tau(1) / (mu_2 * (mu_1 - mu_2))) * (np.exp(mu_2) * n_X_mu_2_minus / P_mu_2 - (pow(mu_2, 2) / pow(mu_1, 2)) * (np.exp(mu_1) * n_X_mu_1_plus) / P_mu_1 - (pow(mu_1, 2) - pow(mu_2, 2)) / (pow(mu_1, 2) * self.tau(0)) * s_X_0_plus) # X basis
        s_X_1_minus = clip(0, n_X, s_X_1_minus)

        s_Z_1_minus = (mu_1 * self.tau(1) / (mu_2 * (mu_1 - mu_2))) * (np.exp(mu_2) * n_Z_mu_2_minus / P_mu_2 - (pow(mu_2, 2) / pow(mu_1, 2)) * (np.exp(mu_1) * n_Z_mu_1_plus) / P_mu_1 - (pow(mu_1, 2) - pow(mu_2, 2)) / (pow(mu_1, 2) * self.tau(0)) * s_Z_0_plus) # Z basis
        s_Z_1_minus = clip(0, n_Z, s_Z_1_minus)

        ### Upper bound on the nb. of single photon errors ###
        v_X_1_plus = (self.tau(1) / (mu_1 - mu_2)) * (np.exp(mu_1) * c_X_mu_1_plus / P_mu_1 - np.exp(mu_2) * c_X_mu_2_minus / P_mu_2) # X basis
        v_X_1_plus = clip(0, n_X, v_X_1_plus) # Not needed

        v_Z_1_plus = (self.tau(1) / (mu_1 - mu_2)) * (np.exp(mu_1) * c_Z_mu_1_plus / P_mu_1 - np.exp(mu_2) * c_Z_mu_2_minus / P_mu_2) # Z basis
        v_Z_1_plus = clip(0, n_Z, v_Z_1_plus)

        ### Upper bound on the QBER ###
        if s_Z_1_minus > 0:
            lambda_Z_plus = v_Z_1_plus / s_Z_1_minus # X basis, used to approximate the phase error for the Z basis
        else:
            lambda_Z_plus = 0.5
        
        if lambda_Z_plus > 0.5:
            lambda_Z_plus = 0.5

        if s_X_1_minus != 0 and s_Z_1_minus != 0:
            lambda_X_plus = lambda_Z_plus + self.gamma(epsilon_0, lambda_Z_plus, s_X_1_minus, s_Z_1_minus) # X basis
        else:
            lambda_X_plus = 0.5
        if lambda_X_plus > 0.5:
            lambda_X_plus = 0.5

        """
        ---------- INFORMATION LEAKED DURING ERROR CORRECTION ----------
        Model taken from "Charles Ci Wen Lim , Concise security bounds for practical decoy-state quantum key distribution, 2014"
        """

        f_EC = 1.16 # Error-correction efficiency
        leak_EC =  n_X * f_EC * self.binary_entropy(c_X / n_X) # [bits] Number of bits leaks during error correction

        """
        ---------- MAXIMUM EXTRACTABLE SECRET KEY LENGTH ----------
        Section 6.2
        """
        if not self.with_vacuum:
            s_X_0_minus = 0
        
        if not asymptotic:
            l_max = s_X_0_minus + s_X_1_minus * (1 - self.binary_entropy(lambda_X_plus)) - leak_EC - np.log2(2 / epsilon_cor) \
                    - 4 * np.log2(15 / (epsilon_sec * pow(2, 1/4)))
        else:
            l_max = s_X_0_minus + s_X_1_minus * (1 - self.binary_entropy(lambda_X_plus)) - leak_EC

        if l_max < 0:
            l_max = 0

        l_max_rate = l_max / integration_time # Note: in the asymptotic case, the division with integration_time cancels with the multiplication with integration_time for the photon number statistics, therefore the block size effectively does not matter in this case, as expected.

        return l_max_rate


    """
    ---------- HELPER FUNCTIONS ----------
    """

    def eta_to_db(self, eta):
        """Converts an attenuation :math:`\\eta` in percent to an attenuation in decibels.

        .. math::

            \\text{dB} = -10 * \\log_{10}(\\eta)

        Parameters
        ----------
        eta : float
            Atttenuation in percent.

        Returns
        -------
        float
        """

        db = -10 * np.log10(eta)
        return db

    def concentration_ineq_deviation(self, n, epsilon):
        """Deviation due to the computation of concentration inequalities. This function depends on the method set for ``qkd_parameters.concentration_inequalities_method``.

        .. math::

            \\delta(n,\\epsilon)=\\sqrt{n\\log(1/\\epsilon)/2}

        Parameters
        ----------
        n : int
            Number of events.
        epsilon: float
            Probability of the bounds failing is :math:`\\epsilon` or :math:`2\\epsilon` depending on the case. 

        Returns
        -------
        float
        """
        if self.qkd_parameters.asymptotic:
            return 0
        if(self.qkd_parameters.concentration_inequalities_method == "Hoeffding"):
            return np.sqrt(n * np.log(1 / epsilon) / 2)
        elif(self.qkd_parameters.concentration_inequalities_method == "Azuma"):
            return np.sqrt(2 * n * np.log(1 / epsilon))
        else:
            print("[ERROR]: Incorrect value for concentration inequality method!")

    def tau(self, m):
        """Function :math:`\\tau_m` from Bayes'r rule.

        .. math::

            \\tau_m = \\sum_{k\\in \\{\\mu_1, \\mu_2\\}} p_k \\frac{e^{-k} k^m}{m!}

        Parameters
        ----------
        m : int
            Number of photons.

        Returns
        -------
        float
        """

        return self.qkd_parameters.P_mu_1 * np.exp(-self.qkd_parameters.mu_1) * pow(self.qkd_parameters.mu_1, m) / factorial(m) + self.qkd_parameters.P_mu_2 * np.exp(-self.qkd_parameters.mu_2) * pow(self.qkd_parameters.mu_2, m) / factorial(m)

    def gamma(self, a, b, c, d):
        """Function :math:`\\gamma` for the estimation of the QBER.

        .. math::

            \\gamma(a, b, c, d) = \\sqrt{\\frac{(c+d)(1-b)b}{cd\\ln 2} \\log_2\\left(\\frac{c+d}{cd(1-b)b} \\frac{1}{a^2}\\right)}
            

    
        Parameters
        ----------
        m : int
            Number of photons.

        Returns
        -------
        float
        """
        if self.qkd_parameters.asymptotic:
            return 0
        if self.using_improved_serfling:
            return np.sqrt((((c + d)*(1 - b)*b) / (c*d*np.log(2))) * np.log2((c + d) / (c*d*(1 - b)*b*(a**2))))
        else: 
            return np.sqrt(((d + c) / (d * c)) * ((c + 1) / (c)) * np.log(1 / a))

    def binary_entropy(self, x):
        """Binary entropy function used to compute the maximum secret key rate.

        .. math::

            h(x) = -x\\log_2(x) - (1-x)\\log_2(1-x)

        Parameters
        ----------
        x : float
            Probability input.

        Returns
        -------
        float
        """

        return -x * np.log2(x) - (1 - x)*np.log2(1 - x)


def main():
    qkd_simulator = QKDSimulator()
    print(qkd_simulator.calculate_skr())


if __name__ == '__main__':
    main()
    
    