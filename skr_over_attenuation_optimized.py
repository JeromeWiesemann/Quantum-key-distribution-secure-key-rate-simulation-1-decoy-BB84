import matplotlib.pyplot as plt
import qkdsimulator as qkdsimulator
from scipy.optimize import minimize
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors

num_curves = 6
colors = cm.viridis_r(np.linspace(0, 1, num_curves))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(5.5, 3.8))

def eta_to_db(eta):
    db = 10*np.log10(eta)
    return db

def skr_from_param(x):
    qkdsim.qkd_parameters.mu_1 = x[0]
    qkdsim.qkd_parameters.mu_2 = x[1]

    qkdsim.qkd_parameters.P_mu_1 = x[2]
    qkdsim.qkd_parameters.P_mu_2 = 1 - x[2]

    qkdsim.qkd_parameters.P_X_alice = x[3]
    qkdsim.qkd_parameters.P_Z_alice = 1 - x[3]
    qkdsim.qkd_parameters.P_X_bob = x[3]
    qkdsim.qkd_parameters.P_Z_bob = 1 - x[3]

    return -qkdsim.calculate_skr() # return -skr because scipy minimizes -skr and so maximizes skr

nb_signals_exp_list_alice = [6, 8, 10, 12, np.inf]
nb_signals_exp_list_bob = [5, 6, 7, 8, 9]

for list_index, list_signals in enumerate([nb_signals_exp_list_alice, nb_signals_exp_list_bob]):
    for i, nb_signals_exp in enumerate(list_signals):
        qkdsim = qkdsimulator.QKDSimulator()
        qkdsim.qkd_parameters.concentration_inequalities_method = "Hoeffding"

        if nb_signals_exp == np.inf:
            qkdsim.qkd_parameters.asymptotic = True
        else:
            nb_signals = pow(10, nb_signals_exp)
        qkdsim.qkd_parameters.N_bob = nb_signals
        qkdsim.qkd_parameters.N_alice = nb_signals
        if list_index == 0:
            qkdsim.qkd_parameters.fix_alice = True
            line = "solid"
            label = r"$N$"
        else: 
            qkdsim.qkd_parameters.fix_alice = False
            line = "dashed"
            label = r"$N_{\mathrm{block}}$"

        secret_key_lengths = []
        distances = []
        temps = []
        x0_initial = np.array([0.5, 0.15, 0.5, 0.5]) # initial optimization parameters
        x0 = x0_initial

        bnds = ((0.00001, None), (0.00001, None), (0.00001, 0.99999), (0.00001, 0.99999)) # Bounds for mu_1, mu_2, P_mu_1 and P_X
        constraints = [{'type': 'ineq', 'fun': lambda x: x[0] - x[1] -0.0001}]

        for L in np.arange(0, 280, 1):
            print(f"Optimizing L={L}...")
                
            qkdsim.qkd_parameters.set_channel_length(L)

            # res = minimize(skr_from_param, x0,  options={"disp": False}, method = "trust-constr", bounds = bnds, constraints=constraints)

            res = minimize(skr_from_param, x0,  options={"disp": False, "maxiter": 100000}, method = "nelder-Mead", bounds = bnds)
            if res.success:
                # secret_key_lengths.append(skr_from_param(res.x))
                secret_key_lengths.append(-skr_from_param(res.x))
                x0 = np.array(res.x) # Use optimal parameters from this round as initial parameters for next round
                distances.append(-eta_to_db(qkdsim.qkd_parameters.eta_sys))
            else:
                print("[WARNING]: Unsuccessful optimization!")

        if nb_signals_exp == np.inf:
            plt.plot(distances, secret_key_lengths, marker = "None", label = rf"$N$ = $\infty$", color=colors[i], linestyle = "solid")
            continue
        plt.plot(distances, secret_key_lengths, marker = "None", label = rf"{label} = $10^{{{nb_signals_exp}}}$", color=colors[i], linestyle = line)


plt.xlabel("Attenuation [dB]")
plt.ylabel("Secure-key rate [bits/s]")
plt.grid(visible=True, linestyle ="dotted")
plt.yscale("log")
plt.ylim(bottom = 3)
plt.xlim(right = 65)
plt.tick_params(direction="in", top=True, right=True)
plt.tick_params(direction="in", which = "minor")
plt.tight_layout()
plt.legend(loc="upper right")
plt.savefig("SKR_EUR.pdf", dpi = 400)
plt.show()