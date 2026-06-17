import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def synnotch_tf1_tf2_ode(y, t, p):
    SynTF, LSynTF, TF, TF_I, DNA_free, DNA_bound, mRNA1, mRNA2, TF2, I = y

    # No ligand input
    hill = 0.0

    # Parameters
    k_L     = p['k_L_TF'] * hill                 # Ms-1
    k_act   = p['k_act_TF'] * hill               # s-1
    k_act_L = p['k_act_L_TF'] * hill             # s-1
    k_prod = p['k_prod_SynTF']                   # mMs-1
    k_minusL = p['k_minusL_TF']                  # s-1
    gamma_S = p['gamma_SynTF']                   # s-1
    gamma_L = p['gamma_LSynTF']                  # s-1
    k_on = p['k_on']                             # M-1s-1
    k_off = p['k_off']                           # s-1
    k_deg_TF = p['k_deg_TF']                     # s-1
    k_deg_TF_I = p['k_deg_TF_I']                 # s-1
    k_bind_DNA = p['k_bind_DNA']                 # mM-1s-1
    k_unbind_DNA = p['k_unbind_DNA']             # s-1
    k_tx = p['k_tx']                             # mMs-1
    k_tx2 = p['k_tx2']                           # mMs-1
    k_deg_mRNA = p['k_deg_mRNA']                 # s-1
    k_deg_mRNA2 = p['k_deg_mRNA2']               # s-1
    k_tl2 = p['k_tl2']                           # s-1
    k_deg_TF2 = p['k_deg_TF2']                   # s-1
    k_prod_I = k_prod                            # mMs-1 (same promoter as SynTF)
    k_deg_I = p['k_deg_I']                       # s-1

    # Reactions
    dProd_SynTF = k_prod
    dProd_I = k_prod_I
    v1 = k_L * SynTF - k_minusL * LSynTF
    v2 = k_act * SynTF + k_act_L * LSynTF
    v3 = k_on * TF * I - k_off * TF_I
    v4 = k_deg_TF * TF
    v5 = k_deg_TF_I * TF_I
    v6 = k_bind_DNA * TF * DNA_free - k_unbind_DNA * DNA_bound
    v7 = k_tx * DNA_bound
    v8 = k_deg_mRNA * mRNA1
    v9 = k_tx2 * mRNA1
    v10 = k_deg_mRNA2 * mRNA2
    v11 = k_tl2 * mRNA2
    v12 = k_deg_TF2 * TF2
    v13 = k_deg_I * I

    # ODEs
    dSynTF     = dProd_SynTF - k_L * SynTF + k_minusL * LSynTF - k_act * SynTF - gamma_S * SynTF
    dLSynTF    = v1 - k_act_L * LSynTF - gamma_L * LSynTF
    dTF        = v2 - v3 + k_off * TF_I - v4 - v6
    dTF_I      = v3 - v5
    dDNA_free  = -v6
    dDNA_bound = v6
    dmRNA1     = v7 - v8
    dmRNA2     = v9 - v10
    dTF2       = v11 - v12
    dI         = dProd_I - v3 + k_off * TF_I - v13

    return [dSynTF, dLSynTF, dTF, dTF_I, dDNA_free, dDNA_bound, dmRNA1, dmRNA2, dTF2, dI]

# --------------------------------------------
# BASE PARAMETERS (same for all except k_prod_SynTF)
# --------------------------------------------
base_params = {
    'k_L_TF':       0.167,           # s-1
    'k_minusL_TF':  0.034,           # s-1
    'k_act_TF':     0.0034,          # s-1
    'k_act_L_TF':   0.34,            # s-1
    'gamma_SynTF':  0.1 / 3600,      # s-1
    'gamma_LSynTF': 0.1 / 3600,      # s-1
    'k_on':         1,               # uM-1s-1
    'k_off':        0.0047,          # s-1
    'k_deg_TF':     0.0014/60,       # s-1
    'k_deg_TF_I':   0.0014/60,       # s-1
    'k_tx':         2.9e-4,          # uMs-1
    'k_tx2':        2.9e-4,          # uMs-1
    'k_bind_DNA':   1,               # uMs-1
    'k_unbind_DNA': 420,             # s-1
    'k_deg_mRNA':   0.0012/60,       # s-1
    'k_deg_mRNA2':  0.0012/60,       # s-1
    'k_tl2':        10.0,            # s-1
    'k_deg_TF2':    0.05,            # s-1
    'k_deg_I':      0.0014/60,       # s-1
    'K_d':          1.0,
    'n_Hill':       2.0
}

# --------------------------------------------
# SIMULATION CONFIGURATION
# --------------------------------------------
promoter_rates = {
    'SFFV':  0.0069,
    'CAG':   0.0046,
    'CMV':   0.004025,
    'EF1a':  0.00276,
    'SV40':  0.000875,
    'PGK':   0.00045,
    'UbC':   0.000325
}

t = np.linspace(0, 360000, 36001)  # 100 hours
y0 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # Initial species concentrations

# --------------------------------------------
# PLOT: SYNNotch DYNAMICS
# --------------------------------------------
plt.figure(figsize=(12, 6))
for name, rate in promoter_rates.items():
    params = base_params.copy()
    params['k_prod_SynTF'] = rate
    sol = odeint(synnotch_tf1_tf2_ode, y0, t, args=(params,))
    SynTF = sol[:, 0]
    Inhibitor = sol[:, 9]

    max_syn = np.max(SynTF)
    max_inhibitor = np.max(Inhibitor)
    print(f"[{name}] Max SynNotch concentration:   {max_syn:.6f} uM")
    print(f"[{name}] Max Inhibitor concentration: {max_inhibitor:.6f} uM\n")

    plt.plot(t / 3600, SynTF, label=f'{name} ({rate})')

plt.xlabel('Time (hours)')
plt.ylabel('SynNotch (SynTF) concentration [uM]')
plt.title('SynNotch Dynamics for Different Promoters')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------
# PLOT: INHIBITOR DYNAMICS
# --------------------------------------------
plt.figure(figsize=(12, 6))
for name, rate in promoter_rates.items():
    params = base_params.copy()
    params['k_prod_SynTF'] = rate
    sol = odeint(synnotch_tf1_tf2_ode, y0, t, args=(params,))
    Inhibitor = sol[:, 9]
    plt.plot(t / 3600, Inhibitor, label=f'{name} ({rate})')

plt.xlabel('Time (hours)')
plt.ylabel('Inhibitor concentration [uM]')
plt.title('Inhibitor Dynamics for Different Promoters')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
