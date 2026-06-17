import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --------------------------
# TIME SETUP
# --------------------------
# 1 unit = 10 seconds → simulate for 1000 units = 10,000 seconds
t_units = np.linspace(0, 100000, 1000)
t_sec = t_units * 10

# --------------------------
# BASE PARAMETERS
# --------------------------
params = {
    'k_prod_SynTF':   0.01,
    'k_L_TF':         0.167,
    'k_minusL_TF':    0.034,
    'k_act_TF':       0.34,
    'k_act_L_TF':     0.34,
    'gamma_SynTF':    0.1 / 3600,
    'gamma_LSynTF':   0.1 / 3600,
    'k_tx2':          1e-3,
    'k_deg_mRNA2':    1e-4,
    'k_tl2':          5e-4,
    'k_tx_r':         1e-3,
    'k_deg_mRNA_r':   1e-4,
    'k_tl_r':         0.04,
    'k_deg_P':        np.log(2) / 150,
    'Ligand_Conc':    1.0,
    'K_d':            1.0,
    'n_Hill':         2.0,
}

# --------------------------
# DEGRON-SPECIFIC DECAY RATES (k_deg in s⁻¹)
# --------------------------
degron_kdeg = {
    'IκBα AR6':            7.70e-4,
    'AID (Auxin-Inducible)': 2.89e-4,
    'mIAA7 (AID2 variant)': 1.28e-3,
    'UbR (Ub-N-end rule)': 1.93e-4,
    'UbV':                 1.29e-4,
    'PEST (CL1-derived)':  5.50e-5
}

# --------------------------
# LIGAND PROFILE
# --------------------------
ligand_profile = {'mode': 'window', 'window': 5 * 10}  # 5 units × 10 sec = 50 sec exposure

# --------------------------
# ODE SYSTEM: SynNotch → TF1 + degron n → TF2 + degron m → Reporter
# --------------------------
def synnotch_tier2_ode(y, t, p, kdeg1, kdeg2):
    SynTF, LSynTF, TF1, mRNA2, TF2, mRNA_r, P = y

    # Ligand active within window
    ligand_active = t <= ligand_profile['window']
    hill = (p['Ligand_Conc']**p['n_Hill']) / (p['K_d']**p['n_Hill'] + p['Ligand_Conc']**p['n_Hill']) if ligand_active else 0.0

    # Tier 1 activation
    k_L   = p['k_L_TF'] * hill
    k_act = p['k_act_TF'] * hill
    dSynTF   = p['k_prod_SynTF'] - k_L*SynTF + p['k_minusL_TF']*LSynTF - k_act*SynTF - p['gamma_SynTF']*SynTF
    dLSynTF  = k_L*SynTF - p['k_minusL_TF']*LSynTF - p['gamma_LSynTF']*LSynTF
    dTF1     = k_act*SynTF + p['k_act_L_TF']*LSynTF - kdeg1*TF1

    # Tier 2 expression (TF2)
    dmRNA2 = p['k_tx2']*TF1 - p['k_deg_mRNA2']*mRNA2
    dTF2   = p['k_tl2']*mRNA2 - kdeg2*TF2

    # Reporter expression
    dmRNA_r = p['k_tx_r']*TF2 - p['k_deg_mRNA_r']*mRNA_r
    dP      = p['k_tl_r']*mRNA_r - p['k_deg_P']*P

    return [dSynTF, dLSynTF, dTF1, dmRNA2, dTF2, dmRNA_r, dP]

# --------------------------
# INITIAL CONDITIONS
# --------------------------
y0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SynTF, LSynTF, TF1, mRNA2, TF2, mRNA_r, Reporter

# --------------------------
# SIMULATION: Matrix of Degron Combinations
# --------------------------
for n_name, kdeg1 in degron_kdeg.items():
    plt.figure(figsize=(9, 5))
    for m_name, kdeg2 in degron_kdeg.items():
        if m_name == n_name:
            continue  # skip diagonal
        sol = odeint(synnotch_tier2_ode, y0, t_sec, args=(params, kdeg1, kdeg2))
        reporter = sol[:, 6]
        plt.plot(t_units, reporter, label=f'TF2 + {m_name}')
    
    # Plot formatting
    plt.axvline(x=5, color='gray', linestyle='--', label='Ligand OFF')
    plt.title(f'SynNotch > TF1 → TF2 + {n_name} → Reporter')
    plt.xlabel('Time (units of 10 s)')
    plt.ylabel('Reporter Protein (arb. units)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

plt.show()
