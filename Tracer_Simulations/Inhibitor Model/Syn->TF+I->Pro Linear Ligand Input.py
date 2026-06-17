import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- ODE System ---
def synnotch_extended_ode(t, y, p):
    SynTF, LSynTF, TF, TF_I, DNA_free, DNA_bound, mRNA, I, Ligand = y

    # Reactions
    v1 = p['k_L_TF'] * SynTF * Ligand - p['k_minusL_TF'] * LSynTF
    v2 = p['k_act_TF'] * SynTF + p['k_act_L_TF'] * LSynTF
    v3 = p['k_on'] * TF * I - p['k_off'] * TF_I
    v4 = p['k_deg_TF'] * TF
    v5 = p['k_deg_TF_I'] * TF_I
    v6 = p['k_bind_DNA'] * TF * DNA_free - p['k_unbind_DNA'] * DNA_bound
    v7 = p['k_tx'] * DNA_bound
    v8 = p['k_deg_mRNA'] * mRNA

    # ODEs
    dSynTF     = -p['k_L_TF'] * SynTF * Ligand + p['k_minusL_TF'] * LSynTF - p['k_act_TF'] * SynTF - p['gamma_SynTF'] * SynTF
    dLSynTF    = v1 - p['k_act_L_TF'] * LSynTF - p['gamma_LSynTF'] * LSynTF
    dTF        = v2 - v3 + p['k_off'] * TF_I - v4 - v6
    dTF_I      = v3 - v5
    dDNA_free  = -v6
    dDNA_bound = v6
    dmRNA      = v7 - v8
    dI         = p['I_max'] * p['k_deg_I'] - p['k_deg_I'] * I - v3 + v5
    dLigand    = -p['k_L_TF'] * SynTF * Ligand + p['k_minusL_TF'] * LSynTF  # only consumption

    return [dSynTF, dLSynTF, dTF, dTF_I, dDNA_free, dDNA_bound, dmRNA, dI, dLigand]

# --- Parameters (fixed) ---
base_params = {
    'k_prod_SynTF': 0.01,
    'k_L_TF':       0.167,
    'k_minusL_TF':  0.034,
    'k_act_TF':     0.00034,
    'k_act_L_TF':   0.34,
    'gamma_SynTF':  0.1 / 3600,
    'gamma_LSynTF': 0.1 / 3600,
    'k_on':         1.0,
    'k_off':        0.01,
    'k_deg_TF':     0.05,
    'k_deg_TF_I':   0.02,
    'k_bind_DNA':   0.05,
    'k_unbind_DNA': 0.01,
    'k_tx':         0.2,
    'k_deg_mRNA':   0.01,
    'k_deg_I':      0.01,
    'I_max':        300,
    'Ligand_Start': 200.0,
}

# --- Ligand concentrations to test ---
ligand_concs = [10, 50, 100, 200, 250, 275, 300, 315, 325, 350, 400, 500, 700, 1000]
mrna_max_vals = []

# --- Time settings ---
t_end = 1000

# --- Run simulations ---
for ligand in ligand_concs:
    # Update params
    p = base_params.copy()
    p['Ligand_Max'] = ligand

    # Initial conditions
    SynTF_ss = p['k_prod_SynTF'] / p['gamma_SynTF']
    y0 = [
        SynTF_ss,  # SynTF
        0.0,       # LSynTF
        0.0,       # TF
        0.0,       # TF_I
        1.0,       # DNA_free
        0.0,       # DNA_bound
        0.0,       # mRNA
        0.0,       # Inhibitor
        0.0        # Ligand (initially 0)
    ]

    # Phase 1: before ligand
    t1_span = (0, p['Ligand_Start'])
    t1_eval = np.linspace(*t1_span, int(p['Ligand_Start']) + 1)
    sol1 = solve_ivp(synnotch_extended_ode, t1_span, y0, args=(p,), t_eval=t1_eval, method='LSODA')

    # Phase 2: after ligand
    y1 = sol1.y[:, -1].copy()
    y1[8] = ligand
    t2_span = (p['Ligand_Start'], t_end)
    t2_eval = np.linspace(*t2_span, int(t_end - p['Ligand_Start']) + 1)
    sol2 = solve_ivp(synnotch_extended_ode, t2_span, y1, args=(p,), t_eval=t2_eval, method='LSODA')

    # Combine results
    y_combined = np.hstack([sol1.y, sol2.y])
    mRNA_vals = y_combined[6, :]
    mrna_max_vals.append(np.max(mRNA_vals))

# --- Plot results ---
plt.figure(figsize=(8, 6))
plt.plot(ligand_concs, mrna_max_vals, marker='o', linestyle='-')
plt.xlabel('Ligand Concentration (nM)')
plt.ylabel('Max mRNA Concentration (nM)')
plt.title('Max mRNA vs Ligand Concentration (300 nM Inhibitor)')
plt.grid(True)
plt.tight_layout()
plt.show()
