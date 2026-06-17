import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def synnotch_extended_ode(y, t, p, ligand_profile):
    SynTF, LSynTF, TF, TF_I, DNA_free, DNA_bound, mRNA1, mRNA2, Protein2 = y
    mode = ligand_profile['mode']

    if mode == 'window':
        ligand_active = t <= ligand_profile['window']
    elif mode == 'pulse':
        period = ligand_profile['period']
        pulse_width = ligand_profile['width']
        ligand_active = (t % period) <= pulse_width
    else:
        ligand_active = True

    if ligand_active:
        ligand_conc = ligand_profile['Ligand_Conc']
        hill = (ligand_conc ** p['n_Hill']) / (p['K_d'] ** p['n_Hill'] + ligand_conc ** p['n_Hill'])
    else:
        hill = 0.0

    k_L     = p['k_L_TF'] * hill
    k_act   = p['k_act_TF'] * hill
    k_act_L = p['k_act_L_TF'] * hill

    k_prod = p['k_prod_SynTF']
    k_minusL = p['k_minusL_TF']
    gamma_S = p['gamma_SynTF']
    gamma_L = p['gamma_LSynTF']
    k_on = p['k_on']
    k_off = p['k_off']
    I = p['I']
    k_deg_TF = p['k_deg_TF']
    k_deg_TF_I = p['k_deg_TF_I']
    k_bind_DNA = p['k_bind_DNA']
    k_unbind_DNA = p['k_unbind_DNA']
    k_tx = p['k_tx']
    k_tx2 = p['k_tx2']
    k_deg_mRNA = p['k_deg_mRNA']
    k_deg_mRNA2 = p['k_deg_mRNA2']
    k_tl2 = p['k_tl2']
    k_deg_protein2 = p['k_deg_protein2']

    dProd = k_prod
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
    v12 = k_deg_protein2 * Protein2

    dSynTF     = dProd - k_L * SynTF + k_minusL * LSynTF - k_act * SynTF - gamma_S * SynTF
    dLSynTF    = v1 - k_act_L * LSynTF - gamma_L * LSynTF
    dTF        = v2 - v3 + k_off * TF_I - v4 - v6
    dTF_I      = v3 - v5
    dDNA_free  = -v6
    dDNA_bound = v6
    dmRNA1     = v7 - v8
    dmRNA2     = v9 - v10
    dProtein2  = v11 - v12

    return [dSynTF, dLSynTF, dTF, dTF_I, dDNA_free, dDNA_bound, dmRNA1, dmRNA2, dProtein2]

# --------------------------------------------
# PROMOTER-SPECIFIC TRANSCRIPTION RATES (k_tx)
# --------------------------------------------
promoter_ktx_dict = {
    'SFFV':    0.04,
    'CAG':     0.035,
    'CMV':     0.03,
    'EF1α':    0.025,
    'SV40':    0.015,
    'PGK':     0.01,
    'UbC':     0.006,
    'TRE':     0.04
}

# Protein production per hour per cell (center of range estimates)
promoter_protein_hr = {
    'SFFV': 3e6,
    'CAG': 2.4e6,
    'CMV': 1.75e6,
    'EF1α': 1.2e6,
    'SV40': 0.85e6,
    'PGK': 0.45e6,
    'UbC': 0.325e6,
    'TRE': 2.5e6
}

# --------------------------------------------
# FIXED PARAMETERS
# --------------------------------------------
base_params = {
    'k_prod_SynTF': 0.01,
    'k_L_TF':       0.167,
    'k_minusL_TF':  0.034,
    'k_act_TF':     0.34,
    'k_act_L_TF':   0.34,
    'gamma_SynTF':  0.1 / 3600,
    'gamma_LSynTF': 0.1 / 3600,
    'k_on':         1.0,
    'k_off':        0.01,
    'k_deg_TF':     0.05,
    'k_deg_TF_I':   0.02,
    'I':            1.0,
    'k_bind_DNA':   0.05,
    'k_unbind_DNA': 0.01,
    'k_deg_mRNA':   0.01,
    'k_deg_mRNA2':  0.01,
    'K_d':          1.0,
    'n_Hill':       2.0,
    'k_tl2':        10.0,
    'k_deg_protein2': 0.005
}

# --------------------------------------------
# SIMULATION FOR MRNA1, MRNA2 AND PROTEIN2
# --------------------------------------------
ligand_concentrations = [0.01, 0.1, 1.0, 10.0, 100.0]
t = np.linspace(0, 10000, 100001)
y0 = [1, 0, 0, 0, 1, 0, 0, 0, 0]

for promoter, k_tx in promoter_ktx_dict.items():
    print(f"\n=== Promoter: {promoter} (k_tx = {k_tx}) ===")
    p = base_params.copy()
    p['k_tx'] = k_tx
    p['k_tx2'] = k_tx

    protein_scaling = promoter_protein_hr[promoter] / 3600  # per second per cell

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ligand_conc in ligand_concentrations:
        ligand_profile = {'mode': 'window', 'window': 1.0 * 100, 'Ligand_Conc': ligand_conc}
        sol = odeint(synnotch_extended_ode, y0, t, args=(p, ligand_profile))
        mRNA1 = sol[:, 6]
        mRNA2 = sol[:, 7]
        protein2 = sol[:, 8] * protein_scaling  # convert to estimated protein number per cell

        axs[0].plot(t / 6, mRNA1, label=f'L {ligand_conc:.2f}')
        axs[1].plot(t / 6, mRNA2, label=f'L {ligand_conc:.2f}')
        axs[2].plot(t / 6, protein2, label=f'L {ligand_conc:.2f}')

    axs[0].set_ylabel('mRNA1 Level')
    axs[0].set_title(f'mRNA1 Dynamics - {promoter}')
    axs[0].grid(True)

    axs[1].set_ylabel('mRNA2 Level')
    axs[1].set_title(f'mRNA2 Dynamics - {promoter}')
    axs[1].grid(True)

    axs[2].set_xlabel('Time (minutes)')
    axs[2].set_ylabel('Protein2 (SynNotch2)')
    axs[2].set_title(f'SynNotch2 Protein Dynamics - {promoter}')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
