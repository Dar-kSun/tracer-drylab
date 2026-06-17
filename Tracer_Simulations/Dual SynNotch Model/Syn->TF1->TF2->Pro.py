import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def synnotch_extended_ode(y, t, p, ligand_profile):
    SynTF, LSynTF, TF, TF_I, DNA_free, DNA_bound, mRNA1, mRNA2, Protein2, Reporter = y
    mode = ligand_profile['mode']

    if mode == 'window':
        ligand_active = t <= ligand_profile['window']
    elif mode == 'pulse':
        ligand_active = (t % ligand_profile['period']) <= ligand_profile['width']
    else:
        ligand_active = True

    if ligand_active:
        L = ligand_profile['Ligand_Conc']
        hill = (L ** p['n_Hill']) / (p['K_d'] ** p['n_Hill'] + L ** p['n_Hill'])
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
    k_deg_mRNA = p['k_deg_mRNA']
    k_deg_mRNA2 = p['k_deg_mRNA2']
    k_tl2 = p['k_tl2']
    k_deg_protein2 = p['k_deg_protein2']
    k_tl_report = p['k_tl_report']
    k_deg_report = p['k_deg_report']

    k_tx2_effective = p['k_tx2'] if ligand_active else 0.0

    dProd = k_prod
    v1 = k_L * SynTF - k_minusL * LSynTF
    v2 = k_act * SynTF + k_act_L * LSynTF
    v3 = k_on * TF * I - k_off * TF_I
    v4 = k_deg_TF * TF
    v5 = k_deg_TF_I * TF_I
    v6 = k_bind_DNA * TF * DNA_free - k_unbind_DNA * DNA_bound
    v7 = k_tx * DNA_bound
    v8 = k_deg_mRNA * mRNA1
    v9 = k_tx2_effective * mRNA1
    v10 = k_deg_mRNA2 * mRNA2
    v11 = k_tl2 * mRNA2
    v12 = k_deg_protein2 * Protein2
    v13 = k_tl_report * mRNA2
    v14 = k_deg_report * Reporter

    dSynTF     = dProd - k_L * SynTF + k_minusL * LSynTF - k_act * SynTF - gamma_S * SynTF
    dLSynTF    = v1 - k_act_L * LSynTF - gamma_L * LSynTF
    dTF        = v2 - v3 + k_off * TF_I - v4 - v6
    dTF_I      = v3 - v5
    dDNA_free  = -v6
    dDNA_bound = v6
    dmRNA1     = v7 - v8
    dmRNA2     = v9 - v10
    dProtein2  = v11 - v12
    dReporter  = v13 - v14

    return [dSynTF, dLSynTF, dTF, dTF_I, dDNA_free, dDNA_bound,
            dmRNA1, dmRNA2, dProtein2, dReporter]

# --------------------------------------------
# PROMOTER-SPECIFIC TRANSCRIPTION RATES (k_tx) (molecules/s)
# --------------------------------------------
promoter_ktx_dict = {
    'GAL4':    0.04,
    'TRE':     0.04
}

# Protein production per hour per cell (proteins/hour/cell)
promoter_protein_hr = {
    'GAL4': 3e6,
    'TRE': 2.5e6
}

# --------------------------------------------
# FIXED PARAMETERS with units
# --------------------------------------------
base_params = {
    'k_prod_SynTF': 0.01,        # molecules/s
    'k_L_TF':       0.167,       # 1/s
    'k_minusL_TF':  0.034,       # 1/s
    'k_act_TF':     0.34,        # 1/s
    'k_act_L_TF':   0.34,        # 1/s
    'gamma_SynTF':  0.1 / 3600,  # 1/s
    'gamma_LSynTF': 0.1 / 3600,  # 1/s
    'k_on':         1.0,         # 1/s
    'k_off':        0.01,        # 1/s
    'k_deg_TF':     0.05,        # 1/s
    'k_deg_TF_I':   0.02,        # 1/s
    'I':            1.0,         # conc units
    'k_bind_DNA':   0.05,        # 1/s
    'k_unbind_DNA': 0.01,        # 1/s
    'k_deg_mRNA':   0.01,        # 1/s
    'k_deg_mRNA2':  0.01,        # 1/s
    'K_d':          1.0,         # conc units
    'n_Hill':       2.0,
    'k_tl2':        10.0,        # proteins/mRNA/s
    'k_deg_protein2': 0.005,     # 1/s
    'k_tl_report':  10.0,        # default, will override
    'k_deg_report': 0.005        # 1/s
}

# --------------------------------------------
# SIMULATION SETUP
# --------------------------------------------
ligand_concentrations = [0.01, 0.1, 1.0, 10.0, 100.0]
t = np.linspace(0, 10000, 100001)
y0 = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]

# All permutations: GAL4-TRE and TRE-GAL4
permutations = [('GAL4', 'TRE'), ('TRE', 'GAL4')]

for promoter1, promoter2 in permutations:
    print(f"\n=== Configuration: SynNotch1-{promoter1}, SynNotch2-{promoter2} ===")
    p = base_params.copy()
    p['k_tx'] = promoter_ktx_dict[promoter1]     # SynNotch1
    p['k_tx2'] = promoter_ktx_dict[promoter2]    # SynNotch2
    p['k_tl_report'] = promoter_protein_hr[promoter2] / 3600.0  # proteins/mRNA/s

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for ligand_conc in ligand_concentrations:
        ligand_profile = {
            'mode': 'window',
            'window': 10 * 10,
            'Ligand_Conc': ligand_conc
        }
        sol = odeint(synnotch_extended_ode, y0, t, args=(p, ligand_profile))
        mRNA1 = sol[:, 6]
        mRNA2 = sol[:, 7]
        protein2 = sol[:, 8]
        reporter = sol[:, 9]

        axs[0].plot(t / 60, mRNA1, label=f"L {ligand_conc:.2f}")
        axs[1].plot(t / 60, mRNA2, label=f"L {ligand_conc:.2f}")
        axs[2].plot(t / 60, reporter, label=f"L {ligand_conc:.2f}")

    axs[0].set_ylabel('mRNA1 (molecules/cell)')
    axs[0].set_title(f'mRNA1 Dynamics - SynNotch1: {promoter1}')
    axs[0].grid(True)

    axs[1].set_ylabel('mRNA2 (molecules/cell)')
    axs[1].set_title(f'mRNA2 Dynamics - SynNotch2: {promoter2}')
    axs[1].grid(True)

    axs[2].set_ylabel('Reporter Protein (proteins/cell)')
    axs[2].set_title(f'Reporter Dynamics')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].axis('off')
    axs[3].text(0.01, 0.5, f"Promoter1 = {promoter1}\nPromoter2 = {promoter2}", fontsize=12)

    axs[2].set_xlabel('Time (minutes)')
    plt.tight_layout()
    plt.show()
