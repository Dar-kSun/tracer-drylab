import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def synnotch_extended_ode(y, t, p, ligand_profile):
    SynTF, LSynTF, TF, TF_I, DNA_free, DNA_bound, mRNA = y
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
    k_deg_mRNA = p['k_deg_mRNA']

    dProd = k_prod
    v1 = k_L * SynTF - k_minusL * LSynTF
    v2 = k_act * SynTF + k_act_L * LSynTF
    v3 = k_on * TF * I - k_off * TF_I
    v4 = k_deg_TF * TF
    v5 = k_deg_TF_I * TF_I
    v6 = k_bind_DNA * TF * DNA_free - k_unbind_DNA * DNA_bound
    v7 = k_tx * DNA_bound
    v8 = k_deg_mRNA * mRNA

    dSynTF     = dProd - k_L * SynTF + k_minusL * LSynTF - k_act * SynTF - gamma_S * SynTF
    dLSynTF    = v1 - k_act_L * LSynTF - gamma_L * LSynTF
    dTF        = v2 - v3 + k_off * TF_I - v4 - v6
    dTF_I      = v3 - v5
    dDNA_free  = -v6
    dDNA_bound = v6
    dmRNA      = v7 - v8

    return [dSynTF, dLSynTF, dTF, dTF_I, dDNA_free, dDNA_bound, dmRNA]

# --------------------------------------------
# FIXED PARAMETERS
# --------------------------------------------
p = {
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
    'k_tx':         0.2,
    'k_deg_mRNA':   0.01,
    'K_d':          1.0,
    'n_Hill':       2.0
}

# --------------------------------------------
# TIME TO PEAK, 10%-90% RISE/FALL ANALYSIS
# --------------------------------------------
ligand_concentrations = [0.01, 0.1, 1.0, 10.0, 100.0]
rise_times = []
fall_times = []
peak_times = []

plt.figure(figsize=(12, 6))
for ligand_conc in ligand_concentrations:
    ligand_profile = {'mode': 'window', 'window': 1.0, 'Ligand_Conc': ligand_conc}
    t = np.linspace(0, 1000, 10001)
    y0 = [1, 0, 0, 0, 1, 0, 0]
    sol = odeint(synnotch_extended_ode, y0, t, args=(p, ligand_profile))
    mRNA = sol[:, 6]

    peak_val = np.max(mRNA)
    peak_time = t[np.argmax(mRNA)]
    peak_times.append(peak_time)

    t_10 = next((t[i] for i in range(len(t)) if mRNA[i] >= 0.1 * peak_val), np.nan)
    t_90 = next((t[i] for i in range(len(t)) if mRNA[i] >= 0.9 * peak_val), np.nan)
    t_fall = next((t[i] for i in range(np.argmax(mRNA), len(t)) if mRNA[i] <= 0.1 * peak_val), np.nan)

    rise_times.append((t_10, t_90))
    fall_times.append((peak_time, t_fall))

    print(f"Ligand = {ligand_conc:6.2f} | Peak mRNA = {peak_val:.4f} at t = {peak_time:.2f} | Rise: {t_10:.2f} → {t_90:.2f} | Fall: {peak_time:.2f} → {t_fall:.2f}")
    plt.plot(t, mRNA, label=f"Ligand {ligand_conc:.2f}")

plt.xlabel('Time')
plt.ylabel('mRNA Concentration')
plt.title('mRNA Dynamics for Various Ligand Concentrations (1-unit exposure)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
