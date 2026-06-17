import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import ipywidgets as widgets
from IPython.display import display, clear_output

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
        ligand_conc = p['Ligand_Conc']
        hill = (ligand_conc ** p['n_Hill']) / (p['K_d'] ** p['n_Hill'] + ligand_conc ** p['n_Hill'])
    else:
        hill = 0.0

    # Apply Hill scaling to ligand-related parameters
    k_L     = p['k_L_TF'] * hill
    k_act   = p['k_act_TF'] * hill
    k_act_L = p['k_act_L_TF'] * hill

    # Unpack remaining parameters
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

    # Reaction rates
    dProd = k_prod
    v1 = k_L * SynTF - k_minusL * LSynTF
    v2 = k_act * SynTF + k_act_L * LSynTF
    v3 = k_on * TF * I - k_off * TF_I
    v4 = k_deg_TF * TF
    v5 = k_deg_TF_I * TF_I
    v6 = k_bind_DNA * TF * DNA_free - k_unbind_DNA * DNA_bound
    v7 = k_tx * DNA_bound
    v8 = k_deg_mRNA * mRNA

    dSynTF    = dProd - k_L * SynTF + k_minusL * LSynTF - k_act * SynTF - gamma_S * SynTF
    dLSynTF   = k_L * SynTF - k_minusL * LSynTF - k_act_L * LSynTF - gamma_L * LSynTF
    dTF       = v2 - v3 + k_off * TF_I - v4 - v6
    dTF_I     = v3 - v5
    dDNA_free = -v6
    dDNA_bound= v6
    dmRNA     = v7 - v8

    return [dSynTF, dLSynTF, dTF, dTF_I, dDNA_free, dDNA_bound, dmRNA]

# --------------------------------------------
# NOMINAL PARAMETER VALUES
# --------------------------------------------
nominal = {
    'k_prod_SynTF': 0.01,
    'k_L_TF':       0.167,
    'k_minusL_TF':  0.034,
    'k_act_TF':     0.34,
    'k_act_L_TF':   0.34,
    'gamma_SynTF':  0.1/3600,
    'gamma_LSynTF': 0.1/3600,
    'k_on':         1.0,
    'k_off':        0.01,
    'k_deg_TF':     0.05,
    'k_deg_TF_I':   0.02,
    'I':            1.0,
    'k_bind_DNA':   0.05,
    'k_unbind_DNA': 0.01,
    'k_tx':         0.2,
    'k_deg_mRNA':   0.01,
    'Ligand_Conc':  1.0,
    'Ligand_Count': 6.022e5,
    'K_d':          1.0,
    'n_Hill':       2.0,
    'Ligand_Interaction_Time': 50.0,
    'Pulse_Period': 20.0,
    'Pulse_Width':  5.0
}

# --------------------------------------------
# CREATE UI WIDGETS
# --------------------------------------------
param_widgets = {}
for name, val in nominal.items():
    step = val * 0.1 if val != 0 else 0.01
    param_widgets[name] = widgets.FloatText(
        value=val,
        description=name,
        step=step,
        layout=widgets.Layout(width='250px')
    )

mode_selector = widgets.Dropdown(
    options=['continuous', 'window', 'pulse'],
    value='window',
    description='Ligand Mode:',
    layout=widgets.Layout(width='250px')
)

run_button = widgets.Button(description="Run Extended ODE Model", button_style='info')
out = widgets.Output()

# --------------------------------------------
# CALLBACK FUNCTION
# --------------------------------------------
def on_run_button_click(b):
    with out:
        clear_output()

        # Read parameters
        p = {k: w.value for k, w in param_widgets.items()}
        mode = mode_selector.value
        ligand_profile = {'mode': mode}

        # Ligand time configuration
        if mode == 'window':
            ligand_profile['window'] = p.pop('Ligand_Interaction_Time')
        elif mode == 'pulse':
            ligand_profile['period'] = p.pop('Pulse_Period')
            ligand_profile['width']  = p.pop('Pulse_Width')
        else:
            p.pop('Ligand_Interaction_Time')
            p.pop('Pulse_Period')
            p.pop('Pulse_Width')

        # --- Interconvert Ligand Count <-> Concentration
        AVOGADRO_SCALE = 6.022e5  # ~1 nM in 1 μL
        p['Ligand_Count'] = p['Ligand_Conc'] * AVOGADRO_SCALE

        # --- Sensitivity Setup
        problem = {
            'num_vars': len(p),
            'names': list(p.keys()),
            'bounds': [
                [max(1e-6, v * 0.1), v * 10] if k in ['Ligand_Conc', 'K_d', 'Ligand_Count'] else [v * 0.1, v * 10]
                for k, v in p.items()
            ]
        }

        param_values = saltelli.sample(problem, 128, calc_second_order=False)
        Y_final, Y_auc = [], []
        t = np.linspace(0, 100, 1001)
        y0 = [1, 0, 0, 0, 1, 0, 0]

        for vals in param_values:
            sample_p = dict(zip(problem['names'], vals))
            sample_p['Ligand_Count'] = sample_p['Ligand_Conc'] * AVOGADRO_SCALE
            sol = odeint(synnotch_extended_ode, y0, t, args=(sample_p, ligand_profile))
            Y_final.append(sol[-1, 2])  # TF
            Y_auc.append(np.trapz(sol[:, 6], t))  # mRNA AUC

        Si_f = sobol.analyze(problem, np.array(Y_final), calc_second_order=False, print_to_console=False)
        Si_a = sobol.analyze(problem, np.array(Y_auc), calc_second_order=False, print_to_console=False)

        # --- Plot Sensitivity Results
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].bar(problem['names'], Si_f['S1'], yerr=Si_f['S1_conf'], capsize=4)
        axs[0].set_title('Sobol S1 - Final TF')
        axs[0].tick_params(axis='x', rotation=90)
        axs[1].bar(problem['names'], Si_a['S1'], yerr=Si_a['S1_conf'], capsize=4)
        axs[1].set_title('Sobol S1 - mRNA AUC')
        axs[1].tick_params(axis='x', rotation=90)
        plt.tight_layout(); plt.show()

        # --- Nominal Simulation
        sol = odeint(synnotch_extended_ode, y0, t, args=(p, ligand_profile))
        plt.figure(figsize=(8, 5))
        plt.plot(t, sol[:,2], label='TF (free)')
        plt.plot(t, sol[:,3], label='TF_I (bound)')
        plt.plot(t, sol[:,6], label='mRNA')

        if mode == 'window':
            plt.axvline(x=ligand_profile['window'], color='gray', linestyle='--', label='Ligand Ends')
        elif mode == 'pulse':
            pulse_period = ligand_profile['period']
            pulse_width = ligand_profile['width']
            max_pulses = min(100, int(t[-1] / pulse_period))
            for i in range(max_pulses):
                start = i * pulse_period
                end = start + pulse_width
                plt.axvspan(start, end, color='gray', alpha=0.2)

        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title(f'Extended SynNotch Dynamics ({mode})')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Attach callback
run_button.on_click(on_run_button_click)

# --------------------------------------------
# DISPLAY UI
# --------------------------------------------
ui_layout = widgets.VBox([
    widgets.HBox([
        widgets.VBox(list(param_widgets.values())[:10]),
        widgets.VBox(list(param_widgets.values())[10:])
    ]),
    mode_selector,
    run_button,
    out
])

display(widgets.HTML('<h3>Extended SynNotch ODE Model with Hill Function</h3>'))
display(ui_layout)
