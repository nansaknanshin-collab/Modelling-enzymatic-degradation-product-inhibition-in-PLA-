import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Kinetic parameters from Hegyesi fit (inhibition model)
# ---------------------------------------------------------
k1 = 9.3e1
k2 = 6.1e1
k3 = 2.8e1
k4 = 7.0e1
k5 = 3.6e-2
params_inhib = (k1, k2, k3, k4, k5)
params_no    = (k1, k2, k3)   # same k1–k3 for no-inhibition

# ---------------------------------------------------------
# ODEs  (unchanged except bug fix: return dP)
# ---------------------------------------------------------
def rhs_inhib(t, y, k1, k2, k3, k4, k5):
    """Reversible inhibition model."""
    E, S, ES, ESP, P = y
    dE   = -k1*E*S + (k2 + k3)*ES
    dS   = -k1*E*S + k2*ES
    dES  = k1*E*S - (k2 + k3)*ES - k4*ES*P + k5*ESP
    dESP = k4*ES*P - k5*ESP
    dP   = k3*ES - k4*ES*P + k5*ESP
    return [dE, dS, dES, dESP, dP]

def rhs_no_inhib(t, y, k1, k2, k3):
    """Simple Michaelis–Menten (no inhibition)."""
    E, S, ES, P = y
    dE  = -k1*E*S + (k2 + k3)*ES
    dS  = -k1*E*S + k2*ES
    dES = k1*E*S - (k2 + k3)*ES
    dP  = k3*ES
    return [dE, dS, dES, dP]

# =========================================================
# A) CUI: NO BUFFER CHANGE (your second script)
# =========================================================
t_data_no = np.array([9.6, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
y_data_no = np.array([12.1, 12.1, 11.9, 11.9, 11.9,
                      11.8, 11.4, 11.1, 10.7, 8.7, 0.0], dtype=float)

# sort to increasing time
idx_no = np.argsort(t_data_no)
t_data_no = t_data_no[idx_no]
y_data_no = y_data_no[idx_no]

# Initial conditions (single continuous run, no refresh)
E0 = 0.02
S0 = 0.3
y0_inhib_no = [E0, S0, 0.0, 0.0, 0.0]
y0_no_no    = [E0, S0, 0.0, 0.0]

t_end_no = t_data_no[-1]
t_plot_no = np.linspace(0.0, t_end_no, 400)

# --- Reversible inhibition model
sol_inhib_no = solve_ivp(
    rhs_inhib, (0.0, t_end_no), y0_inhib_no,
    t_eval=t_plot_no, args=params_inhib,
    method="BDF", rtol=1e-8, atol=1e-10
)
if not sol_inhib_no.success:
    raise RuntimeError("Inhibition solver failed (no-change case)")

P_inhib_no = sol_inhib_no.y[4]
WL_inhib_no = 100.0 * P_inhib_no

# --- No-inhibition model
sol_no_no = solve_ivp(
    rhs_no_inhib, (0.0, t_end_no), y0_no_no,
    t_eval=t_plot_no, args=params_no,
    method="BDF", rtol=1e-8, atol=1e-10
)
if not sol_no_no.success:
    raise RuntimeError("No-inhibition solver failed (no-change case)")

P_no_no = sol_no_no.y[3]
WL_no_no = 100.0 * P_no_no

# =========================================================
# B) CUI: DAILY BUFFER CHANGE (your first script)
# =========================================================
t_data_daily = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8.6], dtype=float)
y_data_daily = np.array([0, 10, 14.5, 24.2, 31.8, 40.1,
                         47.7, 60.6, 68, 73.1], dtype=float)

# Initial conditions per *day* (refreshed every day)
E0_day  = 0.02
S0_day  = 0.3
ES0_day = 0.0
ESP0_day = 0.0
P0_day  = 0.0

change_times = t_data_daily

t_full_daily = []
P_cum_full_daily = []
P_cum_offset = 0.0

for i in range(len(change_times) - 1):
    t_start = change_times[i]
    t_end   = change_times[i+1]
    dt      = t_end - t_start

    t_local = np.linspace(0.0, dt, 100)
    y0_day = [E0_day, S0_day, ES0_day, ESP0_day, P0_day]

    sol_day = solve_ivp(
        rhs_inhib, (0.0, dt), y0_day,
        t_eval=t_local, args=params_inhib,
        method="BDF", rtol=1e-8, atol=1e-10
    )
    if not sol_day.success:
        raise RuntimeError("Inhibition ODE failed on day", i+1)

    P_local = sol_day.y[4]
    t_segment = t_local + t_start
    P_cum_segment = P_cum_offset + P_local

    if i == 0:
        t_full_daily.extend(t_segment)
        P_cum_full_daily.extend(P_cum_segment)
    else:
        t_full_daily.extend(t_segment[1:])
        P_cum_full_daily.extend(P_cum_segment[1:])

    P_cum_offset += P_local[-1]

t_full_daily = np.array(t_full_daily)
P_cum_full_daily = np.array(P_cum_full_daily)
WL_inhib_daily = 100.0 * P_cum_full_daily

# No-inhibition model WITHOUT refresh (single continuous run) – as in your first script
y0_no_daily = [E0_day, S0_day, 0.0, 0.0]
t_no_daily = np.linspace(0.0, t_data_daily[-1], 500)

sol_no_daily = solve_ivp(
    rhs_no_inhib, (0.0, t_data_daily[-1]), y0_no_daily,
    t_eval=t_no_daily, args=params_no,
    method="BDF", rtol=1e-8, atol=1e-10
)
if not sol_no_daily.success:
    raise RuntimeError("No-inhibition ODE failed (daily-change case)")

P_no_daily = sol_no_daily.y[3]
WL_no_daily = 100.0 * P_no_daily

# =========================================================
# SINGLE PLOT: both datasets + both models on same axes
# =========================================================
# =========================================================
# SINGLE PLOT: both datasets + both models on same axes
# =========================================================
plt.figure(figsize=(8, 5))

# Experimental data
plt.scatter(t_data_no,    y_data_no,    c='k', s=80,
            label='No buffer change (data)')
plt.scatter(t_data_daily, y_data_daily, facecolors='none', edgecolors='k',
            s=80, marker='s', label='Daily buffer change (data)')

# Models
# Red = inhibition model (both protocols)
plt.plot(t_plot_no,   WL_inhib_no,   'r-',  lw=3,
         label='Reversible Inhibition model')
plt.plot(t_full_daily, WL_inhib_daily, 'r-', lw=3,
         label='_nolegend_')  # same model, different protocol (no legend)

# Blue = no-inhibition model (both protocols)
plt.plot(t_plot_no,   WL_no_no,      'b-',  lw=3,
         label='No-inhibition model')
plt.plot(t_no_daily,   WL_no_daily,    'b--', lw=3,
         label='_nolegend_')  # same model, different protocol (no legend)

plt.xlabel('Time (days)', fontsize=20)
plt.ylabel('Weight loss (%)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("cui_validation_singleplot.png", dpi=600)
plt.show()
