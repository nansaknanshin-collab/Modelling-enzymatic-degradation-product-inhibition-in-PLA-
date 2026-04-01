import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
from pyswarm import pso

# ------------------ Utils ------------------

def calculate_bic(n, rss, num_params):
    """Bayesian Information Criterion from RSS."""
    return n * np.log(rss / n) + num_params * np.log(n)

def get_publish_folder():
    path_out = r'C:\Users\S00231993\Downloads'
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    return path_out

# ------------------ No-Inhibition model ------------------

def simulate_no_inhib(tspan, ic, p):
    k1, k2, k3 = p

    def rhs(y, t, p):
        E, S, ES, P = y
        dE  = -k1*E*S + (k2 + k3)*ES
        dS  = -k1*E*S + k2*ES
        dES =  k1*E*S - (k2 + k3)*ES
        dP  =  k3*ES
        return [dE, dS, dES, dP]

    sol = odeint(rhs, ic, tspan, args=(p,))
    P = sol[:, 3]
    return P * 100.0  # convert to percent

def fit_no_inhibition_model(time_all, data_all):
    # Initial conditions: [E, S, ES, P]
    ic = [0.03, 0.3, 0.0, 0.0]  # S0=0.3 → max P≈30%

    # Bounds for [k1, k2, k3]
    lb = np.array([0, 0, 0])
    ub = np.array([200, 150, 350])

    def resid(p):
        return simulate_no_inhib(time_all, ic, p) - data_all

    def sse(p):
        r = resid(p)
        return np.sum(r * r)

    # PSO for a good start
    p0, _ = pso(sse, lb, ub)
    # Refine with LS (returns residual vector)
    result = least_squares(resid, p0, bounds=(lb, ub))
    best_p = result.x

    # Fine curve for plotting
    t_fine = np.linspace(0, 15, 500)
    fit_fine = simulate_no_inhib(t_fine, ic, best_p)
    # Predictions at data points for RSS/BIC
    pred_at_data = simulate_no_inhib(time_all, ic, best_p)

    return t_fine, fit_fine, best_p, pred_at_data

# ------------------ Reversible inhibition model ------------------

def simulate_inhib(tspan, ic, p):
    """
    Reversible product inhibition via ESP.
    States: y = [E, S, ES, ESP, P]
    Params p = [k1, k2, k3, k4, k5]
    Returns P(t) in PERCENT.
    """
    k1, k2, k3, k4, k5 = p

    def rhs(y, t, p):
        E, S, ES, ESP, P = y
        dE   = -k1*E*S + (k2 + k3)*ES
        dS   = -k1*E*S + k2*ES
        dES  =  k1*E*S - (k2 + k3)*ES - k4*ES*P + k5*ESP
        dESP =  k4*ES*P - k5*ESP
        dP   =  k3*ES - k4*ES*P + k5*ESP
        return [dE, dS, dES, dESP, dP]

    sol = odeint(rhs, ic, tspan, args=(p,))
    P = sol[:, 4]
    return P * 100.0  # percent

def joint_model_output(p, time1, ic1, time2, ic2):
    y1 = simulate_inhib(time1, ic1, p)
    y2 = simulate_inhib(time2, ic2, p)
    return np.concatenate([y1, y2])

def fit_reversible_inhibition_model():
    # Phase 1 (before solution change)
    time1 = np.array([0.0, 1, 3, 6], dtype=float)
    data1 = np.array([0, 11.3, 12.4, 14], dtype=float)
    ic1 = [0.03, 0.3, 0.0, 0.0, 0.0]  # [E,S,ES,ESP,P%→0]

    # Phase 2 (after solution change)
    time2 = np.array([7, 8, 12, 14, 15], dtype=float)
    data2 = np.array([14.4, 24.1, 26.6, 28.1, 28.3], dtype=float)
    ic2 = [0.03, 0.3, 0.07, 0.0, 0.144]

    # Bounds for [k1, k2, k3, k4, k5]
    lb = np.array([0, 0, 0, 0, 0])
    ub = np.array([200, 150, 350, 100.0, 30.0])

    data_all = np.concatenate([data1, data2])

    def resid(p):
        return joint_model_output(p, time1, ic1, time2, ic2) - data_all

    def sse(p):
        r = resid(p)
        return np.sum(r * r)

    # PSO seed then LS refine
    p0, _ = pso(sse, lb, ub)
    result = least_squares(resid, p0, bounds=(lb, ub))
    best_p = result.x

    # Fine curves for plotting
    t_fine1 = np.linspace(0, 7, 250)
    t_fine2 = np.linspace(7, 15, 250)
    fit1 = simulate_inhib(t_fine1, ic1, best_p)
    fit2 = simulate_inhib(t_fine2, ic2, best_p)
    fit_combined = np.concatenate([fit1, fit2])

    # Predictions at actual data points (for RSS/BIC)
    pred_at_data = joint_model_output(best_p, time1, ic1, time2, ic2)

    return (t_fine1, t_fine2, fit1, fit2, fit_combined, best_p, pred_at_data,
            time1, data1, time2, data2)

# ------------------ Plotting ------------------

def plot_no_inhibition_model(time_all, data_all, errors, t_fine, fit_no_inhib, bic_no_inhib):
    plt.figure(figsize=(10, 6))
    plt.errorbar(time_all, data_all, yerr=errors, fmt='ko', markersize=10,
                 markerfacecolor='k', capsize=5, label='PLA-Proteinase K data')
    plt.plot(t_fine, fit_no_inhib, 'b-', linewidth=5,
             label=f'No Inhibition Model Fit (BIC: {bic_no_inhib:.1f})')
    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Relative weight loss (%)', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.annotate('', xy=(9, 1.5), xytext=(7, 14.4),
                 arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
                 annotation_clip=False)
    plt.text(9.5, 2.0, 'Change of Solution', color='blue', fontsize=16,
             fontweight='bold', ha='center', va='top')
    legend = plt.legend(loc='upper left', fontsize=12)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.ylim([0, 35])
    plt.xlim([0, 16])
    plt.grid(True)
    plt.savefig(os.path.join(get_publish_folder(), 'no_inhibition_model_fit.png'), dpi=1200)
    print('✅ No Inhibition Model fit plot saved!')

def plot_reversible_inhibition_model(time_all, data_all, errors, t_fine1, t_fine2, fit1, fit2, bic_inhib):
    plt.figure(figsize=(10, 6))
    plt.errorbar(time_all, data_all, yerr=errors, fmt='ko', markersize=10,
                 markerfacecolor='k', capsize=5, label='PLA-Proteinase K data')
    plt.plot(t_fine1, fit1, 'r-', linewidth=5,
             label=f'Reversible Inhibition Model Fit (BIC: {bic_inhib:.1f})')
    plt.plot(t_fine2, fit2, 'r-', linewidth=5)
    plt.annotate('', xy=(9, 1.5), xytext=(7, 14.4),
                 arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
                 annotation_clip=False)
    plt.text(9.5, 2.0, 'Change of Solution', color='blue', fontsize=16,
             fontweight='bold', ha='center', va='top')
    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Relative weight loss (%)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    legend = plt.legend(loc='upper left', fontsize=12)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.ylim([0, 35])
    plt.xlim([0, 16])
    plt.grid(True)
    plt.savefig(os.path.join(get_publish_folder(), 'reversible_inhibition_model_fit.png'), dpi=1200)
    print('✅ Reversible Inhibition Model fit plot saved!')

def plot_comparison(time_all, data_all, errors, t_fine, fit_no_inhib, t_fine1, t_fine2, fit1, fit2,
                    bic_no_inhib, bic_inhib):
    plt.figure(figsize=(10, 6))
    plt.errorbar(time_all, data_all, yerr=errors, fmt='ko', markersize=10,
                 markerfacecolor='k', capsize=5, label='PLA-Proteinase K data')
    plt.plot(t_fine, fit_no_inhib, 'b-', linewidth=5,
             label=f'No Inhibition Model (BIC: {bic_no_inhib:.1f})')
    plt.plot(t_fine1, fit1, 'r-', linewidth=5,
             label=f'Reversible Inhibition Model (BIC: {bic_inhib:.1f})')
    plt.plot(t_fine2, fit2, 'r-', linewidth=5)
    plt.annotate('', xy=(9, 1.5), xytext=(7, 14.4),
                 arrowprops=dict(arrowstyle='-|>', color='green', lw=2),
                 annotation_clip=False)
    plt.text(9.5, 2.0, 'Change of Solution', color='blue', fontsize=16,
             fontweight='bold', ha='center', va='top')
    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Relative weight loss (%)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    legend = plt.legend(loc='upper left', fontsize=12)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.ylim([0, 35])
    plt.xlim([0, 16])
    plt.grid(True)
    plt.savefig(os.path.join(get_publish_folder(), 'model_comparison_plot.png'), dpi=1200)
    print('✅ Model comparison plot saved!')

def plot_residuals_inhib(time_all, residuals_inhib):

    plt.figure(figsize=(10, 6))
    

    plt.plot(time_all, residuals_inhib, 'go',markersize=10, 
             label='Residuals')

    plt.xlabel('Time (days)', fontsize=20)
    plt.ylabel('Residual (%)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([-5, 5])  # requested y-limits
    legend = plt.legend(loc='upper left', fontsize=12)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.grid(True)
    plt.xlim([0, 16])
    plt.savefig(os.path.join(get_publish_folder(),
                             'residuals_reversible_inhib_over_time.png'),
                dpi=1200)
    print('✅ Residuals (reversible inhibition) plot saved!')

# ------------------ Driver ------------------

def compare_models():
    # Experimental data
    time_all = np.array([0.0, 1, 3, 6, 7, 8, 12, 14, 15], dtype=float)
    data_all = np.array([0, 11.3, 12.4, 14, 14.4, 24.1, 26.6, 28.1, 28.3], dtype=float)
    error1 = np.array([0.0, 0.2, 0.9, 1.1], dtype=float)
    error2 = np.array([1.0, 1.2, 1.3, 1.0, 0.7], dtype=float)
    errors = np.concatenate([error1, error2])

    # --- No inhibition ---
    t_fine_no, fit_no_inhib, best_p_no, pred_no_at_data = fit_no_inhibition_model(time_all, data_all)
    rss_no = np.sum((pred_no_at_data - data_all) ** 2)
    bic_no = calculate_bic(len(data_all), rss_no, num_params=3)

    print('Fitted parameters for No Inhibition Model:')
    print(f'k1 = {best_p_no[0]:.6g}')
    print(f'k2 = {best_p_no[1]:.6g}')
    print(f'k3 = {best_p_no[2]:.6g}')
    print(f'RSS = {rss_no:.4f}')
    print(f'BIC = {bic_no:.4f}\n')

    # --- Reversible inhibition ---
    (t_fine1, t_fine2, fit1, fit2, fit_inhib_combined,
     best_p_inhib, pred_inhib_at_data, time1, data1, time2, data2) = fit_reversible_inhibition_model()

    rss_inhib = np.sum((pred_inhib_at_data - np.concatenate([data1, data2])) ** 2)
    bic_inhib = calculate_bic(len(data_all), rss_inhib, num_params=5)

    print('Fitted parameters for Reversible Inhibition Model:')
    print(f'k1 = {best_p_inhib[0]:.6g}')
    print(f'k2 = {best_p_inhib[1]:.6g}')
    print(f'k3 = {best_p_inhib[2]:.6g}')
    print(f'k4 = {best_p_inhib[3]:.6g}')
    print(f'k5 = {best_p_inhib[4]:.6g}')
    print(f'RSS = {rss_inhib:.4f}')
    print(f'BIC = {bic_inhib:.4f}\n')

    # Raw residuals for reversible inhibition: Observed - Predicted
    residuals_inhib = data_all - pred_inhib_at_data

    # Plots
    plot_no_inhibition_model(time_all, data_all, errors, t_fine_no, fit_no_inhib, bic_no)
    plot_reversible_inhibition_model(time_all, data_all, errors, t_fine1, t_fine2, fit1, fit2, bic_inhib)
    plot_comparison(time_all, data_all, errors, t_fine_no, fit_no_inhib,
                    t_fine1, t_fine2, fit1, fit2, bic_no, bic_inhib)
    plot_residuals_inhib(time_all, residuals_inhib)

if __name__ == "__main__":
    compare_models()
