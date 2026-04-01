# Modeling Product Inhibition in Enzymatic Degradation of Poly(lactic acid)

This repository contains the code accompanying the published paper:

> **Nansak, N., Creedon, L., O'Mahoney, D., Ghosh, R., & McAfee, M. (2026).**
> *Modeling product inhibition in enzymatic degradation of Polylactic Acid (PLA).*
> Polymer Testing, 154, 109074.
> https://doi.org/10.1016/j.polymertesting.2025.109074

---

## Overview

This work presents a mechanistic kinetic model of enzymatic polymer degradation that extends classical Michaelis–Menten kinetics to include **reversible uncompetitive product inhibition**. The model captures the experimentally observed biphasic mass-loss behaviour of PLA degraded by Proteinase K, where accumulating acidic degradation products inhibit enzyme activity and slow further erosion.

The reaction scheme is:

```
E + S  ⇌  ES  →  E + P       (binding and catalysis)
ES + P  ⇌  ESP               (reversible product inhibition)
```

governed by five kinetic rate constants: `k1` (association), `k2` (dissociation), `k3` (catalysis), `k4` (inhibition), and `k5` (inhibition recovery).

---

## Repository Structure

| File | Language | Description |
|------|----------|-------------|
| `reversible_and_no_inhibition_parameter_estimation.py` | Python | PSO + least-squares parameter estimation for both the reversible inhibition and no-inhibition (standard Michaelis–Menten) models; BIC-based model comparison |
| `Model_validation_based_on_Cui_data.py` | Python | Forward simulation and validation of the calibrated inhibition model against the independent Cui et al. (2022) dataset under two buffer-change protocols (no change and daily change) |
| `Uncertainty_Analysis_for_inhibition_model.m` | MATLAB | Bayesian parameter uncertainty analysis for the inhibition model calibrated on the Hegyesi et al. (2019) data; uses DE-MCz MCMC sampling; produces posterior credible intervals, prediction uncertainty bands, and pairwise posterior plots |
| `Unvertainty_Analysis_for_Cui_data.m` | MATLAB | Bayesian parameter uncertainty analysis using the Cui et al. (2022) validation datasets (no-buffer-change and daily-buffer-change simultaneously); DE-MCz sampling in log-parameter unit-cube space |
| `sobol_sensitivity_analysis.m` | MATLAB | Time-dependent Sobol global sensitivity analysis (first-order and total-order indices) for the PDE-based weight-loss model, identifying the dominant kinetic parameters across the degradation timeline |

---

## Experimental Datasets

### Calibration — Hegyesi et al. (2019)
> Hegyesi, N., Zhang, Y., Kohári, A., Polyák, P., Sui, X., & Pukánszky, B. (2019). Enzymatic degradation of PLA/cellulose nanocrystal composites. *Industrial Crops and Products*, 141, 111799.

- PLA plates (neat, 0% CNC) degraded by Proteinase K in tris buffer at 37 °C
- Buffer refreshed at day 7 to restore enzyme activity
- 9 relative mass-loss measurements over 15 days (4 before refresh, 5 after)
- All five kinetic parameters (`k1`–`k5`) were identified from this dataset

### Validation — Cui et al. (2022)
> Cui, L., Wang, X., Szarka, G., Hegyesi, N., Wang, Y., Sui, X., & Pukánszky, B. (2022). Quantitative analysis of factors determining the enzymatic degradation of poly(lactic acid). *International Journal of Biological Macromolecules*, 209, 1703–1709.

- PLA films degraded by Proteinase K under two protocols:
  - **No buffer change**: lactic acid accumulates, pH drops below 5, degradation plateaus at ~12% mass loss
  - **Daily buffer change**: fresh enzyme restored each day, nearly linear degradation rate
- Parameters from Hegyesi et al. were applied **without any re-fitting**

---

## Methods

### Parameter Estimation
A two-step optimisation is used: global search with **Particle Swarm Optimisation (PSO)** followed by local refinement with the **trust-region reflective least-squares** method (SciPy). Model selection uses **Residual Sum of Squares (RSS)** and the **Bayesian Information Criterion (BIC)**, which penalises model complexity.

### Bayesian Uncertainty Analysis
Parameter uncertainty follows the framework of:

> **van Mourik, S., Ter Braak, C., Stigter, H., & Molenaar, J. (2014).** Prediction uncertainty assessment of a systems biology model requires a sample of the full probability distribution of its parameters. *PeerJ*, 2, e433.

MCMC sampling is performed using the **DE-MCz** (Differential Evolution Markov Chain with snooker updates) algorithm. Parameters are sampled in log-space over bounds `k_i ∈ [10⁻³, 10³]` day⁻¹ with a log-uniform prior. 2000 posterior draws are used to construct 95% credible intervals for parameters and propagate prediction uncertainty through the model.

### Global Sensitivity Analysis
Time-dependent **Sobol variance-based sensitivity indices** (first-order and total-order) are computed following:
- Saltelli et al. (2010) estimator for first-order indices
- Jansen (1999) estimator for total-order indices

using scrambled Sobol quasi-random sequences. The analysis identifies `k3` (catalytic rate) and `k4` (inhibition rate) as the dominant controls on degradation dynamics throughout the experiment.

---

## Dependencies

### Python
```
numpy
scipy
matplotlib
pyswarm
```

Install with:
```bash
pip install numpy scipy matplotlib pyswarm
```

### MATLAB
- MATLAB R2019b or later recommended
- The `DEMCz` sampler is implemented directly within the MATLAB scripts (no additional toolboxes required beyond the standard ODE solvers `ode45`, `ode15s`)
- `sobolset` and `scramble` require the **Statistics and Machine Learning Toolbox**

---

## Usage

### Python: Parameter Estimation
```bash
python "reversible_and_no_inhibition_parameter_estimation.py"
```
Fits both models to the Hegyesi et al. (2019) data, prints RSS and BIC, and saves comparison plots to your `Downloads` folder.

### Python: Model Validation
```bash
python "Model_validation_based_on_Cui_data.py"
```
Runs the calibrated inhibition model forward on both Cui et al. (2022) protocols and saves a combined validation plot.

### MATLAB: Uncertainty Analysis (Hegyesi data)
```matlab
correcthegyesiDimensionalboxplot()
```
Runs DE-MCz MCMC on the Hegyesi et al. (2019) data and saves posterior CI, prediction uncertainty, and pairwise posterior plots to `Downloads`.

### MATLAB: Uncertainty Analysis (Cui data)
```matlab
CuiDimensionalboxplotttgh()
```
Runs DE-MCz MCMC on both Cui et al. (2022) protocols simultaneously and saves corresponding uncertainty plots to `Downloads`.

### MATLAB: Sobol Sensitivity Analysis
```matlab
correctsoboldimensional11()
```
Computes first-order and total-order Sobol indices for the PDE model over a time grid of `[0, 4, 8, ..., 72]` hours and saves index plots and CSV files to the working directory.

> **Note on MCMC run length**: The `nit` parameter in the MATLAB uncertainty scripts is set to 250 iterations for fast testing. For publication-quality results, increase `nit` to 1500 or more.

---

## Key Results

| Model | RSS | BIC |
|-------|-----|-----|
| No-Inhibition (Michaelis–Menten) | 111.0 | 29.2 |
| Reversible Inhibition | 3.1 | 1.5 |

The reversible inhibition model substantially outperforms the no-inhibition model on both criteria. All five kinetic parameters were identifiable within a single order of magnitude from the Hegyesi et al. (2019) data alone.

---

## Citation

If you use this code in your own work, please cite:

```bibtex
@article{nansak2026modeling,
  title   = {Modeling product inhibition in enzymatic degradation of Polylactic Acid ({PLA})},
  author  = {Nansak, Nanshin and Creedon, Leo and O'Mahoney, Denis and Ghosh, Ramen and McAfee, Marion},
  journal = {Polymer Testing},
  volume  = {154},
  pages   = {109074},
  year    = {2026},
  doi     = {10.1016/j.polymertesting.2025.109074}
}
```

---

## Acknowledgements

This research was supported by Atlantic Technological University through the Postgraduate Research Training Programme in Modelling and Computation for Health and Society (MOCHAS). Support from Research Ireland (grant 21/FFP-A/9152) is also acknowledged.

---

## Licence

This code is released under the [MIT Licence](LICENSE).
The associated article is open access under a [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/).
