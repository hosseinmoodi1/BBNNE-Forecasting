# BBNNE: Bat-Based Boosted Neural Network Ensemble with RLSG

Reference implementation of the **BBNNE** model for multi-step stock
price time series prediction, accompanying the manuscript:

> *Developing a Bat-Based Boosted Neural Network Ensemble Model for
> Stock Price Time Series Prediction.*

BBNNE combines three complementary ideas:

1. **Relevant Lag Space Generation (RLSG)** — a novel tournament-based
   lag-selection method that produces a diverse set of input subspaces,
   one per ensemble member.
2. **Bat Algorithm** — a metaheuristic used to train each MLP expert.
3. **APE-thresholded Boosting** — a continuous, regression-oriented
   adaptation of AdaBoost in which the error threshold is set
   adaptively to the mean Absolute Percentage Error (APE) of the
   current expert, eliminating any user-supplied cutoff.

Each expert is trained on the *h*-step-ahead log-return target

$$\Delta_h(k) = \log\!\left(\frac{P(k+h)}{P(k)}\right)$$

and the final price forecast is recovered through

$$\hat{P}(k+h) = P(k)\,\exp\!\big(\gamma\,\hat{r}_h(k)\big)$$

with a shrinkage coefficient γ ∈ [0, 1].

---

## Repository Structure
bbnne/
├── bbnne/ # Core library
│ ├── init.py
│ ├── model.py # BBNNE main model
│ ├── ablation.py # Ablation-study variants
│ ├── rlsg.py # Relevant Lag Space Generation
│ ├── bat_algorithm.py # Bat Algorithm optimizer
│ ├── neural_network.py # MLP expert (single hidden layer)
│ ├── features.py # Feature engineering (rich / raw_lags)
│ ├── metrics.py # MAPE, MAE, RMSE, DA, Theil's U, R²
│ ├── benchmarks.py # MLP, RBF, Bagging, AdaBoost, LSTM
│ └── data_loader.py # Loaders for TSE and US equity CSVs
├── examples/ # Runnable scripts
│ ├── run_pipeline.py # Main pipeline: experiments → tables → figures
│ ├── run_ablation.py # Ablation study
│ ├── run_stages.py # Multi-stage hyperparameter search
│ ├── tune_parameters.py # Grid search over BBNNE hyperparameters
│ └── generate_figures.py # Standalone figure generator
├── tests/
│ └── test_components.py # Unit tests for all components
├── data/ # Benchmark datasets
│ ├── SHPETRO.csv # Tehran Stock Exchange (TSE)
│ ├── FEOLAD.csv # Tehran Stock Exchange (TSE)
│ ├── DailyData_#IBM_2026-09-14.csv # NYSE
│ └── DailyData_#MSFT_2026-09-14.csv # NASDAQ
├── results/ # Output directory (created at runtime)
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md

---

## Installation

### Requirements

- Python ≥ 3.9 (tested up to 3.14)
- pip

### Setup

```bash
git clone https://github.com/hosseinmoodi1/BBNNE-Forecasting.git
cd BBNNE-Forecasting
pip install -r requirements.txt
pip install -e .

Optional (LSTM baseline)
The LSTM benchmark requires TensorFlow, which is not installed by
default because it is incompatible with Python 3.14 at the time of
writing. To enable it:

pip install tensorflow>=2.13

If TensorFlow is not available, the LSTM benchmark is automatically
skipped and all other models run normally.

Quick Start
Run the full pipeline (Stage-6 settings)

python examples/run_pipeline.py --stage all

This executes three stages in order:

Experiments — trains BBNNE and all baselines on the four datasets
(SHPETRO, FOLAD, IBM, MSFT) and writes results/all_metrics.json,
results/table6_summary.csv, and the Wilcoxon test tables.

Tables — generates LaTeX tables (results/latex/table6.tex,
table7.tex, tables_combined.tex).

Figures — produces five publication-quality figures
(results/figures/fig1_mape_bars.png through fig5_wilcoxon_z.png).

Run only one stage

python examples/run_pipeline.py --stage experiments
python examples/run_pipeline.py --stage tables
python examples/run_pipeline.py --stage figures

Run the ablation study

python examples/run_ablation.py

This compares Full BBNNE against two ablated variants
(BBNNE w/o RLSG, BBNNE w/o Boosting) and writes
results/table_ablation.csv, results/table_ablation_wilcoxon.csv,
and results/figures/fig_ablation_box.png.

Configuration
The pipeline uses the following Stage-6 configuration, validated
through the cascade experiments in run_stages.py:

Parameter	Value
Shrinkage γ	0.5
Forecast horizon h	3
Feature mode	raw_lags
Ensemble size T	7
Subspace size τ	5
Hidden-layer width	10
Number of seeds	5
Train/test split	80/20 chronological
Bat population Np	30
Bat iterations Imax	200
α, γ (Bat)	0.9, 0.9
A₀, r₀	0.5, 0.5
Bounds	[-1, 1]
All hyperparameters can be overridden by editing the constants at the
top of examples/run_pipeline.py.

Bat Algorithm Parameters
Parameter	Symbol	Value
Population size	Np	30
Maximum iterations	Imax	200
Minimum frequency	f_min	0.0
Maximum frequency	f_max	2.0
Loudness decay factor	α	0.9
Pulse-rate increase	γ	0.9
Initial loudness	A₀	0.5
Initial pulse rate	r₀	0.5
Parameter bounds	[lb, ub]	[-1, 1]
