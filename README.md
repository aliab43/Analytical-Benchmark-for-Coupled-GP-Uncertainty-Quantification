# Analytical Benchmark for Coupled GP Uncertainty Quantification

This benchmark validates the uncertainty propagation strategy used in the coupled surrogate workflow by comparing two sampling methods on a controlled analytical test case.

This folder is intentionally standalone: it can be copied into its own GitHub repository without any dependency on `FSI_Coupling3/`.

The benchmark is implemented in:

`analytical_benchmark/Analytical_Benchmark_Coupled_GP_Validation.py`

This analytical benchmark runs out‑of‑the‑box and does not require the large `.npy` or covariance matrix assets used by the full fuel‑assembly case.

## Goal

Evaluate whether an efficient approximation for coupled‑GP uncertainty (Method 3) matches the reference sequential conditional sampling (Method 2) for a fixed‑point coupled system.

The coupled system is defined by two functions `f1(x)` and `f2(x)` and a fixed‑point equation:

`y = 0.5 * (f1(y) + f2(y))`

The code builds GP surrogates for `f1` and `f2`, then compares the distribution of the coupled solution produced by two methods.

## Methods Compared

Method 2: Sequential conditional sampling
- At each fixed‑point iteration, sample from the GP posterior **conditioned on all previously sampled points**.
- This is statistically rigorous but computationally heavy.

Method 3: Constant‑offset approximation
- Compute the deterministic mean path first.
- Sample the joint GP posterior **only once** on that path.
- Use sampled offsets as constants during the fixed‑point iterations.
- This is much faster and is the proposed approximation.

## What the Script Produces

The script outputs both numerical comparisons and plots:

- Distributions of the coupled solution for Method 2 vs Method 3.
- Surrogate plots for different DOE sizes.
- Comparison statistics:
  - Mean difference
  - Kolmogorov–Smirnov test
  - Welch t‑test

Generated figures:

- `analytical_benchmark/figures/surrogates_smallDOE.png`
- `analytical_benchmark/figures/surrogates_largeDOE.png`
- `analytical_benchmark/figures/surrogates_overview.png`
- `analytical_benchmark/figures/benchmark_validation_overview.png`

## How to Run

From the project root:

```bash
python3 -m pip install -r analytical_benchmark/requirements.txt
python3 analytical_benchmark/Analytical_Benchmark_Coupled_GP_Validation.py
```

If you run on a cluster or a read-only home directory, use a writable Matplotlib cache:

```bash
env MPLCONFIGDIR=/tmp/mpl python3 analytical_benchmark/Analytical_Benchmark_Coupled_GP_Validation.py
```

Key parameters are defined at the top of the file:

- `FP_TOL`: fixed‑point tolerance
- `MAX_IT`: maximum iterations
- `MC_SAMPLES`: number of Monte Carlo samples

You can adjust DOE size and sampling strategy in the `configs` list.

## Interpretation of Results

- If Method 3 matches Method 2 in mean and distribution (small KS statistic, non‑significant p‑values), the approximation is validated for this test case.
- Large divergence indicates that sequential conditioning is necessary for accuracy.

This benchmark supports the main project by showing when the faster approximation is acceptable for uncertainty propagation in coupled surrogate systems.

## Notes

- The GP models use fixed Matérn kernels with no hyperparameter optimization for stability.
- A tiny numerical jitter is added for Cholesky stability.

If you want to extend the benchmark to multi‑dimensional inputs or alternative kernels, start by modifying the `f1_true`, `f2_true`, and `build_surrogates` sections.

## Data Availability Note

The full fuel‑assembly simulation depends on large covariance matrices and `.npy` datasets for the GP surrogates, which are not included in this Git repository due to size. This repository focuses on the code and development workflow; the missing surrogate assets must be provided separately.

## Copyright

Copyright (c) 2026 Ali Abboud.  
Contact: ali.ib.abboud95@gmail.com, ali.abboud@polytechnique.edu
