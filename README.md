# Analytical Benchmarks for Coupled GP Uncertainty Quantification

This repository contains analytical benchmark examples for validating uncertainty
propagation through coupled Gaussian-process (GP) surrogate models.

The main benchmark is now the multidimensional case in `3D_benchmark/`. It was
designed to match the theoretical framework of the associated paper: vector
outputs, structured multi-output GP covariance, a fixed-point coupling map, and
a comparison between a rigorous trajectory-conditioned sampling method and a
practical fixed-path approximation.

## Repository Contents

```text
.
|-- Analytical_Benchmark_Coupled_GP_Validation.py
|-- figures/
|-- requirements.txt
|-- 3D_benchmark/
|   |-- Analytical_Benchmark_Coupled_GP_Validation_3D.py
|   |-- README_3D_Benchmark.md
|   |-- figures_multidimensional_validation/
|       |-- multidim_surrogate_quality_table.tex
|       |-- multidim_surrogate_quality_table.csv
|       |-- multidim_method_validation.png
|       |-- multidim_3d_output_cloud.png
```

The root-level script is the original scalar benchmark. The `3D_benchmark/`
folder contains the multidimensional benchmark used for the paper validation.

## Main 3D Benchmark

The multidimensional benchmark considers two deterministic vector-valued
functions

$$
g^{(1)},g^{(2)}:[0,1]^3\to\mathbb{R}^3,
$$

and a coupled fixed point

$$
\mathbf{y}^\star
=
\frac12\left(g^{(1)}(\mathbf{y}^\star)+g^{(2)}(\mathbf{y}^\star)\right).
$$

The example is written in the general composition form

$$
\mathcal{T}=\Gamma_2\circ\Gamma_1,
$$

where

$$
\Gamma_1:\mathbb{R}^3\to\mathbb{R}^6,
\qquad
\Gamma_1(\mathbf{x})
=
\left(x_1,x_2,x_3,g^{(1)}_1(\mathbf{x}),g^{(1)}_2(\mathbf{x}),g^{(1)}_3(\mathbf{x})\right),
$$

and

$$
\Gamma_2:\mathbb{R}^6\to\mathbb{R}^3,
\qquad
\Gamma_{2,j}(\mathbf{y})
=
\frac{y_{3+j}}{2}
+
\frac12 g^{(2)}_j\left((y_\ell)_{\ell=1}^3\right),
\quad j=1,2,3.
$$

The deterministic reference fixed point is

$$
\mathbf{y}^\star=(0.385949,\;0.386257,\;0.383437)^\top.
$$

## Structured GP Surrogates

The GP surrogates used in the 3D benchmark have the specific structured form
required by the coupling framework.

For \(f_1:\mathbb{R}^3\to\mathbb{R}^6\),

$$
m_{f_1}(\mathbf{x})=(x_1,x_2,x_3,0,0,0),
$$

and

$$
\mathrm{Cov}\left(f_1(\mathbf{x}),f_1(\mathbf{x}')\right)
=
\mathrm{diag}\left(
0,0,0,
k(\mathbf{x},\mathbf{x}'),
k(\mathbf{x},\mathbf{x}'),
k(\mathbf{x},\mathbf{x}')
\right).
$$

For \(f_2:\mathbb{R}^6\to\mathbb{R}^3\),

$$
m_{f_2}(\mathbf{y})
=
\left(\frac{y_4}{2},\frac{y_5}{2},\frac{y_6}{2}\right),
$$

and

$$
\mathrm{Cov}\left(f_2(\mathbf{y}),f_2(\mathbf{y}')\right)
=
\mathrm{diag}\left(
k(\tilde{\mathbf{y}},\tilde{\mathbf{y}}'),
k(\tilde{\mathbf{y}},\tilde{\mathbf{y}}'),
k(\tilde{\mathbf{y}},\tilde{\mathbf{y}}')
\right),
\qquad
\tilde{\mathbf{y}}=(y_1,y_2,y_3).
$$

The scalar kernel \(k\) is a fixed Matern \(5/2\) kernel.

## Methods Compared

The benchmark compares two uncertainty propagation strategies.

**Method 2: rigorous trajectory-conditioned sampling.**
For each Monte Carlo replication, the GP is sampled sequentially along the
random fixed-point trajectory, conditioning each new query on all previously
sampled values from the same trajectory.

**Method 3: fixed mean-path constant-offset approximation.**
The deterministic GP-mean fixed-point path is computed once. Then GP offsets are
sampled on this fixed path and reused inside the Monte Carlo coupling iteration.

The purpose of the benchmark is to show that Method 3 becomes a good
approximation of Method 2 when the DOE is sufficiently dense.

## How to Run

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the multidimensional benchmark:

```bash
python 3D_benchmark/Analytical_Benchmark_Coupled_GP_Validation_3D.py
```

Run the original scalar benchmark:

```bash
python Analytical_Benchmark_Coupled_GP_Validation.py
```

## Outputs

The 3D benchmark writes its outputs to:

```text
3D_benchmark/figures_multidimensional_validation/
```

The main outputs are:

- `multidim_surrogate_quality_table.tex`: LaTeX table of metamodel quality;
- `multidim_surrogate_quality_table.csv`: same table in CSV format;
- `multidim_method_validation.png`: histograms of the coupled output
  components and Euclidean norm;
- `multidim_3d_output_cloud.png`: complementary 3D cloud of Monte Carlo
  coupled outputs.

## Summary of the Numerical Message

For a poor design (`smallDOE`, \(n=20\)), the rigorous and proposed methods show
visible distributional differences because the stochastic trajectories can move
away from the deterministic GP-mean path.

For a denser design (`largeDOE`, \(n=500\)), the posterior uncertainty decreases,
the stochastic trajectories remain close to the deterministic path, and Method 3
becomes statistically close to Method 2 while remaining computationally cheaper.

## Citation and Contact

If you use this benchmark, please cite the associated paper or contact the
author for the appropriate reference.

Copyright (c) 2026 Ali Abboud.  
Contact: ali.ib.abboud95@gmail.com, ali.abboud@polytechnique.edu
