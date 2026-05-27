# 3D Analytical Benchmark for Coupled GP Uncertainty Quantification

This directory contains the multidimensional benchmark implemented in
`Analytical_Benchmark_Coupled_GP_Validation_3D.py`.

The benchmark is a controlled three-dimensional example designed to validate
uncertainty propagation through coupled Gaussian-process (GP) surrogate models.
It compares:

- **Method 2:** rigorous trajectory-conditioned GP sampling;
- **Method 3:** fixed mean-path constant-offset approximation.

The benchmark is intentionally analytical: the exact deterministic functions and
the deterministic reference fixed point are known, which makes the comparison
between the two stochastic propagation methods transparent.

## 1. Analytical Coupled Problem

Let

$$
\mathbf{x}=(x_1,x_2,x_3)\in[0,1]^3.
$$

The benchmark uses two deterministic vector-valued functions

$$
g^{(1)},g^{(2)}:[0,1]^3\to\mathbb{R}^3.
$$

They are defined by

$$
g^{(1)}(\mathbf{x})=
\begin{pmatrix}
0.30+0.08\cos(\pi x_1)+0.06x_2^2+0.04x_3+0.03x_1x_2\\
0.34+0.07\sin(\pi x_2)+0.05x_3^2-0.04x_1x_3+0.02x_1\\
0.32+0.06\cos(\pi x_3)+0.05x_1^2+0.03x_2+0.02x_2x_3
\end{pmatrix},
$$

and

$$
g^{(2)}(\mathbf{x})=
\begin{pmatrix}
0.36+0.07\sin(\pi x_1)-0.05x_2+0.04x_3^2+0.02x_1x_3\\
0.31+0.08\cos(\pi x_2)+0.05x_1^2+0.04x_3-0.02x_1x_2\\
0.35+0.06\sin(\pi x_3)+0.04x_2^2-0.03x_1+0.02x_1x_2
\end{pmatrix}.
$$

The coupled reference solution is the fixed point

$$
\mathbf{y}^\star
=
\frac12
\left(
g^{(1)}(\mathbf{y}^\star)
+
g^{(2)}(\mathbf{y}^\star)
\right).
$$

The deterministic fixed-point iteration gives

$$
\mathbf{y}^\star=(0.385949,\;0.386257,\;0.383437)^\top.
$$

## 2. Embedding in the General Coupling Framework

The example is written as a composition

$$
\mathcal{T}=\Gamma_2\circ\Gamma_1,
$$

so that it fits the theoretical framework used for coupled surrogate models.

Define

$$
\Gamma_1:\mathbb{R}^3\to\mathbb{R}^6
$$

by

$$
\Gamma_1(\mathbf{x})
=
\left(
x_1,x_2,x_3,
g^{(1)}_1(\mathbf{x}),
g^{(1)}_2(\mathbf{x}),
g^{(1)}_3(\mathbf{x})
\right).
$$

Equivalently,

$$
\Gamma_{1,j}(\mathbf{x})=x_j,\qquad j=1,2,3,
$$

and

$$
\Gamma_{1,j}(\mathbf{x})=g^{(1)}_{j-3}(\mathbf{x}),\qquad j=4,5,6.
$$

Define

$$
\Gamma_2:\mathbb{R}^6\to\mathbb{R}^3
$$

by

$$
\Gamma_{2,j}(\mathbf{y})
=
\frac{y_{3+j}}{2}
+
\frac12 g^{(2)}_j\left((y_\ell)_{\ell=1}^3\right),
\qquad j=1,2,3.
$$

Then

$$
\mathcal{T}(\mathbf{x})
=
\Gamma_2(\Gamma_1(\mathbf{x}))
=
\frac12
\left(
g^{(1)}(\mathbf{x})
+
g^{(2)}(\mathbf{x})
\right),
$$

and the benchmark solves

$$
\mathbf{y}^\star\in\mathrm{Fix}(\mathcal{T}).
$$

The code checks numerically that

$$
\mathcal{T}([0,1]^3)\subset[0,1]^3,
$$

and estimates the contraction modulus as

$$
\rho\simeq 0.2079<1.
$$

## 3. Structured GP Surrogates

The GP surrogates have the structured form required by the coupling.

### GP for the first block

For

$$
f_1:\mathbb{R}^3\to\mathbb{R}^6,
$$

the prior mean is

$$
m_{f_1}(\mathbf{x})
=
(x_1,x_2,x_3,0,0,0),
$$

and the covariance is

$$
\mathrm{Cov}
\left(
f_1(\mathbf{x}),f_1(\mathbf{x}')
\right)
=
\mathrm{diag}
\left(
0,0,0,
k(\mathbf{x},\mathbf{x}'),
k(\mathbf{x},\mathbf{x}'),
k(\mathbf{x},\mathbf{x}')
\right).
$$

Only the last three components are uncertain; the first three are deterministic
identity components.

### GP for the second block

For

$$
f_2:\mathbb{R}^6\to\mathbb{R}^3,
$$

the prior mean is

$$
m_{f_2}(\mathbf{y})
=
\left(
\frac{y_4}{2},
\frac{y_5}{2},
\frac{y_6}{2}
\right),
$$

and the covariance is

$$
\mathrm{Cov}
\left(
f_2(\mathbf{y}),f_2(\mathbf{y}')
\right)
=
\mathrm{diag}
\left(
k(\tilde{\mathbf{y}},\tilde{\mathbf{y}}'),
k(\tilde{\mathbf{y}},\tilde{\mathbf{y}}'),
k(\tilde{\mathbf{y}},\tilde{\mathbf{y}}')
\right),
$$

where

$$
\tilde{\mathbf{y}}=(y_1,y_2,y_3).
$$

The scalar kernel \(k\) is a fixed Matern \(5/2\) kernel with length-scale

$$
\ell=(0.80,0.80,0.80).
$$

A small nugget

$$
\alpha=10^{-12}
$$

is used only for numerical stability.

## 4. Numerical Configuration

The script uses:

- smallDOE: \(n=20\) training points per function;
- largeDOE: \(n=500\) training points per function;
- \(N=500\) Monte Carlo replications;
- fixed-point tolerance \(10^{-6}\);
- maximum iteration count \(M_{\max}=200\);
- Latin hypercube sampling on \([0,1]^3\);
- random seed `42`.

## 5. Outputs

Running the script creates:

```text
figures_multidimensional_validation/
```

with:

- `multidim_surrogate_quality_table.tex`;
- `multidim_surrogate_quality_table.csv`;
- `multidim_method_validation.png`;
- `multidim_3d_output_cloud.png`.

The table reports:

- \(Q^2\), the predictive coefficient on an independent validation set;
- \(\overline{\sigma}\), the average posterior standard deviation;
- \(\overline{w}_{95}\), the average width of the \(95\%\) credible interval;
- \(\widehat{C}_{95}\), the empirical coverage probability of that interval.

The main validation figure reports separate histograms for:

$$
y_1,\qquad y_2,\qquad y_3,\qquad ||\mathbf{y}||_2.
$$

The 3D cloud is a complementary visualization of the Monte Carlo vector outputs
\(\mathbf{y}^{(j)}=(y_1^{(j)},y_2^{(j)},y_3^{(j)})\).

## 6. How to Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python 3D_benchmark/Analytical_Benchmark_Coupled_GP_Validation_3D.py
```

Alternatively, from this directory:

```bash
python Analytical_Benchmark_Coupled_GP_Validation_3D.py
```

## 7. Interpretation

For the small design, the rigorous trajectory-conditioned method produces a
wider coupled-output distribution and the Kolmogorov--Smirnov tests detect
distributional differences between Method 2 and Method 3.

For the dense design, posterior uncertainty decreases, the stochastic
trajectories remain close to the deterministic GP-mean path, and Method 3
becomes statistically close to Method 2. This supports the fixed-path
constant-offset approximation when the surrogate design is sufficiently
informative.
