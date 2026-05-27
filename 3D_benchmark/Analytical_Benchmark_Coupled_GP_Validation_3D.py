from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import solve_triangular
from scipy.stats import ks_2samp, norm, ttest_ind
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


warnings.filterwarnings("ignore", category=ConvergenceWarning)

RNG = np.random.default_rng(42)

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures_multidimensional_validation"
FIGURES_DIR.mkdir(exist_ok=True)

INPUT_DIM = 3
INTERMEDIATE_DIM = 6
OUTPUT_DIM = 3

FP_TOL = 1e-6
MAX_IT = 200
MC_SAMPLES = 500
LENGTH_SCALE = np.array([0.80, 0.80, 0.80])
IMAGE_CHECK_GRID_SIZE = 35

COLOR_RIGOROUS = "#0072B2"
COLOR_PROPOSED = "#E69F00"
COLOR_TRUE = "#D55E00"
COLOR_GP_PATH = "#222222"
COMPONENT_COLORS = ["#0072B2", "#009E73", "#CC79A7"]
TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 11


def matern_kernel():
    """Scalar Matern 5/2 kernel used in the structured GP covariances."""
    return Matern(length_scale=LENGTH_SCALE, length_scale_bounds="fixed", nu=2.5)


def _as_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def make_query(u):
    """Map the 3D interface state to the 3D input of one analytical function."""
    return np.clip(np.asarray(u, dtype=float).reshape(-1), 0.0, 1.0)


def _as_vector(x, size):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != size:
        raise ValueError(f"Expected vector of size {size}, got {x.size}")
    return x


def g1_true(X):
    """Analytical function 1: g1(x1, x2, x3) = (y1, y2, y3)."""
    X = _as_2d(X)
    x0, x1, x2 = X.T
    y0 = (
        0.30
        + 0.08 * np.cos(np.pi * x0)
        + 0.06 * x1**2
        + 0.04 * x2
        + 0.03 * x0 * x1
    )
    y1 = (
        0.34
        + 0.07 * np.sin(np.pi * x1)
        + 0.05 * x2**2
        - 0.04 * x0 * x2
        + 0.02 * x0
    )
    y2 = (
        0.32
        + 0.06 * np.cos(np.pi * x2)
        + 0.05 * x0**2
        + 0.03 * x1
        + 0.02 * x1 * x2
    )
    return np.column_stack([y0, y1, y2])


def g2_true(X):
    """Analytical function 2: g2(x1, x2, x3) = (y1, y2, y3)."""
    X = _as_2d(X)
    x0, x1, x2 = X.T
    y0 = (
        0.36
        + 0.07 * np.sin(np.pi * x0)
        - 0.05 * x1
        + 0.04 * x2**2
        + 0.02 * x0 * x2
    )
    y1 = (
        0.31
        + 0.08 * np.cos(np.pi * x1)
        + 0.05 * x0**2
        + 0.04 * x2
        - 0.02 * x0 * x1
    )
    y2 = (
        0.35
        + 0.06 * np.sin(np.pi * x2)
        + 0.04 * x1**2
        - 0.03 * x0
        + 0.02 * x0 * x1
    )
    return np.column_stack([y0, y1, y2])


def gamma1_true(u):
    """Gamma_1: R^3 -> R^6, Gamma_1(u) = (u, g1(u))."""
    u = make_query(u)
    return np.r_[u, g1_true(u)[0]]


def gamma2_true(v):
    """Gamma_2: R^6 -> R^3, Gamma_2(v)_j = v_{3+j}/2 + g2_j(v_1,v_2,v_3)/2."""
    v = _as_vector(v, INTERMEDIATE_DIM)
    interface_state = make_query(v[:INPUT_DIM])
    return 0.5 * v[INPUT_DIM:] + 0.5 * g2_true(interface_state)[0]


def f1_prior_mean_structure(x):
    """Mean structure requested for f1: (x1, x2, x3, 0, 0, 0)."""
    x = make_query(x)
    return np.r_[x, np.zeros(OUTPUT_DIM)]


def f1_prior_covariance_structure(x, x_prime):
    """
    Covariance requested for f1:
    diag(0, 0, 0, k(x,x'), k(x,x'), k(x,x')).
    """
    x = make_query(x).reshape(1, -1)
    x_prime = make_query(x_prime).reshape(1, -1)
    k_value = matern_kernel()(x, x_prime)[0, 0]
    return np.diag(np.r_[np.zeros(INPUT_DIM), np.full(OUTPUT_DIM, k_value)])


def f2_prior_mean_structure(y):
    """Mean structure requested for f2: (y4/2, y5/2, y6/2)."""
    y = _as_vector(y, INTERMEDIATE_DIM)
    return 0.5 * y[INPUT_DIM:]


def f2_prior_covariance_structure(y, y_prime):
    """
    Covariance requested for f2:
    diag(k(y_1:3,y'_1:3), k(y_1:3,y'_1:3), k(y_1:3,y'_1:3)).
    """
    y = _as_vector(y, INTERMEDIATE_DIM)
    y_prime = _as_vector(y_prime, INTERMEDIATE_DIM)
    k_value = matern_kernel()(
        make_query(y[:INPUT_DIM]).reshape(1, -1),
        make_query(y_prime[:INPUT_DIM]).reshape(1, -1),
    )[0, 0]
    return np.diag(np.full(OUTPUT_DIM, k_value))


def coupled_true_map(u):
    return gamma2_true(gamma1_true(u))


def estimate_coupling_image_bounds(grid_size=IMAGE_CHECK_GRID_SIZE):
    """Numerically check that the deterministic coupling maps [0, 1]^3 into itself."""
    grid = np.linspace(0.0, 1.0, grid_size)
    mesh = np.meshgrid(grid, grid, grid, indexing="ij")
    X = np.column_stack([axis.ravel() for axis in mesh])
    T_values = np.vstack([coupled_true_map(x) for x in X])
    lower = T_values.min(axis=0)
    upper = T_values.max(axis=0)
    maps_into_unit_cube = np.all(lower >= -1e-12) and np.all(upper <= 1.0 + 1e-12)
    return lower, upper, maps_into_unit_cube


def solve_fixed_point(map_fun, x0=None, tol=1e-12, maxit=300):
    y = np.full(OUTPUT_DIM, 0.5) if x0 is None else np.asarray(x0, dtype=float)
    path = [y.copy()]
    for _ in range(maxit):
        y_new = np.asarray(map_fun(y), dtype=float)
        path.append(y_new.copy())
        if np.linalg.norm(y_new - y) < tol:
            return y_new, np.asarray(path)
        y = y_new
    return y, np.asarray(path)


def estimate_contraction(n_probe=2500, step=1e-5):
    """Finite-difference estimate of max spectral norm of the true coupling map."""
    max_norm = 0.0
    for u in RNG.uniform(0.05, 0.95, size=(n_probe, OUTPUT_DIM)):
        J = np.zeros((OUTPUT_DIM, OUTPUT_DIM))
        for k in range(OUTPUT_DIM):
            e = np.zeros(OUTPUT_DIM)
            e[k] = step
            J[:, k] = (coupled_true_map(u + e) - coupled_true_map(u - e)) / (2.0 * step)
        max_norm = max(max_norm, np.linalg.norm(J, 2))
    return max_norm


def lhs_nd(n, dim, low=0.0, high=1.0):
    cut = np.linspace(low, high, n + 1)
    X = np.empty((n, dim))
    for k in range(dim):
        X[:, k] = cut[:-1] + (cut[1:] - cut[:-1]) * RNG.random(n)
        RNG.shuffle(X[:, k])
    return X


def build_surrogates(n_train, length_scale=LENGTH_SCALE):
    """Fit the uncertain diagonal GP blocks of f1 and f2."""
    X = lhs_nd(n_train, INPUT_DIM)
    y1 = g1_true(X)
    y2_relaxation_residual = 0.5 * g2_true(X)

    kernel = Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5)
    gps1 = []
    gps2 = []
    for k in range(OUTPUT_DIM):
        gp1 = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-12,
            optimizer=None,
            normalize_y=False,
        )
        gp2 = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-12,
            optimizer=None,
            normalize_y=False,
        )
        gp1.fit(X, y1[:, k])
        gp2.fit(X, y2_relaxation_residual[:, k])
        gps1.append(gp1)
        gps2.append(gp2)
    return gps1, gps2, X, y1, g2_true(X)


def gp_mean(gps, x):
    x = np.asarray(x, dtype=float).reshape(1, -1)
    return np.array([(gp.kernel_(x, gp.X_train_) @ gp.alpha_)[0] for gp in gps])


def f1_surrogate_mean(gps1, u):
    """Posterior mean of f1: R^3 -> R^6 with deterministic identity components."""
    u = make_query(u)
    mean = f1_prior_mean_structure(u)
    mean[INPUT_DIM:] = gp_mean(gps1, u)
    return mean


def f2_surrogate_mean(gps2, v):
    """Posterior mean of f2: R^6 -> R^3 with deterministic relaxation contribution."""
    v = _as_vector(v, INTERMEDIATE_DIM)
    interface_state = make_query(v[:INPUT_DIM])
    return f2_prior_mean_structure(v) + gp_mean(gps2, interface_state)


def gp_posterior_mean_cov(gp, X):
    """Fast latent posterior mean/covariance using the fitted sklearn GP factors."""
    X = _as_2d(X)
    K_trans = gp.kernel_(X, gp.X_train_)
    mean = K_trans @ gp.alpha_
    v = np.linalg.solve(gp.L_, K_trans.T)
    cov = gp.kernel_(X) - v.T @ v
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def compute_mean_path(gps1, gps2, x0=None, tol=FP_TOL, maxit=MAX_IT):
    def mean_map(u):
        return f2_surrogate_mean(gps2, f1_surrogate_mean(gps1, u))

    _, path = solve_fixed_point(mean_map, x0=x0, tol=tol, maxit=maxit)
    return path


class ConditionalVectorGPSampler:
    """
    Sequential sampler for the uncertain blocks of f1 or f2.

    For f1, the full map is (x, sampled g1(x)); for f2, the full map is
    y_{4:6}/2 + sampled g2(y_{1:3})/2. The sampler below handles only the
    uncertain diagonal GP blocks; deterministic Gamma components are added outside.
    """

    def __init__(self, gps, jitter=1e-14):
        self.gp0 = gps[0]
        self.alpha = np.column_stack([gp.alpha_ for gp in gps])
        self.jitter = jitter
        self.X_seen = np.empty((0, INPUT_DIM))
        self.Z_seen = np.empty((0, OUTPUT_DIM))
        self.mean_seen = np.empty((0, OUTPUT_DIM))
        self.v_seen = np.empty((len(self.gp0.X_train_), 0))
        self.L_seen = np.empty((0, 0))

    def sample(self, x_new):
        x_new = np.asarray(x_new, dtype=float).reshape(1, -1)
        K_trans = self.gp0.kernel_(x_new, self.gp0.X_train_)
        mean_new = (K_trans @ self.alpha).ravel()
        v_new = solve_triangular(self.gp0.L_, K_trans.T, lower=True, check_finite=False)
        var_new = self.gp0.kernel_(x_new)[0, 0] - (v_new.T @ v_new)[0, 0]

        if len(self.Z_seen) == 0:
            cond_var = max(var_new, 0.0)
            z_new = mean_new + np.sqrt(cond_var) * RNG.standard_normal(OUTPUT_DIM)
            self.X_seen = x_new
            self.Z_seen = z_new.reshape(1, -1)
            self.mean_seen = mean_new.reshape(1, -1)
            self.v_seen = v_new
            self.L_seen = np.array([[np.sqrt(max(var_new + self.jitter, self.jitter))]])
            return z_new

        cov_new_seen = self.gp0.kernel_(x_new, self.X_seen).ravel() - (v_new.T @ self.v_seen).ravel()
        w = solve_triangular(self.L_seen, cov_new_seen, lower=True, check_finite=False)
        alpha_seen = solve_triangular(
            self.L_seen.T,
            solve_triangular(self.L_seen, self.Z_seen - self.mean_seen, lower=True, check_finite=False),
            lower=False,
            check_finite=False,
        )
        cond_mean = mean_new + cov_new_seen @ alpha_seen
        cond_var = max(var_new - w @ w, 0.0)
        z_new = cond_mean + np.sqrt(cond_var) * RNG.standard_normal(OUTPUT_DIM)

        diag = np.sqrt(max(var_new + self.jitter - w @ w, self.jitter))
        m = len(self.Z_seen)
        L_new = np.zeros((m + 1, m + 1))
        L_new[:m, :m] = self.L_seen
        L_new[m, :m] = w
        L_new[m, m] = diag

        self.X_seen = np.vstack([self.X_seen, x_new])
        self.Z_seen = np.vstack([self.Z_seen, z_new])
        self.mean_seen = np.vstack([self.mean_seen, mean_new])
        self.v_seen = np.column_stack([self.v_seen, v_new.ravel()])
        self.L_seen = L_new
        return z_new


def run_rigorous_method(gps1, gps2, x0=None, tol=FP_TOL, maxit=MAX_IT, n_samples=MC_SAMPLES):
    """Sequentially condition on every random query visited by each MC trajectory."""
    Ys = np.zeros((n_samples, OUTPUT_DIM))
    iterations = np.zeros(n_samples, dtype=int)
    path_distances = np.zeros(n_samples)

    x0 = np.full(OUTPUT_DIM, 0.5) if x0 is None else np.asarray(x0, dtype=float)
    reference_path = compute_mean_path(gps1, gps2, x0=x0, tol=tol, maxit=maxit)

    for j in range(n_samples):
        y = x0.copy()
        sampler_g1 = ConditionalVectorGPSampler(gps1)
        sampler_g2 = ConditionalVectorGPSampler(gps2)
        trajectory = [y.copy()]

        for m in range(maxit):
            q1 = make_query(y)
            g1_value = sampler_g1.sample(q1)
            f1_value = np.r_[q1, g1_value]

            q2 = make_query(f1_value[:INPUT_DIM])
            f2_random_residual = sampler_g2.sample(q2)
            y_new = 0.5 * f1_value[INPUT_DIM:] + f2_random_residual
            trajectory.append(y_new.copy())
            if np.linalg.norm(y_new - y) < tol:
                y = y_new
                break
            y = y_new

        Ys[j] = y
        iterations[j] = len(trajectory) - 1
        aligned = reference_path[: min(len(reference_path), len(trajectory))]
        path_distances[j] = np.max(
            np.linalg.norm(np.asarray(trajectory[: len(aligned)]) - aligned, axis=1)
        )

    return Ys, iterations, path_distances


def _path_queries(path):
    return np.vstack([make_query(u) for u in path[:-1]])


def _path_gamma1_mean_queries(gps1, path):
    return np.vstack([f1_surrogate_mean(gps1, u) for u in path[:-1]])


def run_proposed_method(gps1, gps2, deterministic_path, n_samples=MC_SAMPLES, tol=FP_TOL):
    """Sample GP errors once on the fixed deterministic path and reuse them by iteration."""
    X_path_1 = _path_queries(deterministic_path)
    F1_path = _path_gamma1_mean_queries(gps1, deterministic_path)
    X_path_2 = F1_path[:, :INPUT_DIM]
    path_len = len(X_path_1)

    chol_1 = []
    chol_2 = []
    for k in range(OUTPUT_DIM):
        _, cov1 = gp_posterior_mean_cov(gps1[k], X_path_1)
        _, cov2 = gp_posterior_mean_cov(gps2[k], X_path_2)
        chol_1.append(np.linalg.cholesky(cov1 + 1e-12 * np.eye(path_len)))
        chol_2.append(np.linalg.cholesky(cov2 + 1e-12 * np.eye(path_len)))

    Ys = np.zeros((n_samples, OUTPUT_DIM))
    iterations = np.zeros(n_samples, dtype=int)

    for j in range(n_samples):
        delta1 = np.column_stack([chol_1[k] @ RNG.standard_normal(path_len) for k in range(OUTPUT_DIM)])
        delta2 = np.column_stack([chol_2[k] @ RNG.standard_normal(path_len) for k in range(OUTPUT_DIM)])

        y = deterministic_path[0].copy()
        for m in range(path_len):
            q1 = make_query(y)
            g1_value = gp_mean(gps1, q1) + delta1[m]
            f1_value = np.r_[q1, g1_value]

            q2 = make_query(f1_value[:INPUT_DIM])
            f2_random_residual = gp_mean(gps2, q2) + delta2[m]
            y_new = 0.5 * f1_value[INPUT_DIM:] + f2_random_residual
            iterations[j] = m + 1
            if np.linalg.norm(y_new - y) < tol:
                y = y_new
                break
            y = y_new
        Ys[j] = y

    return Ys, iterations


def summarize_samples(samples):
    return {
        "mean": samples.mean(axis=0),
        "cov": np.cov(samples.T, ddof=1),
        "var": samples.var(axis=0, ddof=1),
        "ci_low": np.percentile(samples, 2.5, axis=0),
        "ci_high": np.percentile(samples, 97.5, axis=0),
    }


def compare_methods(Y_rigorous, Y_proposed):
    component_rows = []
    for k in range(OUTPUT_DIM):
        ks_stat, ks_p = ks_2samp(Y_rigorous[:, k], Y_proposed[:, k])
        t_stat, t_p = ttest_ind(Y_rigorous[:, k], Y_proposed[:, k], equal_var=False)
        component_rows.append((ks_stat, ks_p, t_stat, t_p))

    norm_r = np.linalg.norm(Y_rigorous, axis=1)
    norm_p = np.linalg.norm(Y_proposed, axis=1)
    ks_norm, ks_p_norm = ks_2samp(norm_r, norm_p)
    t_norm, t_p_norm = ttest_ind(norm_r, norm_p, equal_var=False)
    return {
        "mean_diff": Y_rigorous.mean(axis=0) - Y_proposed.mean(axis=0),
        "component_tests": np.asarray(component_rows),
        "norm_test": (ks_norm, ks_p_norm, t_norm, t_p_norm),
    }


def empirical_coverage_95(y_true, mean, std):
    std = np.maximum(std, 1e-14)
    z_value = norm.ppf(0.975)
    return np.mean(np.abs(y_true - mean) <= z_value * std)


def evaluate_surrogate_quality(gps1, gps2, n_eval=4000, seed=202405):
    rng = np.random.default_rng(seed)
    X_eval = rng.uniform(0.0, 1.0, size=(n_eval, INPUT_DIM))
    rows = []

    for function_name, gps, true_values, scale in [
        (r"$g^{(1)}$", gps1, g1_true(X_eval), 1.0),
        (r"$g^{(2)}$", gps2, g2_true(X_eval), 2.0),
    ]:
        for k in range(OUTPUT_DIM):
            mean, std = gps[k].predict(X_eval, return_std=True)
            mean = scale * mean
            std = scale * std
            y_true = true_values[:, k]
            ss_res = np.sum((y_true - mean) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            q2 = 1.0 - ss_res / ss_tot
            rows.append(
                {
                    "function": function_name,
                    "component": rf"$y_{k + 1}$",
                    "q2": q2,
                    "mean_sigma": std.mean(),
                    "mean_ci_width": (2.0 * 1.96 * std).mean(),
                    "coverage_95": empirical_coverage_95(y_true, mean, std),
                }
            )
    return rows


def write_surrogate_quality_table(quality_by_case):
    tex_path = FIGURES_DIR / "multidim_surrogate_quality_table.tex"
    csv_path = FIGURES_DIR / "multidim_surrogate_quality_table.csv"

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("DOE,function,component,Q2,mean_sigma,mean_95_width,coverage_95\n")
        for case_name, rows in quality_by_case:
            for row in rows:
                handle.write(
                    f"{case_name},{row['function']},{row['component']},"
                    f"{row['q2']:.6f},{row['mean_sigma']:.6e},"
                    f"{row['mean_ci_width']:.6e},{row['coverage_95']:.6f}\n"
                )

    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[ht!]\\centering\n")
        handle.write("\\caption{Quality indicators for the multidimensional GP metamodels. "
                     "$Q^2$ is computed on an independent validation set; "
                     "$\\overline{\\sigma}$ is the average posterior standard deviation; "
                     "$\\overline{w}_{95}$ is the average width of the $95\\%$ credible interval; "
                     "$\\widehat{C}_{95}$ is the empirical coverage probability of the $95\\%$ credible interval.}\n")
        handle.write("\\label{tab:surrogate_quality_3d}\n")
        handle.write("\\begin{tabular}{llrrrr}\n")
        handle.write("\\hline\n")
        handle.write("DOE / output & component & $Q^2$ & $\\overline{\\sigma}$ & "
                     "$\\overline{w}_{95}$ & $\\widehat{C}_{95}$ \\\\\n")
        handle.write("\\hline\n")
        for case_name, rows in quality_by_case:
            for row in rows:
                handle.write(
                    f"{case_name} {row['function']} & {row['component']} & "
                    f"{row['q2']:.4f} & {row['mean_sigma']:.2e} & "
                    f"{row['mean_ci_width']:.2e} & {row['coverage_95']:.3f} \\\\\n"
                )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")
    return tex_path, csv_path


def fmt_vec(v):
    return np.array2string(np.asarray(v), precision=6, suppress_small=False)


def plot_validation(cases, y_star):
    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, OUTPUT_DIM + 1, figsize=(18, 4.2 * n_cases), squeeze=False)

    for row, case in enumerate(cases):
        name = case["name"]
        Y_r = case["Y_rigorous"]
        Y_p = case["Y_proposed"]
        det_path = case["det_path"]

        for k in range(OUTPUT_DIM):
            ax = axes[row, k]
            comp_color = COMPONENT_COLORS[k]
            bins = np.linspace(
                min(Y_r[:, k].min(), Y_p[:, k].min()),
                max(Y_r[:, k].max(), Y_p[:, k].max()),
                30,
            )
            ax.hist(
                Y_r[:, k],
                bins=bins,
                density=True,
                alpha=0.42,
                color=COLOR_RIGOROUS,
                label=rf"rigorous $y_{k + 1}$",
            )
            ax.hist(
                Y_p[:, k],
                bins=bins,
                density=True,
                alpha=0.42,
                color=COLOR_PROPOSED,
                label=rf"proposed $y_{k + 1}$",
            )
            ax.axvline(
                y_star[k],
                color=comp_color,
                linestyle="--",
                lw=2.0,
                alpha=0.95,
                label=rf"true component $y^\star_{k + 1}$",
            )
            ax.axvline(
                det_path[-1, k],
                color=COLOR_GP_PATH,
                linestyle=":",
                lw=2.0,
                label=rf"GP-mean $y_{k + 1}$",
            )
            ax.set_title(rf"{name}: $y_{k + 1}$", fontsize=TITLE_FONTSIZE)
            ax.set_xlabel(rf"$y_{k + 1}$", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("density", fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGEND_FONTSIZE)

        ax = axes[row, OUTPUT_DIM]
        norm_r = np.linalg.norm(Y_r, axis=1)
        norm_p = np.linalg.norm(Y_p, axis=1)
        bins = np.linspace(min(norm_r.min(), norm_p.min()), max(norm_r.max(), norm_p.max()), 32)
        ax.hist(
            norm_r,
            bins=bins,
            density=True,
            alpha=0.48,
            color=COLOR_RIGOROUS,
            label="rigorous pathwise",
        )
        ax.hist(
            norm_p,
            bins=bins,
            density=True,
            alpha=0.48,
            color=COLOR_PROPOSED,
            label="proposed fixed path",
        )
        ax.axvline(
            np.linalg.norm(y_star),
            color=COLOR_TRUE,
            linestyle="--",
            lw=2.0,
            label=r"true $\|\mathbf{y}^\star\|_2$",
        )
        ax.axvline(
            np.linalg.norm(det_path[-1]),
            color=COLOR_GP_PATH,
            linestyle=":",
            lw=2.0,
            label=r"GP-mean $\|\mathbf{y}\|_2$",
        )
        ax.set_title(rf"{name}: distribution of $\|\mathbf{{y}}\|_2$", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(
            r"$\|\mathbf{y}\|_2$ = Euclidean norm of coupled output $\mathbf{y}=(y_1,y_2,y_3)$",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        ax.set_ylabel("density", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "multidim_method_validation.png", dpi=200)
    plt.close(fig)


def plot_3d_cloud(cases, y_star):
    fig = plt.figure(figsize=(7 * len(cases), 6.8))

    all_points = [np.asarray(y_star).reshape(1, -1)]
    for case in cases:
        all_points.extend([case["Y_rigorous"], case["Y_proposed"], case["det_path"]])
    all_points = np.vstack(all_points)
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    radius = 0.55 * np.max(all_points.max(axis=0) - all_points.min(axis=0))
    radius = max(radius, 1e-3)
    axis_limits = [(center[k] - radius, center[k] + radius) for k in range(OUTPUT_DIM)]

    for idx, case in enumerate(cases, start=1):
        ax = fig.add_subplot(1, len(cases), idx, projection="3d")
        Y_r = case["Y_rigorous"]
        Y_p = case["Y_proposed"]
        std_radius_r = np.sqrt(np.trace(np.cov(Y_r.T, ddof=1)))
        std_radius_p = np.sqrt(np.trace(np.cov(Y_p.T, ddof=1)))

        ax.scatter(
            Y_r[:, 0],
            Y_r[:, 1],
            Y_r[:, 2],
            s=10,
            alpha=0.30,
            color=COLOR_RIGOROUS,
        )
        ax.scatter(
            Y_p[:, 0],
            Y_p[:, 1],
            Y_p[:, 2],
            s=10,
            alpha=0.30,
            color=COLOR_PROPOSED,
        )
        ax.scatter(
            Y_r[:, 0].mean(),
            Y_r[:, 1].mean(),
            Y_r[:, 2].mean(),
            c=COLOR_RIGOROUS,
            s=70,
            marker="X",
            edgecolor=COLOR_GP_PATH,
            linewidth=0.7,
        )
        ax.scatter(
            Y_p[:, 0].mean(),
            Y_p[:, 1].mean(),
            Y_p[:, 2].mean(),
            c=COLOR_PROPOSED,
            s=70,
            marker="X",
            edgecolor=COLOR_GP_PATH,
            linewidth=0.7,
        )
        ax.plot(
            case["det_path"][:, 0],
            case["det_path"][:, 1],
            case["det_path"][:, 2],
            color=COLOR_GP_PATH,
            marker="o",
            ms=3,
            lw=1.4,
        )
        ax.scatter(y_star[0], y_star[1], y_star[2], c=COLOR_TRUE, s=90, marker="*")
        ax.set_title(f"{case['name']} (n={case['n_train']})")
        ax.text2D(
            0.03,
            0.03,
            rf"cloud size $\sqrt{{\mathrm{{tr}}(\widehat{{\mathrm{{Cov}}}})}}$"
            + "\n"
            + rf"rigorous = {std_radius_r:.3f}"
            + "\n"
            + rf"proposed = {std_radius_p:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
        )
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.set_zlabel(r"$y_3$")
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])
        ax.set_box_aspect((1, 1, 1))
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, color=COLOR_RIGOROUS, label="rigorous samples"),
        Line2D([0], [0], marker="o", linestyle="", markersize=6, color=COLOR_PROPOSED, label="proposed samples"),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="",
            markersize=8,
            markerfacecolor=COLOR_RIGOROUS,
            markeredgecolor=COLOR_GP_PATH,
            label="rigorous sample mean",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="",
            markersize=8,
            markerfacecolor=COLOR_PROPOSED,
            markeredgecolor=COLOR_GP_PATH,
            label="proposed sample mean",
        ),
        Line2D([0], [0], color=COLOR_GP_PATH, marker="o", markersize=4, lw=1.5, label="GP-mean deterministic path"),
        Line2D([0], [0], marker="*", linestyle="", markersize=11, color=COLOR_TRUE, label=r"true fixed point $\mathbf{y}^\star$"),
    ]
    fig.suptitle(r"3D coupled-output cloud for $\mathbf{y}=(y_1,y_2,y_3)$", fontsize=15)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    plt.tight_layout(rect=[0, 0.12, 1, 0.94])
    fig.savefig(FIGURES_DIR / "multidim_3d_output_cloud.png", dpi=200)
    plt.close(fig)


def main():
    y_star, true_path = solve_fixed_point(coupled_true_map)
    rho_est = estimate_contraction()
    image_lower, image_upper, maps_into_unit_cube = estimate_coupling_image_bounds()
    print("Multidimensional analytical benchmark")
    print(f"  function input dimension = {INPUT_DIM}")
    print(f"  intermediate Gamma_1 output dimension = {INTERMEDIATE_DIM}")
    print(f"  coupled output dimension = {OUTPUT_DIM}")
    print("  structured GP f1 mean = (x1, x2, x3, 0, 0, 0)")
    print("  structured GP f1 covariance diagonal = (0, 0, 0, k, k, k)")
    print("  structured GP f2 mean = (y4/2, y5/2, y6/2)")
    print("  structured GP f2 covariance diagonal = (k, k, k) on (y1, y2, y3)")
    print(f"  T([0,1]^3) componentwise min = {fmt_vec(image_lower)}")
    print(f"  T([0,1]^3) componentwise max = {fmt_vec(image_upper)}")
    print(f"  numerical check T([0,1]^3) subset [0,1]^3 = {maps_into_unit_cube}")
    if not maps_into_unit_cube:
        print("  WARNING: numerical image check failed; T may not map [0,1]^3 into itself.")
    print(f"  estimated contraction spectral norm = {rho_est:.4f}")
    print(f"  true fixed point y* = {fmt_vec(y_star)}")
    print(f"  true fixed-point iterations = {len(true_path) - 1}")

    configs = [
        {"name": "smallDOE", "n_train": 20},
        {"name": "largeDOE", "n_train": 500},
    ]

    cases = []
    quality_by_case = []
    for cfg in configs:
        print(f"\n=== {cfg['name']} | n_train={cfg['n_train']} | MC={MC_SAMPLES} ===")
        gps1, gps2, X, y1, y2 = build_surrogates(cfg["n_train"])
        quality_rows = evaluate_surrogate_quality(gps1, gps2)
        quality_by_case.append((cfg["name"], quality_rows))

        print("Metamodel quality:")
        for row in quality_rows:
            print(
                f"  {row['function']} {row['component']}: "
                f"Q2={row['q2']:.4f}, "
                f"mean sigma={row['mean_sigma']:.3e}, "
                f"mean 95% width={row['mean_ci_width']:.3e}, "
                f"95% coverage={row['coverage_95']:.3f}"
            )

        det_path = compute_mean_path(gps1, gps2)
        print(f"GP-mean fixed point = {fmt_vec(det_path[-1])}")
        print(f"GP-mean path length = {len(det_path) - 1}")

        t0 = time.perf_counter()
        Y_rigorous, it_r, path_dist = run_rigorous_method(gps1, gps2)
        t_rigorous = time.perf_counter() - t0

        t0 = time.perf_counter()
        Y_proposed, it_p = run_proposed_method(gps1, gps2, det_path)
        t_proposed = time.perf_counter() - t0

        summary_r = summarize_samples(Y_rigorous)
        summary_p = summarize_samples(Y_proposed)
        tests = compare_methods(Y_rigorous, Y_proposed)

        print("Rigorous method:")
        print(f"  mean = {fmt_vec(summary_r['mean'])}")
        print(f"  var  = {fmt_vec(summary_r['var'])}")
        print(f"  95% CI low/high = {fmt_vec(summary_r['ci_low'])} / {fmt_vec(summary_r['ci_high'])}")
        print(f"  average iterations = {it_r.mean():.2f}")
        print(f"  elapsed time = {t_rigorous:.2f} s")

        print("Proposed fixed-path method:")
        print(f"  mean = {fmt_vec(summary_p['mean'])}")
        print(f"  var  = {fmt_vec(summary_p['var'])}")
        print(f"  95% CI low/high = {fmt_vec(summary_p['ci_low'])} / {fmt_vec(summary_p['ci_high'])}")
        print(f"  average iterations = {it_p.mean():.2f}")
        print(f"  elapsed time = {t_proposed:.2f} s")

        print("Method comparison:")
        print(f"  mean difference rigorous - proposed = {fmt_vec(tests['mean_diff'])}")
        for k, (ks_stat, ks_p, t_stat, t_p) in enumerate(tests["component_tests"], start=1):
            print(f"  y{k}: KS={ks_stat:.4f} (p={ks_p:.4f}), Welch t={t_stat:.3f} (p={t_p:.4f})")
        ks_norm, ks_p_norm, t_norm, t_p_norm = tests["norm_test"]
        print(f"  ||y||: KS={ks_norm:.4f} (p={ks_p_norm:.4f}), Welch t={t_norm:.3f} (p={t_p_norm:.4f})")
        print(f"  max trajectory distance to deterministic path: mean={path_dist.mean():.3e}, q95={np.percentile(path_dist, 95):.3e}")
        print(f"  speed-up proposed vs rigorous = {t_rigorous / max(t_proposed, 1e-12):.1f}x")

        cases.append(
            {
                "name": cfg["name"],
                "Y_rigorous": Y_rigorous,
                "Y_proposed": Y_proposed,
                "det_path": det_path,
                "n_train": cfg["n_train"],
            }
        )

    table_tex, table_csv = write_surrogate_quality_table(quality_by_case)
    plot_validation(cases, y_star)
    plot_3d_cloud(cases, y_star)
    print(f"\nFigures written to: {FIGURES_DIR}")
    print(f"  - {table_tex.name}")
    print(f"  - {table_csv.name}")
    print("  - multidim_method_validation.png")
    print("  - multidim_3d_output_cloud.png")


if __name__ == "__main__":
    main()
