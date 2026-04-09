
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import ks_2samp, ttest_ind
from sklearn.exceptions import ConvergenceWarning
import warnings

# Silence optimizer bound warnings from sklearn GP fitting
warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Global controls (reported for reproducibility)
FP_TOL = 1e-8      # fixed-point tolerance epsilon
MAX_IT = 50        # max fixed-point iterations
MC_SAMPLES = 500   # Monte Carlo replications per method

# For plots: avoid huge "density peaks" when distributions are very concentrated


def f1_true(x):
    x = np.asarray(x)
    return (
        0.12
        + 0.18 * x
        + 0.06 * np.sin(2.0 * np.pi * x)
        + 0.05 * np.exp(-60.0 * (x - 0.70) ** 2)
        + 0.03 * x * (1.0 - x)
    )

def f2_true(x):
    x = np.asarray(x)
    return (
        0.58
        - 0.22 * x
        + 0.03 * np.tanh(8.0 * (x - 0.40))
        + 0.015 * np.sin(4.0 * np.pi * x)
    )

# ---- Optional (recommended): check contraction numerically on a fine grid
def _sech2(z):
    return 1.0 / np.cosh(z) ** 2

def f1_prime(x):
    x = np.asarray(x)
    return (
        0.18
        + 0.06 * (2.0 * np.pi) * np.cos(2.0 * np.pi * x)
        + 0.05 * np.exp(-60.0 * (x - 0.70) ** 2) * (-120.0 * (x - 0.70))
        + 0.03 * (1.0 - 2.0 * x)
    )

def f2_prime(x):
    x = np.asarray(x)
    return (
        -0.22
        + 0.03 * 8.0 * _sech2(8.0 * (x - 0.40))
        + 0.015 * (4.0 * np.pi) * np.cos(4.0 * np.pi * x)
    )

_grid = np.linspace(0.0, 1.0, 20001)
rho_est = np.max(np.abs(0.5 * (f1_prime(_grid) + f2_prime(_grid))))
print("Estimated contraction modulus rho_est =", rho_est)

# ---- Exact coupled solution y*: solve g(y)=0 on [0,1] with bisection
def g_fp(y):
    # fixed-point equation: y = 0.5*(f1(y)+f2(y))
    return 0.5 * (float(f1_true(y)) + float(f2_true(y))) - y

def bisection_root(fun, a=0.0, b=1.0, tol=1e-14, maxit=300):
    fa, fb = fun(a), fun(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise RuntimeError(
            "Bisection failed: g(0) and g(1) have the same sign. "
            "Either the fixed point is not in [0,1] or change the functions/bracket."
        )
    for _ in range(maxit):
        m = 0.5 * (a + b)
        fm = fun(m)
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            return m
        if fa * fm > 0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    return 0.5 * (a + b)

y_star = bisection_root(g_fp, a=0.0, b=1.0, tol=1e-14, maxit=300)
print("Exact coupled solution y* =", y_star)
# ==========================================================
# 2) Build surrogates for a given design (DET + Matérn + hyperparams FIXÉS)
# ==========================================================
def lhs_1d(n, low=0.0, high=1.0):
    """Simple 1D Latin Hypercube sampling on [low, high]."""
    cut = np.linspace(low, high, n + 1)
    u = cut[:-1] + (cut[1:] - cut[:-1]) * np.random.rand(n)
    np.random.shuffle(u)
    return u.reshape(-1, 1)

def build_surrogates(n_train, noise, length_scale=0.1, sampling="lhs"):
    """
    Deterministic data (noise must be 0.0 to match your theory).
    GP: Constant * Matérn with FIXED hyperparameters (optimizer=None),
    and ONLY a tiny numerical jitter via alpha (not an observation noise nugget).
    This avoids the MLE length-scale blow-up that makes posterior variance ~0 everywhere.
    """
    if sampling == "lhs":
        X = lhs_1d(n_train)
    else:
        X = np.linspace(0, 1, n_train).reshape(-1, 1)

    # deterministic training data
    y1 = f1_true(X).ravel()
    y2 = f2_true(X).ravel()

    # fixed kernel (no MLE)
    kernel = Matern(
        length_scale=length_scale,
        length_scale_bounds="fixed",
        nu=2.5
    )

    # purely numerical jitter for stability (not "noise in theory")
    jitter = 1e-12

    gp1 = GaussianProcessRegressor(
        kernel=kernel,
        alpha=jitter,
        optimizer=None,       # <-- crucial: no hyperparameter optimization
        normalize_y=False
    )
    gp2 = GaussianProcessRegressor(
        kernel=kernel,
        alpha=jitter,
        optimizer=None,
        normalize_y=False
    )

    gp1.fit(X, y1)
    gp2.fit(X, y2)
    return gp1, gp2, X, y1, y2

# ==========================================================
# 4) Coupled solver (fixed-point iteration)
# ==========================================================
def solve_coupled(f1, f2, x0=0.5, tol=1e-8, maxit=200):
    y = x0
    for _ in range(maxit):
        y_new = 0.5*(f1(y) + f2(y))
        if abs(y_new - y) < tol:
            return y_new
        y = y_new
    return y

# GP mean utility
def gp_mean(gp, x):
    return gp.predict(np.array([[x]]))[0]

def gp_mean_and_latent_std(gp, X_query):
    """Return GP predictive mean and std (latent uncertainty; no nugget kernel)."""
    mean, std = gp.predict(X_query, return_std=True)
    return mean, std

# ==========================================================
# 5) Method 2: rigorous conditional sampling along trajectory
# ==========================================================
def sample_conditional(gp, x_new, X_seen, y_seen, jitter=1e-10):
    """
    Sample f(x_new) from the GP posterior conditioned on:
      - training data (already inside gp)
      - previously drawn path values (X_seen, y_seen)
    This matches the "rigorous but heavy" sequential conditioning described in theory.
    """
    if X_seen.size == 0:
        mean, std = gp.predict(np.array([[x_new]]), return_std=True)
        return mean[0] + std[0]*np.random.randn()

    X_full = np.vstack([X_seen, [[x_new]]])
    mean_full, cov_full = gp.predict(X_full, return_cov=True)  # posterior over X_full (given training)

    mean_seen, mean_new = mean_full[:-1], mean_full[-1]
    K_ss = cov_full[:-1, :-1]
    k_s_new = cov_full[:-1, -1]

    K_ss_jitter = K_ss + jitter*np.eye(K_ss.shape[0])

    alpha = np.linalg.solve(K_ss_jitter, y_seen - mean_seen)
    cond_mean = mean_new + k_s_new @ alpha
    cond_var = cov_full[-1, -1] - k_s_new @ np.linalg.solve(K_ss_jitter, k_s_new)
    cond_std = np.sqrt(max(cond_var, 0.0))
    return cond_mean + cond_std*np.random.randn()

def run_method2(gp_1, gp_2, x0=0.5, tol=1e-8, maxit=50, n_samples=500):
    Ys = np.zeros(n_samples)
    for j in range(n_samples):
        y = x0
        X_seen_1 = np.empty((0,1))
        X_seen_2 = np.empty((0,1))
        y_seen_1 = np.array([])
        y_seen_2 = np.array([])

        for _ in range(maxit):
            f1_val = sample_conditional(gp_1, y, X_seen_1, y_seen_1)
            f2_val = sample_conditional(gp_2, y, X_seen_2, y_seen_2)

            X_seen_1 = np.vstack([X_seen_1, [[y]]])
            y_seen_1 = np.append(y_seen_1, f1_val)
            X_seen_2 = np.vstack([X_seen_2, [[y]]])
            y_seen_2 = np.append(y_seen_2, f2_val)

            y_new = 0.5*(f1_val + f2_val)
            if abs(y_new - y) < tol:
                y = y_new
                break
            y = y_new

        Ys[j] = y
    return Ys

# ==========================================================
# 6) Method 3: proposed constant-offset approximation
# ==========================================================
def compute_mean_path(mu1_func, mu2_func, x0=0.5, tol=1e-10, maxit=50):
    path = [x0]
    y = x0
    for _ in range(maxit):
        y_new = 0.5*(mu1_func(y) + mu2_func(y))
        path.append(y_new)
        if abs(y_new - y) < tol:
            break
        y = y_new
    return np.array(path)

def run_method3(gp_1, gp_2, deterministic_path, n_samples=500, tol=1e-8):
    """
    Method 3 (theory): fix deterministic mean path x_m, sample joint GP vector on {x_m},
    define deltas = f(x_m)-mu(x_m), then use perturbed surrogate mu(x)+delta_m at iteration m.
    """
    Y_path = deterministic_path[1:]  # exclude initial iterate

    # posterior joint cov on path points (given training)
    mu1_vec, cov1 = gp_1.predict(Y_path.reshape(-1,1), return_cov=True)
    mu2_vec, cov2 = gp_2.predict(Y_path.reshape(-1,1), return_cov=True)

    # small jitter for Cholesky stability (path can contain near-duplicates)
    chol_jitter = 1e-12
    L1 = np.linalg.cholesky(cov1 + chol_jitter*np.eye(len(Y_path)))
    L2 = np.linalg.cholesky(cov2 + chol_jitter*np.eye(len(Y_path)))

    def mu1_func(x): return gp_mean(gp_1, x)
    def mu2_func(x): return gp_mean(gp_2, x)

    Ys = np.zeros(n_samples)
    for j in range(n_samples):
        z1 = np.random.randn(len(Y_path))
        z2 = np.random.randn(len(Y_path))
        deltas1 = L1 @ z1
        deltas2 = L2 @ z2

        y = deterministic_path[0]
        for m in range(len(Y_path)):
            # constant offset at iteration m (independent of x)
            y_new = 0.5*((mu1_func(y) + deltas1[m]) + (mu2_func(y) + deltas2[m]))
            if abs(y_new - y) < tol:
                y = y_new
                break
            y = y_new

        Ys[j] = y
    return Ys

# ==========================================================
# 7) Reporting and plotting helpers
# ==========================================================
def summarize_samples(samples):
    mean_val = samples.mean()
    var_val = samples.var(ddof=1)
    ci = (np.percentile(samples, 2.5), np.percentile(samples, 97.5))
    return mean_val, var_val, ci


def compare_methods(Y2, Y3):
    """Basic stats to quantify difference between Method 2 and Method 3 samples."""
    mean_diff = Y2.mean() - Y3.mean()
    ks_stat, ks_p = ks_2samp(Y2, Y3)
    t_stat, t_p = ttest_ind(Y2, Y3, equal_var=False)
    return {
        "mean_diff": mean_diff,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "t_stat": t_stat,
        "t_p": t_p,
    }

def plot_surrogates(gp1, gp2, X, y1_train, y2_train, cfg_name, n_train, noise):
    """Plot true functions, GP means, training data, and ±2σ bands for both surrogates."""
    grid = np.linspace(0, 1, 200).reshape(-1, 1)
    mu1, std1 = gp_mean_and_latent_std(gp1, grid)
    mu2, std2 = gp_mean_and_latent_std(gp2, grid)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Code 1
    axes[0].fill_between(grid.ravel(), mu1 - 2*std1, mu1 + 2*std1,
                         color='C0', alpha=0.2, label='f1 ±2σ', zorder=1)
    axes[0].plot(grid, mu1, color='C0', linewidth=2.0, label='f1 mean', zorder=2)
    axes[0].plot(grid, f1_true(grid), 'k--', linewidth=2.2, label='g1 true', zorder=3)
    axes[0].scatter(X, y1_train, color='C0', s=18, alpha=0.7, label='DOE points', zorder=4)
    axes[0].set_title(f'Code 1 surrogate ({cfg_name})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('g1(x)')
    axes[0].grid(True)
    axes[0].legend(fontsize=12)

    # Code 2
    axes[1].fill_between(grid.ravel(), mu2 - 2*std2, mu2 + 2*std2,
                         color='C1', alpha=0.2, label='f2 ±2σ', zorder=1)
    axes[1].plot(grid, mu2, color='C1', linewidth=2.0, label='f2 mean', zorder=2)
    axes[1].plot(grid, f2_true(grid), 'k--', linewidth=2.2, label='g2 true', zorder=3)
    axes[1].scatter(X, y2_train, color='C1', s=18, alpha=0.7, label='DOE points', zorder=4)
    axes[1].set_title(f'Code 2 surrogate ({cfg_name})')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('g2(x)')
    axes[1].grid(True)
    axes[1].legend(fontsize=12)

    fig.suptitle(f'Surrogates (n_train={n_train}, noise={noise})', fontsize=16)
    plt.subplots_adjust(wspace=0.25, top=0.82)
    plt.savefig(FIGURES_DIR / f"surrogates_{cfg_name}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_distributions_overview(cases, plot_as_density=True, nbins=30):
    if len(cases) < 2:
        return

    # IMPORTANT: sharey=False (sinon largeDOE écrase smallDOE en densité)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, (cfg_name, Ys_method2, Ys_method3) in zip(axes, cases):
        all_y = np.concatenate([Ys_method2, Ys_method3])
        bins = np.linspace(all_y.min(), all_y.max(), nbins + 1)

        # histogrammes (bins communs + densité)
        ax.hist(Ys_method2, bins=bins, density=plot_as_density, alpha=0.55, label='Method 2')
        ax.hist(Ys_method3, bins=bins, density=plot_as_density, alpha=0.55, label='Method 3')

        ax.axvline(y_star, color='red', linestyle='--', linewidth=2, label='Exact y*')
        ax.axvline(Ys_method2.mean(), color='blue', linestyle='-', linewidth=2, label='M2 mean')
        ax.axvline(Ys_method3.mean(), color='green', linestyle='-', linewidth=2, label='M3 mean')

        ax.set_title(f'{cfg_name}', fontsize=16, pad=14)
        ax.set_xlabel('Coupled solution y', fontsize=14)
        ax.grid(True)

        # Optionnel mais utile: ylim automatique par subplot (lisible)
        h2, _ = np.histogram(Ys_method2, bins=bins, density=plot_as_density)
        h3, _ = np.histogram(Ys_method3, bins=bins, density=plot_as_density)
        ax.set_ylim(0.0, 1.10 * max(h2.max(), h3.max()))

    axes[0].set_ylabel('Density' if plot_as_density else 'Count', fontsize=14)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               bbox_to_anchor=(0.5, 1.10), fontsize=14, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURES_DIR / "benchmark_validation_overview.png", dpi=200, bbox_inches='tight')
    plt.close(fig)





# ==========================================================
# 10) Combined surrogate overview (four panels)
# ==========================================================
def plot_surrogates_combined(cases):
    if len(cases) < 2:
        return
    grid = np.linspace(0, 1, 200).reshape(-1, 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for idx, (cfg_name, gp1, gp2, X, y1_train, y2_train, n_train, noise, sampling) in enumerate(cases[:2]):
        row = idx
        mu1, std1 = gp_mean_and_latent_std(gp1, grid)
        mu2, std2 = gp_mean_and_latent_std(gp2, grid)

        ax1 = axes[row, 0]
        ax1.fill_between(grid.ravel(), mu1 - 2*std1, mu1 + 2*std1,
                         color='C0', alpha=0.2, label='f1 ±2σ', zorder=1)
        ax1.plot(grid, mu1, color='C0', linewidth=2.0, label='f1 mean', zorder=2)
        ax1.plot(grid, f1_true(grid), 'k--', linewidth=2.2, label='g1 true', zorder=3)
        ax1.scatter(X, y1_train, color='C0', s=14, alpha=0.7, label='DOE points', zorder=4)
        ax1.set_title(f'Code 1 ({cfg_name})', fontsize=14)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('g1(x)', fontsize=12)
        ax1.grid(True)

        ax2 = axes[row, 1]
        ax2.fill_between(grid.ravel(), mu2 - 2*std2, mu2 + 2*std2,
                         color='C1', alpha=0.2, label='f2 ±2σ', zorder=1)
        ax2.plot(grid, mu2, color='C1', linewidth=2.0, label='f2 mean', zorder=2)
        ax2.plot(grid, f2_true(grid), 'k--', linewidth=2.2, label='g2 true', zorder=3)
        ax2.scatter(X, y2_train, color='C1', s=14, alpha=0.7, label='DOE points', zorder=4)
        ax2.set_title(f'Code 2 ({cfg_name})', fontsize=14)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('g2(x)', fontsize=12)
        ax2.grid(True)

    # Combine legend entries and deduplicate
    handles0, labels0 = axes[0, 0].get_legend_handles_labels()
    handles1, labels1 = axes[0, 1].get_legend_handles_labels()
    all_handles = handles0 + handles1
    all_labels = labels0 + labels1

    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, lab in zip(all_handles, all_labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq_handles.append(h)
        uniq_labels.append(lab)

    fig.legend(uniq_handles, uniq_labels, loc='upper center', ncol=len(uniq_labels),
               bbox_to_anchor=(0.5, 1.06), fontsize=12, frameon=True)
    fig.suptitle('Surrogates overview (small vs large DOE)', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(FIGURES_DIR / "surrogates_overview.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

# ==========================================================
# Run (two DOEs)
# ==========================================================
configs = [
    {"name": "smallDOE", "n_train": 20,  "noise": 0., "sampling": "lhs"},
    {"name": "largeDOE", "n_train": 200, "noise": 0., "sampling": "lhs"},
]

surrogate_cases = []
dist_cases = []

for cfg in configs:
    print(f"\n=== Experiment: {cfg['name']} (n_train={cfg['n_train']}, noise={cfg['noise']}, sampling={cfg['sampling']}, eps={FP_TOL}, MC={MC_SAMPLES}) ===")

    gp1, gp2, X, y1, y2 = build_surrogates(
        cfg["n_train"],
        cfg["noise"],
        length_scale=0.25,   # adjust here if you want wider/narrower bands
        sampling=cfg.get("sampling", "lhs")
    )
    surrogate_cases.append((cfg["name"], gp1, gp2, X, y1, y2, cfg["n_train"], cfg["noise"], cfg["sampling"]))

    # optional: save individual surrogate plots too
    plot_surrogates(gp1, gp2, X, y1, y2, cfg["name"], cfg["n_train"], cfg["noise"])

    mu1_func = lambda x, gp1=gp1: gp_mean(gp1, x)
    mu2_func = lambda x, gp2=gp2: gp_mean(gp2, x)
    det_path = compute_mean_path(mu1_func, mu2_func, tol=FP_TOL, maxit=MAX_IT)

    Ys_method2 = run_method2(gp1, gp2, tol=FP_TOL, maxit=MAX_IT, n_samples=MC_SAMPLES)
    Ys_method3 = run_method3(gp1, gp2, det_path, n_samples=MC_SAMPLES, tol=FP_TOL)

    mean2, var2, ci2 = summarize_samples(Ys_method2)
    mean3, var3, ci3 = summarize_samples(Ys_method3)
    tests = compare_methods(Ys_method2, Ys_method3)

    print(f"Method 2: mean={mean2:.6f}, var={var2:.3e}, 95% CI={ci2}")
    print(f"Method 3: mean={mean3:.6f}, var={var3:.3e}, 95% CI={ci3}")
    print(f"Mean difference (M2 - M3): {tests['mean_diff']:.6f}")
    print(f"KS test: statistic={tests['ks_stat']:.4f}, p-value={tests['ks_p']:.4f}")
    print(f"Welch t-test: t={tests['t_stat']:.4f}, p-value={tests['t_p']:.4f}")

    dist_cases.append((cfg["name"], Ys_method2, Ys_method3))

# Combined surrogate overview for quick visual comparison
plot_surrogates_combined(surrogate_cases)

# Combined benchmark validation distributions
plot_distributions_overview(dist_cases)
