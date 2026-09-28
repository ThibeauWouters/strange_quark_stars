"""Compare the Mauviard and Miller J0614 NICER inference runs using jester.

Loads the two ``results.h5`` posteriors via jester's ``InferenceResult``
loader (through the postprocessing module's ``load_eos_data`` helper), then
produces:

1. A 2x2 histogram comparison of M_TOV, R_1.4, Lambda_1.4 and n_TOV.
2. A mass-radius credible-interval band and a c_s^2-vs-density credible-
   interval band, overlaid for both runs.
"""

import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from jesterTOV.inference.postprocessing.postprocessing import (
    HDI_PROB,
    TEX_ENABLED,
    _get_density_mask,
    _get_valid_indices,
    load_eos_data,
    report_credible_interval,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "comparison_plots")

RUNS = {
    "mauviard": {
        "outdir": os.path.join(HERE, "mauviard", "outdir"),
        "label": "Mauviard+",
        "color": "#f4a261",
    },
    "miller": {
        "outdir": os.path.join(HERE, "miller", "outdir"),
        "label": "Miller+",
        "color": "#5e3c99",
    },
}

M_MIN, M_MAX = 0.75, 3.5
R_MIN, R_MAX = 8.0, 16.0
N_MIN, N_MAX = 0.5, 6.0

TITLE_HDI_PROB = 0.95


def _mtov(mass_row: np.ndarray, radius_row: np.ndarray) -> float:
    nonzero = np.nonzero(radius_row > 0)[0]
    if len(nonzero) == 0:
        return float(np.max(mass_row))
    return float(mass_row[nonzero[-1]])


def _interp_on_branch(
    target_mass: float,
    mass_row: np.ndarray,
    radius_row: np.ndarray,
    value_row: np.ndarray,
) -> float:
    nonzero = np.nonzero(radius_row > 0)[0]
    if len(nonzero) == 0:
        return float("nan")
    lo, hi = mass_row[nonzero[0]], mass_row[nonzero[-1]]
    if target_mass < lo or target_mass > hi:
        return float("nan")
    return float(np.interp(target_mass, mass_row[nonzero], value_row[nonzero]))


def compute_summary_quantities(data: dict) -> dict:
    """Compute M_TOV, R_1.4, Lambda_1.4 and n_TOV arrays for one run."""
    m, r, l = data["masses"], data["radii"], data["lambdas"]

    MTOV = np.array([_mtov(mass, radius) for mass, radius in zip(m, r)])
    R14 = np.array(
        [_interp_on_branch(1.4, mass, radius, radius) for mass, radius in zip(m, r)]
    )
    Lambda14 = np.array(
        [_interp_on_branch(1.4, mass, radius, lam) for mass, radius, lam in zip(m, r, l)]
    )
    R14 = R14[~np.isnan(R14)]
    Lambda14 = Lambda14[~np.isnan(Lambda14)]

    n_TOV_raw = data.get("n_TOV", None)
    n_TOV = np.array(n_TOV_raw)[np.array(n_TOV_raw) > 0.0] if n_TOV_raw is not None else None

    return {"MTOV": MTOV, "R14": R14, "Lambda14": Lambda14, "n_TOV": n_TOV}


def _plot_kde(ax, values: np.ndarray, color: str, label: str) -> None:
    if values is None or len(values) < 2 or float(np.std(values)) == 0.0:
        return
    low_err, med, high_err = report_credible_interval(values, hdi_prob=HDI_PROB)
    x_min = med - low_err - 0.25 * (low_err + high_err)
    x_max = med + high_err + 0.25 * (low_err + high_err)
    kde = gaussian_kde(values)
    x = np.linspace(x_min, x_max, 500)
    y = kde(x)
    ax.plot(x, y, color=color, lw=2.5, label=label)
    ax.fill_between(x, y, alpha=0.25, color=color)


def _format_hdi_summary(values: np.ndarray, hdi_prob: float = TITLE_HDI_PROB) -> str:
    """Median +/- HDI interval string, formatted like jester's histogram titles."""
    median = float(np.median(values))
    hdi = az.hdi(np.asarray(values), hdi_prob=hdi_prob)
    hdi_low, hdi_high = float(hdi[0]), float(hdi[1])
    low_err = median - hdi_low
    high_err = hdi_high - median
    if TEX_ENABLED:
        return f"${median:.2f}_{{-{low_err:.2f}}}^{{+{high_err:.2f}}}$"
    return f"{median:.2f} -{low_err:.2f} +{high_err:.2f}"


def plot_histogram_comparison(runs_data: dict, plot_format: str = "pdf") -> None:
    """Create a 2x2 KDE comparison of M_TOV, R_1.4, Lambda_1.4 and n_TOV."""
    summaries = {name: compute_summary_quantities(data) for name, data in runs_data.items()}

    if TEX_ENABLED:
        panels = [
            ("MTOV", r"$M_{\rm{TOV}}$ [$M_{\odot}$]"),
            ("R14", r"$R_{1.4}$ [km]"),
            ("Lambda14", r"$\Lambda_{1.4}$"),
            ("n_TOV", r"$n_{\rm{TOV}}$ [$n_{\rm{sat}}$]"),
        ]
    else:
        panels = [
            ("MTOV", "M_TOV [M_sun]"),
            ("R14", "R_1.4 [km]"),
            ("Lambda14", "Lambda_1.4"),
            ("n_TOV", "n_TOV [n_sat]"),
        ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (key, xlabel) in zip(axes, panels):
        titles = []
        for name, run_info in RUNS.items():
            values = summaries[name][key]
            _plot_kde(ax, values, run_info["color"], run_info["label"])
            if values is not None and len(values) > 1 and float(np.std(values)) > 0.0:
                titles.append((_format_hdi_summary(values), run_info["color"]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.set_ylim(bottom=0.0)

        if len(titles) == 2:
            (str1, col1), (str2, col2) = titles
            ax.text(
                0.0, 1.02, str1, transform=ax.transAxes,
                ha="left", va="bottom", color=col1, fontsize=14,
            )
            ax.text(
                1.0, 1.02, str2, transform=ax.transAxes,
                ha="right", va="bottom", color=col2, fontsize=14,
            )

    axes[0].legend()
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    save_name = os.path.join(OUTDIR, f"histogram_comparison.{plot_format}")
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram comparison saved to {save_name}")


def plot_band_comparison(runs_data: dict, plot_format: str = "pdf") -> None:
    """Create mass-radius and c_s^2-density credible-interval band comparisons."""
    fig, (ax_mr, ax_cs2) = plt.subplots(1, 2, figsize=(16, 7))

    masses_array = np.linspace(M_MIN, M_MAX, 100)
    dens_array = np.linspace(N_MIN, N_MAX, 100)

    for name, run_info in RUNS.items():
        data = runs_data[name]
        color = run_info["color"]
        label = run_info["label"]

        # --- Mass-radius band ---
        m, r = data["masses"], data["radii"]
        valid_indices, _ = _get_valid_indices(data)
        m_valid = [m[i] for i in valid_indices]
        r_valid = [r[i] for i in valid_indices]

        radii_low = np.full_like(masses_array, np.nan)
        radii_high = np.full_like(masses_array, np.nan)
        for i, mass_point in enumerate(masses_array):
            radii_at_mass = np.array(
                [
                    float(np.interp(mass_point, mass, radius, left=np.nan, right=np.nan))
                    for mass, radius in zip(m_valid, r_valid)
                ]
            )
            radii_at_mass = radii_at_mass[~np.isnan(radii_at_mass)]
            if len(radii_at_mass) == 0:
                continue
            low, med, high = report_credible_interval(radii_at_mass, hdi_prob=HDI_PROB)
            radii_low[i] = med - low
            radii_high[i] = med + high

        mask = ~np.isnan(radii_low) & ~np.isnan(radii_high)
        ax_mr.fill_betweenx(
            masses_array[mask], radii_low[mask], radii_high[mask], alpha=0.4, color=color
        )
        ax_mr.plot(radii_low[mask], masses_array[mask], lw=2.0, color=color, label=label)
        ax_mr.plot(radii_high[mask], masses_array[mask], lw=2.0, color=color)

        # --- c_s^2 vs density band ---
        n, cs2 = data["densities"], data["cs2"]
        n_TOV = data.get("n_TOV", None)

        cs2_low = np.full_like(dens_array, np.nan)
        cs2_high = np.full_like(dens_array, np.nan)
        for i, dens_point in enumerate(dens_array):
            vals = []
            for j in valid_indices:
                mask_j = _get_density_mask(
                    n[j], float(n_TOV[j]) if n_TOV is not None else None
                )
                n_j, cs2_j = n[j][mask_j], cs2[j][mask_j]
                if len(n_j) == 0 or dens_point < n_j.min() or dens_point > n_j.max():
                    continue
                vals.append(float(np.interp(dens_point, n_j, cs2_j)))
            if len(vals) == 0:
                continue
            low, med, high = report_credible_interval(np.array(vals), hdi_prob=HDI_PROB)
            cs2_low[i] = med - low
            cs2_high[i] = med + high

        mask = ~np.isnan(cs2_low) & ~np.isnan(cs2_high)
        ax_cs2.fill_between(
            dens_array[mask], cs2_low[mask], cs2_high[mask], alpha=0.4, color=color
        )
        ax_cs2.plot(dens_array[mask], cs2_low[mask], lw=2.0, color=color, label=label)
        ax_cs2.plot(dens_array[mask], cs2_high[mask], lw=2.0, color=color)

    ax_mr.set_xlabel(r"$R$ [km]" if TEX_ENABLED else "R [km]")
    ax_mr.set_ylabel(r"$M$ [$M_{\odot}$]" if TEX_ENABLED else "M [M_sun]")
    ax_mr.set_xlim(R_MIN, R_MAX)
    ax_mr.set_ylim(M_MIN, M_MAX)
    ax_mr.legend()

    ax_cs2.set_xlabel(r"$n$ [$n_{\rm{sat}}$]" if TEX_ENABLED else "n [n_sat]")
    ax_cs2.set_ylabel(r"$c_s^2$" if TEX_ENABLED else "cs2")
    ax_cs2.set_xlim(N_MIN, N_MAX)
    ax_cs2.set_ylim(0.0, 1.2)
    ax_cs2.legend()

    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    save_name = os.path.join(OUTDIR, f"band_comparison.{plot_format}")
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)
    print(f"Band comparison saved to {save_name}")


def main() -> None:
    runs_data = {name: load_eos_data(run_info["outdir"]) for name, run_info in RUNS.items()}

    plot_histogram_comparison(runs_data)
    plot_band_comparison(runs_data)


if __name__ == "__main__":
    main()
