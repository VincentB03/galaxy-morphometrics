# Comparison plots adapted from deep_galaxy_models (deepgal/validation/plotting.py,
# Figure_Moments.ipynb, Figure_Morphology.ipynb), for any number of named datasets.
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("darkgrid")
sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2})


def _colors(names):
    """Assigns each dataset name a stable color along the viridis colormap."""
    n = len(names)
    return {name: plt.cm.viridis(0.1 + 0.7 * i / max(n - 1, 1)) for i, name in enumerate(names)}


def _mask(table, require_morph=False, extra_cols=()):
    m = np.asarray(table["flag"])
    if require_morph and "flag_morph" in table.colnames:
        m = m & np.asarray(table["flag_morph"])
    for c in extra_cols:
        if c in table.colnames:
            m = m & np.isfinite(np.asarray(table[c]))
    return m


def _savefig(out_dir, name):
    path = os.path.join(out_dir, name)
    plt.savefig(path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()
    return path


def moment_distributions(tables, pixel_scale=0.03, out_dir=".", filename="moments.pdf"):
    """Ellipticity |g| and size (determinant radius) distributions."""
    colors = _colors(list(tables.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, tab in tables.items():
        m = _mask(tab)
        sns.kdeplot(np.asarray(tab["g"][m]), label=name, color=colors[name], ax=axes[0])
        axes[0].axvline(np.mean(np.asarray(tab["g"][m])), color=colors[name])
    axes[0].set_xlim(0, 0.8)
    axes[0].set_xlabel("Ellipticity $g$")
    axes[0].legend()

    for name, tab in tables.items():
        m = _mask(tab)
        size = np.asarray(tab["sigma_e"][m]) * pixel_scale
        sns.kdeplot(size, label=name, color=colors[name], ax=axes[1])
        axes[1].axvline(np.mean(size), color=colors[name])
    axes[1].set_xlabel("Determinant radius [arcsec]")
    return _savefig(out_dir, filename)


def g1_g2_distributions(tables, out_dir=".", filename="g1_g2.pdf"):
    colors = _colors(list(tables.keys()))
    bins = np.linspace(-0.75, 0.75, 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, tab in tables.items():
        m = _mask(tab)
        sns.kdeplot(np.asarray(tab["g1"][m]), label=name, color=colors[name], ax=axes[0])
        sns.kdeplot(np.asarray(tab["g2"][m]), label=name, color=colors[name], ax=axes[1])
    axes[0].set_xlabel("$g_1$")
    axes[0].set_xlim(bins[0], bins[-1])
    axes[0].legend(loc=2)
    axes[1].set_xlabel("$g_2$")
    axes[1].set_xlim(bins[0], bins[-1])
    return _savefig(out_dir, filename)


def rho4_distribution(tables, out_dir=".", filename="rho4.pdf"):
    colors = _colors(list(tables.keys()))
    plt.figure()
    for name, tab in tables.items():
        m = _mask(tab)
        sns.kdeplot(np.asarray(tab["rho4"][m]), label=name, color=colors[name])
    plt.xlabel(r"$\rho_4$")
    plt.legend()
    return _savefig(out_dir, filename)


def binned_statistic_plot(
    tables,
    y_column,
    x_values,
    bins,
    xlabel,
    ylabel,
    out_dir=".",
    filename="binned.pdf",
    logy=False,
):
    """
    Plots mean(`y_column`) +/- sem in `bins` of a per-object quantity
    (`x_values`: dataset name -> array aligned with its table), e.g.
    ellipticity vs magnitude.
    """
    from scipy.stats import sem

    colors = _colors(list(tables.keys()))
    n = len(bins)
    offsets = np.linspace(-0.4, 0.4, len(tables)) * (bins[1] - bins[0]) * 0.2 if len(tables) > 1 else [0]

    plt.figure(figsize=(8, 5))
    for (name, tab), offset in zip(tables.items(), offsets):
        if name not in x_values:
            continue
        m = _mask(tab)
        y = np.asarray(tab[y_column][m])
        x = np.asarray(x_values[name])[m]
        inds = np.digitize(x, bins)
        res_m = np.zeros(n)
        res_std = np.zeros(n)
        for i in range(n):
            sel = y[inds == i]
            if len(sel) == 0:
                res_m[i] = np.nan
                res_std[i] = np.nan
                continue
            res_m[i] = np.mean(sel)
            res_std[i] = sem(sel) if len(sel) > 1 else 0.0
        plt.errorbar(bins + offset, res_m, res_std, color=colors[name], label=name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.legend()
    return _savefig(out_dir, filename)


def gini_m20_panels(tables, out_dir=".", filename="gini_m20.pdf", vmax=55):
    names = list(tables.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        tab = tables[name]
        m = _mask(tab, require_morph=True, extra_cols=("M20", "Gini"))
        ax.hist2d(
            np.asarray(tab["M20"][m]),
            np.asarray(tab["Gini"][m]),
            32,
            range=[[-2.0, -0.5], [0.05, 0.6]],
            cmap="viridis",
            vmax=vmax,
        )
        ax.set_xlim(-0.5, -2.0)
        ax.set_xlabel("M20")
        ax.set_title(name)
    axes[0].set_ylabel("Gini")
    return _savefig(out_dir, filename)


def M_I_panels(tables, out_dir=".", filename="M_I.pdf", vmax=120):
    names = list(tables.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        tab = tables[name]
        m = _mask(tab, require_morph=True, extra_cols=("M", "I"))
        m = m & (np.asarray(tab["I"]) > 0) & (np.asarray(tab["M"]) > 0)
        ax.hist2d(
            np.log10(np.asarray(tab["M"][m])),
            np.log10(np.asarray(tab["I"][m])),
            32,
            range=[[-5, 0], [-2.5, 0]],
            cmap="viridis",
            vmax=vmax,
        )
        ax.set_xlabel("log10(M)")
        ax.set_title(name)
    axes[0].set_ylabel("log10(I)")
    return _savefig(out_dir, filename)


def M_D_panels(tables, out_dir=".", filename="M_D.pdf", vmax=65):
    names = list(tables.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        tab = tables[name]
        m = _mask(tab, require_morph=True, extra_cols=("M", "D", "I"))
        m = m & (np.asarray(tab["I"]) > 0) & (np.asarray(tab["M"]) > 0)
        ax.hist2d(
            np.asarray(tab["D"][m]),
            np.log10(np.asarray(tab["M"][m])),
            32,
            range=[[0, 1], [-4, 0]],
            cmap="viridis",
            vmax=vmax,
        )
        ax.set_xlabel("D")
        ax.set_title(name)
    axes[0].set_ylabel("log10(M)")
    return _savefig(out_dir, filename)


def mid_distributions(tables, out_dir=".", filename="MID.pdf"):
    colors = _colors(list(tables.keys()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for name, tab in tables.items():
        base_mask = _mask(tab, require_morph=True)
        m_M = base_mask & (np.asarray(tab["M"]) > 0)
        if m_M.any():
            sns.kdeplot(np.log10(np.asarray(tab["M"][m_M])), color=colors[name], ax=axes[0])
        m_I = base_mask & (np.asarray(tab["I"]) > 0)
        if m_I.any():
            sns.kdeplot(np.log10(np.asarray(tab["I"][m_I])), color=colors[name], ax=axes[1])
        sns.kdeplot(np.asarray(tab["D"][base_mask]), label=name, color=colors[name], ax=axes[2])
    axes[0].set_xlabel("log10(M)")
    axes[1].set_xlabel("log10(I)")
    axes[2].set_xlabel("D")
    axes[2].set_xlim(0, 1)
    axes[2].legend()
    return _savefig(out_dir, filename)


def cas_distributions(tables, out_dir=".", filename="CAS.pdf"):
    """Distributions of the Concentration (C) and Asymmetry (A) indicators."""
    colors = _colors(list(tables.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, tab in tables.items():
        m = _mask(tab, require_morph=True, extra_cols=("C", "A"))
        m = m & (np.asarray(tab["C"]) > -9) & (np.asarray(tab["A"]) > -9)
        sns.kdeplot(np.asarray(tab["C"][m]), label=name, color=colors[name], ax=axes[0])
        sns.kdeplot(np.asarray(tab["A"][m]), color=colors[name], ax=axes[1])
    axes[0].set_xlabel("Concentration C")
    axes[0].legend()
    axes[1].set_xlabel("Asymmetry A")
    return _savefig(out_dir, filename)


def paired_reconstruction_error(
    reference_table,
    other_table,
    reference_name="real",
    other_name="reconstruction",
    pixel_scale=0.03,
    out_dir=".",
    filename_prefix="",
):
    """
    Per-object relative size error and flux ratio vs the reference size, for
    two object-by-object aligned tables (e.g. real vs reconstruction).
    """
    from scipy.stats import sem

    mask = _mask(reference_table) & _mask(other_table)

    size_ref = np.asarray(reference_table["sigma_e"])
    size_err = (np.asarray(reference_table["sigma_e"]) - np.asarray(other_table["sigma_e"])) / size_ref

    plt.figure(figsize=(6, 5))
    plt.hexbin(size_ref[mask] * pixel_scale, size_err[mask], gridsize=50, bins="log", cmap="viridis")
    plt.axhline(0, color="C1")
    n = 10
    b = np.logspace(np.log10(max(size_ref[mask].min(), 1e-3)), np.log10(size_ref[mask].max()), n) * pixel_scale
    inds = np.digitize(size_ref[mask] * pixel_scale, b)
    res_m = np.array([np.mean(size_err[mask][inds == i]) if np.any(inds == i) else np.nan for i in range(n)])
    res_std = np.array([sem(size_err[mask][inds == i]) if np.sum(inds == i) > 1 else 0.0 for i in range(n)])
    plt.errorbar(b, res_m, res_std, color="C3")
    plt.xlabel("%s determinant radius [arcsec]" % reference_name)
    plt.ylabel("(%s - %s) / %s size" % (reference_name, other_name, reference_name))
    size_path = _savefig(out_dir, filename_prefix + "size_error.pdf")

    amp_ratio = np.asarray(other_table["amp"]) / np.asarray(reference_table["amp"])
    m2 = mask & (amp_ratio < 2)
    plt.figure(figsize=(6, 5))
    plt.hexbin(size_ref[m2] * pixel_scale, np.clip(amp_ratio[m2], 0.0, 2), gridsize=50, bins="log", cmap="viridis")
    plt.axhline(1, color="C1")
    plt.xlabel("%s determinant radius [arcsec]" % reference_name)
    plt.ylabel("flux(%s) / flux(%s)" % (other_name, reference_name))
    flux_path = _savefig(out_dir, filename_prefix + "flux_error.pdf")

    return size_path, flux_path


def make_all_plots(
    tables,
    out_dir=".",
    pixel_scale=0.03,
    binning_values=None,
    binning_label="binning quantity",
    reference_name=None,
    paired_names=None,
    skip_morph=False,
):
    """
    Writes every applicable plot to `out_dir` and returns their paths. Plots
    whose inputs are missing (R columns, `binning_values`) are skipped.

    Parameters
    ----------
    tables: dict[str, astropy.table.Table]
        Output of `galmorph.pipeline.compute_statistics`.
    binning_values: dict[str, array_like], optional
        Per-dataset quantity (e.g. magnitude) for the binned plots.
    reference_name: str, optional
        Reference dataset for the per-object error plots.
    paired_names: list[str], optional
        Datasets aligned object by object with `reference_name` (default: all
        others). Exclude unconditional samples such as "flow_prior".
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    written.append(moment_distributions(tables, pixel_scale=pixel_scale, out_dir=out_dir))
    written.append(g1_g2_distributions(tables, out_dir=out_dir))
    written.append(rho4_distribution(tables, out_dir=out_dir))

    if binning_values:
        bins = np.linspace(
            min(np.min(v) for v in binning_values.values()),
            max(np.max(v) for v in binning_values.values()),
            8,
        )
        written.append(
            binned_statistic_plot(
                tables, "g", binning_values, bins, binning_label, "Mean ellipticity",
                out_dir=out_dir, filename="ellipticity_vs_binning.pdf",
            )
        )
        written.append(
            binned_statistic_plot(
                tables, "rho4", binning_values, bins, binning_label, r"$\rho_4$",
                out_dir=out_dir, filename="rho4_vs_binning.pdf",
            )
        )
    else:
        print("[plotting] binning_values not provided, skipping magnitude/size-binned plots")

    if not skip_morph and all("Gini" in t.colnames for t in tables.values()):
        written.append(gini_m20_panels(tables, out_dir=out_dir))
        written.append(M_I_panels(tables, out_dir=out_dir))
        written.append(M_D_panels(tables, out_dir=out_dir))
        written.append(mid_distributions(tables, out_dir=out_dir))
        written.append(cas_distributions(tables, out_dir=out_dir))
    else:
        print("[plotting] morphological (Gini/M20/CAS/MID) columns not found, skipping those plots")

    if reference_name is not None and reference_name in tables:
        names = paired_names if paired_names is not None else [n for n in tables if n != reference_name]
        for name in names:
            if name == reference_name or name not in tables:
                continue
            tab = tables[name]
            if len(tab) != len(tables[reference_name]):
                print("[plotting] %s and %s have different lengths, skipping paired error plots" % (reference_name, name))
                continue
            written.extend(
                paired_reconstruction_error(
                    tables[reference_name], tab,
                    reference_name=reference_name, other_name=name,
                    pixel_scale=pixel_scale, out_dir=out_dir,
                    filename_prefix="%s_vs_%s_" % (reference_name, name),
                )
            )

    return written
