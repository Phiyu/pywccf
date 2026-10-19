"""Amplitude-ratio maps and AR--mass figures (former ``func3.py``)."""
from ..io import load_combined_h5, combined_h5_inventory
from ..ratio import mass_bin_centers as _mass_bin_centers
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.style.use('default')
import numpy as np

WEB_TYPES = ("knot", "filament", "sheet", "void")
DIRECTIONS = (
    ("e1", (slice(None), 0, 0), "o"),
    ("e2", (0, slice(None), 0), "s"),
    ("e3", (0, 0, slice(None)), "^"),
)


def _mean_ratio_and_error(data_high, data_low, index):
    xi_high = data_high["xi"][index] + 1.0
    xi_low = data_low["xi"][index] + 1.0
    xi_high_sd = data_high["xi_sd"][index]
    xi_low_sd = data_low["xi_sd"][index]

    valid = (
        np.isfinite(xi_high)
        & np.isfinite(xi_low)
        & np.isfinite(xi_high_sd)
        & np.isfinite(xi_low_sd)
        & (xi_low > 0.0)
    )
    if not np.any(valid):
        return np.nan, np.nan

    xi_high = xi_high[valid]
    xi_low = xi_low[valid]
    xi_high_sd = xi_high_sd[valid]
    xi_low_sd = xi_low_sd[valid]

    ratios = xi_high / xi_low
    # Diagonal delta-method propagation. The combined HDF5 file does not
    # contain numerator/denominator or inter-bin covariances for a full error.
    ratio_var = (
        (xi_high_sd / xi_low) ** 2
        + (xi_high * xi_low_sd / xi_low**2) ** 2
    )
    ratio_mean = np.mean(ratios)
    ratio_mean_error = np.sqrt(np.sum(ratio_var)) / ratio_var.size
    return ratio_mean, ratio_mean_error


def plot_ratio(h5_file, sample="age"):
    mass_bin_centers = _mass_bin_centers(h5_file)   # open last bin -> median
    mass_bins = len(mass_bin_centers)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, web in zip(axes.flat, WEB_TYPES):
        bs = {direction: [] for direction, _, _ in DIRECTIONS}
        b_errors = {direction: [] for direction, _, _ in DIRECTIONS}

        for mass_bin_index in range(mass_bins):
            data_high = load_combined_h5(
                h5_file,
                dimension="3d",
                sample=sample,
                tail="high",
                web=web,
                mass_bin_index=mass_bin_index,
            )
            data_low = load_combined_h5(
                h5_file,
                dimension="3d",
                sample=sample,
                tail="low",
                web=web,
                mass_bin_index=mass_bin_index,
            )

            for direction, index, _ in DIRECTIONS:
                b, b_error = _mean_ratio_and_error(
                    data_high, data_low, index
                )
                bs[direction].append(b)
                b_errors[direction].append(b_error)

        for direction, _, marker in DIRECTIONS:
            values = np.asarray(bs[direction])
            errors = np.asarray(b_errors[direction])
            axis.plot(
                mass_bin_centers,
                values,
                marker=marker,
                label=rf"$e_{{{direction[-1]}}}$",
            )
            axis.fill_between(
                mass_bin_centers,
                values - errors,
                values + errors,
                alpha=0.2,
            )
        axis.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.5)
        axis.set_ylim(0.5, 2.0)
        axis.set_title(web.capitalize())
        axis.grid(alpha=0.25)

    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$b_1^{\rm %s}$" % sample)
    axes[0, 0].legend()
    plt.show()
    return figure, axes


def _snr_weight(numerator, denominator, denominator_sd, threshold, sharpness):
    """Soft reliability weight of a ratio bin, in [0, 1].

        w = 1 / (1 + (threshold * sigma_den / |xi_den|)^sharpness)

    w = 0.5 exactly at |xi_den| = threshold * sigma_den (the old hard cut),
    -> 1 for well-measured denominators and -> 0 near zero crossings.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    denominator_sd = np.asarray(denominator_sd, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = float(threshold) * denominator_sd / np.abs(denominator)
        weight = 1.0 / (1.0 + x ** float(sharpness))
    weight[~np.isfinite(weight) | ~np.isfinite(numerator)] = 0.0
    return weight


def _gaussian_kernel(sigma):
    half = max(int(np.ceil(3.0 * sigma)), 1)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _blur2d(values, sigma):
    """Separable Gaussian blur in bin units; symmetric padding reproduces the
    folding about r = 0 at the inner edge."""
    kernel = _gaussian_kernel(sigma)
    half = len(kernel) // 2
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (half, half)
        padded = np.pad(values, pad, mode="symmetric")
        values = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"), axis, padded
        )
    return values


def _fill_low_snr(ratio, weight, sigma, numerator, denominator, max_iter=5):
    """Blend each bin with a smooth local amplitude:

        R = w * xi_num/xi_den + (1 - w) * B,
        B = G*(xi_num xi_den) / G*(xi_den^2).

    B is the least-squares amplitude of xi_num = B xi_den inside a Gaussian
    neighbourhood -- the local version of the projection used by
    ``fit_anisotropic_A``. It never diverges, and bins with xi_den ~ 0 carry
    ~zero weight in it automatically, so the biased flanks of a zero crossing
    do not leak into the fill. Where w ~ 1 the data are returned untouched.
    Holes wider than the kernel are bridged by doubling sigma; anything still
    empty takes the projection amplitude of the whole panel.
    """
    good = np.isfinite(numerator) & np.isfinite(denominator)
    num = np.where(good, numerator, 0.0)
    den = np.where(good, denominator, 0.0)
    cross, power = num * den, den**2
    if float(power.sum()) <= 0.0:
        return np.full_like(ratio, np.nan)
    background = np.full_like(ratio, np.nan)
    if sigma and sigma > 0.0:
        floor = 1.0e-6 * float(power.max())
        current = float(sigma)
        for _ in range(int(max_iter)):
            support = _blur2d(power, current)
            fill = np.isnan(background) & (support > floor)
            background[fill] = _blur2d(cross, current)[fill] / support[fill]
            if not np.isnan(background).any():
                break
            current *= 2.0
    background[np.isnan(background)] = float(cross.sum()) / float(power.sum())
    ratio = np.where(weight > 0.0, ratio, 0.0)
    return weight * ratio + (1.0 - weight) * background


def _mirror_quadrants(values):
    return np.block([
        [values[::-1, ::-1], values[::-1, :]],
        [values[:, ::-1], values],
    ])


def _plot_ratio_3d_contour(
    h5_file,
    mass_bin_index,
    sample,
    *,
    numerator_tail,
    denominator_sample,
    denominator_tail,
    ratio_label,
    title,
    skip_underflow=True,
    mirror=True,
    cmap="coolwarm",
    vlim=None,
    vcenter=1.0,
    levels=13,
    regularize=True,
    snr_threshold=5.0,
    snr_sharpness=4.0,
    smooth_sigma=1.5,
    mark_filled=False,
    save_path=None,
    show=True,
):
    panel_data = {}
    for web in WEB_TYPES:
        numerator_data = load_combined_h5(
            h5_file, dimension="3d", sample=sample, tail=numerator_tail,
            web=web, mass_bin_index=mass_bin_index,
        )
        denominator_data = load_combined_h5(
            h5_file, dimension="3d", sample=denominator_sample,
            tail=denominator_tail,
            web=web, mass_bin_index=mass_bin_index,
        )
        for num_axis, den_axis in zip(
            numerator_data["centers"], denominator_data["centers"]
        ):
            if not np.array_equal(num_axis, den_axis):
                raise ValueError(f"numerator/denominator grids differ for {web}")

        numerator = numerator_data["xi"]
        denominator = denominator_data["xi"]
        weight = _snr_weight(
            numerator, denominator, denominator_data["xi_sd"],
            snr_threshold, snr_sharpness,
        )
        ratio = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=ratio, where=weight > 0.0)

        r1, r2, r3 = numerator_data["centers"]
        slicers = (
            ((slice(None), slice(None), 0), r1, r2),
            ((slice(None), 0, slice(None)), r1, r3),
            ((0, slice(None), slice(None)), r2, r3),
        )
        views = []
        for index, x, y in slicers:
            values, view_weight = ratio[index], weight[index]
            view_num, view_den = numerator[index], denominator[index]
            if skip_underflow:
                values, view_weight = values[1:, 1:], view_weight[1:, 1:]
                view_num, view_den = view_num[1:, 1:], view_den[1:, 1:]
                x, y = x[1:], y[1:]
            if regularize:
                values = _fill_low_snr(
                    values, view_weight, smooth_sigma, view_num, view_den
                )
            else:
                values = np.where(view_weight >= 0.5, values, np.nan)
            if mirror:
                values = _mirror_quadrants(values)
                view_weight = _mirror_quadrants(view_weight)
                x, y = np.r_[-x[::-1], x], np.r_[-y[::-1], y]
            views.append((values, view_weight, x, y))
        panel_data[web] = views

    figure, axes = plt.subplots(
        4, 3, figsize=(12, 15), squeeze=False, constrained_layout=True
    )
    titles = (
        r"$(\mathbf{e}_1,\mathbf{e}_2)$ plane; $r_3$ fixed",
        r"$(\mathbf{e}_1,\mathbf{e}_3)$ plane; $r_2$ fixed",
        r"$(\mathbf{e}_2,\mathbf{e}_3)$ plane; $r_1$ fixed",
    )
    labels = (
        (r"$r_1$ [$h^{-1}{\rm Mpc}$]", r"$r_2$ [$h^{-1}{\rm Mpc}$]"),
        (r"$r_1$ [$h^{-1}{\rm Mpc}$]", r"$r_3$ [$h^{-1}{\rm Mpc}$]"),
        (r"$r_2$ [$h^{-1}{\rm Mpc}$]", r"$r_3$ [$h^{-1}{\rm Mpc}$]"),
    )
    for row, web in enumerate(WEB_TYPES):
        finite = np.concatenate([
            values[np.isfinite(values)]
            for values, _, _, _ in panel_data[web]
        ])
        if vlim is None:
            if finite.size:
                row_vlim = (
                    min(float(np.min(finite)), vcenter - 1e-6),
                    max(float(np.max(finite)), vcenter + 1e-6),
                )
            else:
                row_vlim = (vcenter - 0.1, vcenter + 0.1)
        else:
            row_vlim = tuple(map(float, vlim))
        if len(row_vlim) != 2 :
            raise ValueError("vlim must satisfy vmin < 1 < vmax")
        n_side = max(int(levels) // 2, 1)
        row_levels = np.r_[
            np.linspace(row_vlim[0], vcenter, n_side + 1),
            np.linspace(vcenter, row_vlim[1], n_side + 1)[1:],
        ]
        norm = TwoSlopeNorm(
            vmin=row_vlim[0], vcenter=vcenter, vmax=row_vlim[1]
        )

        for col, (values, view_weight, x, y) in enumerate(panel_data[web]):
            axis = axes[row, col]
            axis.set_facecolor("white")
            plot_values = np.ma.masked_invalid(values.T)
            artist = axis.contourf(
                x, y, plot_values,
                levels=row_levels, cmap=cmap, norm=norm, extend="both",
            )
            valid_values = plot_values.compressed()
            if (valid_values.size
                    and np.min(valid_values) < 1.0 < np.max(valid_values)):
                axis.contour(
                    x, y, plot_values,
                    levels=[1.0], colors="k", linewidths=1.0,
                )
            if (mark_filled and regularize
                    and np.min(view_weight) < 0.5 < np.max(view_weight)):
                axis.contour(
                    x, y, view_weight.T, levels=[0.5],
                    colors="0.3", linewidths=0.7, linestyles=":",
                )
            axis.set_aspect("equal")
            axis.set_xlabel(labels[col][0])
            axis.set_ylabel(labels[col][1])
            if not finite.size:
                axis.text(
                    0.5, 0.5, r"all bins masked ($\xi_{\rm den}<3\sigma$)",
                    transform=axis.transAxes, ha="center", va="center",
                )
            if mirror:
                axis.axhline(0.0, color="k", lw=0.5, alpha=0.4)
                axis.axvline(0.0, color="k", lw=0.5, alpha=0.4)
            if row == 0:
                axis.set_title(titles[col])
        axes[row, 0].annotate(
            web.capitalize(), (-0.25, 0.5), xycoords="axes fraction",
            ha="center", va="center", rotation=90, fontweight="bold",
        )
        colorbar = figure.colorbar(
            artist, ax=axes[row, :], shrink=0.88, pad=0.018
        )
        colorbar.set_label(ratio_label)

    edges = combined_h5_inventory(h5_file)["mass_bin_edges"]
    mass_label = (
        f"{edges[mass_bin_index]:g}--{edges[mass_bin_index + 1]:g}"
        if np.isfinite(edges[mass_bin_index + 1])
        else f">={edges[mass_bin_index]:g}"
    )
    # figure.suptitle(
    #     f"{sample}: {title}, mass bin {mass_label}",
    #     fontweight="bold",
    # )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_ratio_3d_contour(
    h5_file, mass_bin_index, sample="age", **plot_kwargs
):
    """Plot xi_high / xi_low in the three orthogonal planes.

    Near zero crossings of xi_low the ratio diverges. With ``regularize=True``
    (default) each bin gets a soft weight

        w = 1 / (1 + (snr_threshold * sigma_low / |xi_low|)^snr_sharpness),

    (w = 0.5 exactly at |xi_low| = snr_threshold * sigma_low; default 5) and
    is displayed as

        R = w * xi_high/xi_low + (1 - w) * B,

    where B = G*(xi_high xi_low) / G*(xi_low^2) is the local least-squares
    amplitude in a Gaussian neighbourhood (``smooth_sigma`` in bins), i.e. the
    local version of the projection used by ``fit_anisotropic_A``. B never
    diverges and gives ~zero weight to bins with xi_low ~ 0. Well-measured bins
    are shown unchanged; unreliable ones are filled instead of left blank.
    ``mark_filled=True`` outlines the w = 0.5 boundary with a dotted line.
    ``regularize=False`` restores the hard mask, blank wherever
    |xi_low| < snr_threshold * sigma_low (pass ``snr_threshold=3.0`` to get the
    previous figure back).

    Fills cannot remove a genuine sign change: where xi_high and xi_low cross
    zero at different separations (e.g. the void boundary along e3) the ratio
    really does run from large positive to <= 0, and that structure remains.
    """
    return _plot_ratio_3d_contour(
        h5_file,
        mass_bin_index,
        sample,
        numerator_tail="high",
        denominator_sample=sample,
        denominator_tail="low",
        ratio_label=r"$\xi_{\rm old}/\xi_{\rm young}$",
        title="high/low ratio",
        **plot_kwargs,
    )


def plot_ratio_3d_contour2(
    h5_file, mass_bin_index, sample="age", sub="high", **plot_kwargs
):
    """Plot xi_sub / xi_total at one mass bin."""
    if sub not in ("high", "low"):
        raise ValueError("sub must be 'high' or 'low'")
    return _plot_ratio_3d_contour(
        h5_file,
        mass_bin_index,
        sample,
        numerator_tail=sub,
        denominator_sample="total",
        denominator_tail="all",
        ratio_label=rf"$\xi_{{\rm {sub}}}/\xi_{{\rm total}}$",
        title=f"{sub}/total ratio",
        **plot_kwargs,
    )


def plot_ratio2(h5_file, sample="age"):
    mass_bin_centers = _mass_bin_centers(h5_file)   # open last bin -> median
    mass_bins = len(mass_bin_centers)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, web in zip(axes.flat, WEB_TYPES):
        bs = {
            tail: {direction: [] for direction, _, _ in DIRECTIONS}
            for tail in ("high", "low")
        }
        b_errors = {
            tail: {direction: [] for direction, _, _ in DIRECTIONS}
            for tail in ("high", "low")
        }

        for mass_bin_index in range(mass_bins):
            data_total = load_combined_h5(
                h5_file,
                dimension="3d",
                sample="total",
                tail="all",
                web=web,
                mass_bin_index=mass_bin_index,
            )
            for tail in ("high", "low"):
                data_tail = load_combined_h5(
                    h5_file,
                    dimension="3d",
                    sample=sample,
                    tail=tail,
                    web=web,
                    mass_bin_index=mass_bin_index,
                )
                for direction, index, _ in DIRECTIONS:
                    b, b_error = _mean_ratio_and_error(
                        data_tail, data_total, index
                    )
                    bs[tail][direction].append(b)
                    b_errors[tail][direction].append(b_error)

        for direction_index, (direction, _, marker) in enumerate(DIRECTIONS):
            color = f"C{direction_index}"
            for tail, linestyle, markerfacecolor in (
                ("high", "-", color),
                ("low", "--", "none"),
            ):
                values = np.asarray(bs[tail][direction])
                errors = np.asarray(b_errors[tail][direction])
                axis.plot(
                    mass_bin_centers,
                    values,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markerfacecolor=markerfacecolor,
                    label=rf"{tail}/total $e_{{{direction[-1]}}}$",
                )
                axis.fill_between(
                    mass_bin_centers,
                    values - errors,
                    values + errors,
                    color=color,
                    alpha=0.12,
                )

        axis.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.5)
        axis.set_ylim(0.7, 1.4)
        axis.set_title(web.capitalize())
        axis.grid(alpha=0.25)

    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$b_1^{\rm %s/total}$" % sample)
    axes[0, 0].legend(ncol=2, fontsize=8)
    plt.show()
    return figure, axes


def _safe_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def plot_anisotropic_A(
    result,
    sample="age",
    *,
    r_range=(5.0, 20.0),
    ylim=None,
    figsize=(12, 8),
    annotate_residual=True,
    save_path=None,
    show=True,
    **fit_kwargs,
):
    """Companion of ``plot_ratio2`` for A = b_high / b_low.

    One panel per web type, mass on the x axis, one coloured curve per tidal
    eigen-direction with its 1-sigma band, and -- when the fit was run with
    ``include_cube=True`` -- the direction-averaged amplitude as a black
    dashed line. A flat set of curves lying on top of the black line means
    the relative assembly bias is a single scale- and direction-independent
    number; separation between the coloured curves is the anisotropy.

    ``result`` is either a ``fit_anisotropic_A`` output (fit once, replot
    cheaply) or an HDF5 path. ``r_range`` may also be a dictionary such as
    ``{"inner": (3, 9), "outer": (10, 20)}``; every range is then fitted
    and drawn with a different linestyle.

    ``ylim`` is None (each panel autoscales), a ``(lo, hi)`` tuple applied to
    all four panels, or a ``{web: (lo, hi)}`` dictionary.
    """
    from ..ratio import fit_anisotropic_A, DIRECTION_COLORS
    from matplotlib.lines import Line2D

    multiple_ranges = isinstance(r_range, dict)
    if multiple_ranges:
        if not r_range:
            raise ValueError("r_range dictionary cannot be empty")
        h5_file = (
            result.get("h5_file") if isinstance(result, dict) else result
        )
        if h5_file is None:
            raise ValueError("result must contain h5_file for multiple r_range")
        fit_sample = (
            result.get("sample", sample)
            if isinstance(result, dict) else sample
        )
        linestyles = ("-", "--", "-.", ":")
        results = []
        for index, (name, fit_range) in enumerate(r_range.items()):
            fitted = fit_anisotropic_A(
                h5_file, fit_sample, r_range=fit_range, **fit_kwargs
            )
            r_lo, r_hi = fitted["r_range"]
            label = f"{name}: {r_lo:g}-{r_hi:g}"
            linestyle = linestyles[index % len(linestyles)]
            results.append((label, linestyle, fitted))
    else:
        if not isinstance(result, dict):
            result = fit_anisotropic_A(
                result, sample, r_range=r_range, **fit_kwargs
            )
        results = [("", "-", result)]

    base_result = results[0][2]
    upper, lower = (
        ("old", "young") if base_result.get("sample", sample) == "age"
        else ("high", "low")
    )
    centers = base_result["mass_bin_centers"]
    markers = {"e1": "o", "e2": "s", "e3": "^"}

    figure, axes = plt.subplots(
        2, 2, figsize=figsize, sharex=True,
        constrained_layout=True,
    )
    for axis, web in zip(axes.flat, base_result["webs"]):
        for _, linestyle, fitted in results:
            for direction in fitted["directions"]:
                values = fitted["A"][web][direction]
                errors = fitted["A_err"][web][direction]
                color = DIRECTION_COLORS.get(direction, "0.4")
                axis.plot(
                    centers, values, color=color, linestyle=linestyle,
                    marker=markers.get(direction, "o"), ms=5,
                    label=(rf"$\mathbf{{e}}_{{{direction[-1]}}}$"
                           if not multiple_ranges else None),
                )
                axis.fill_between(
                    centers, values - errors, values + errors,
                    color=color, alpha=0.08 if multiple_ranges else 0.15,
                    lw=0,
                )
            if fitted.get("include_cube"):
                values = fitted["A"][web]["cube"]
                errors = fitted["A_err"][web]["cube"]
                axis.plot(
                    centers, values, color="k",
                    ls=linestyle if multiple_ranges else "--", lw=1.4,
                    marker="D", ms=4,
                    label=None if multiple_ranges else "all 3D bins",
                )
                axis.fill_between(
                    centers, values - errors, values + errors,
                    color="k", alpha=0.07 if multiple_ranges else 0.10,
                    lw=0,
                )
        axis.axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.6)
        if type(ylim) is dict:
            axis.set_ylim(ylim[web])
        axis.grid(alpha=0.25)
        text_box = dict(
            boxstyle="round,pad=0.25", facecolor="white",
            edgecolor="none", alpha=0.8,
        )
        axis.text(
            0.03, 0.96, web.capitalize(), transform=axis.transAxes,
            ha="left", va="top", fontsize=12, fontweight="bold",
            bbox=text_box, zorder=5,
        )
        if annotate_residual:
            residuals = []
            for label, _, fitted in results:
                values = ",  ".join(
                    rf"$\mathbf{{e}}_{{{direction[-1]}}}$="
                    rf"{_safe_median(fitted['shape_residual'][web][direction]):.2f}"
                    for direction in fitted["directions"]
                )
                residuals.append(f"{label}: {values}" if label else values)
            text = "median shape residual\n" + "\n".join(residuals)
            axis.text(
                0.03, 0.86, text, transform=axis.transAxes,
                ha="left", va="top", fontsize=7.5,
                bbox=text_box, zorder=5,
            )

    if isinstance(ylim, (tuple, list)):
        for axis in axes.flat:
            axis.set_ylim(*ylim)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    for axis in axes[:, 0]:
        axis.set_ylabel(
            rf"$A=\xi_{{\rm Wh}}^{{\rm {upper}}}/\xi_{{\rm Wh}}^{{\rm {lower}}}$"
        )
    if multiple_ranges:
        handles = [
            Line2D(
                [], [], color=DIRECTION_COLORS.get(direction, "0.4"),
                marker=markers.get(direction, "o"), linestyle="none",
                label=rf"$\mathbf{{e}}_{{{direction[-1]}}}$",
            )
            for direction in base_result["directions"]
        ]
        if base_result.get("include_cube"):
            handles.append(Line2D(
                [], [], color="k", marker="D", linestyle="none",
                label="all 3D bins",
            ))
        handles.extend(
            Line2D([], [], color="0.25", linestyle=style, label=label)
            for label, style, _ in results
        )
        figure.legend(
            handles=handles, fontsize=8, ncol=4, frameon=False,
            loc="outside lower center",
        )
        range_text = "multiple radial ranges"
    else:
        axes[0, 0].legend(fontsize=8, ncol=2, loc="upper right")
        r_lo, r_hi = base_result["r_range"]
        range_text = (
            rf"fitted over ${r_lo:g}\leq r\leq{r_hi:g}\ h^{{-1}}$Mpc"
        )

    # figure.suptitle(
    #     f"{base_result['sample']}: anisotropic relative assembly bias "
    #     rf"$A=b^{{\rm high}}/b^{{\rm low}}$, {range_text}",
    #     fontweight="bold",
    # )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_anisotropic_A2(
    result,
    sample="age",
    *,
    r_range=(5.0, 20.0),
    ylim=(0.7, 1.6),
    figsize=(12, 8),
    annotate_residual=True,
    save_path=None,
    show=True,
    **fit_kwargs,
):
    """Companion of ``plot_ratio2`` for A^S = b^S / b^total.

    Both tails share a panel: one colour per tidal eigen-direction, solid
    line with filled markers for ``high``, dashed line with open markers for
    ``low``, each with its 1-sigma band. When the fit carries the cube key,
    the direction-averaged amplitude of each tail is drawn in black with the
    same solid/dashed coding.

    Reading it: the vertical gap between a solid and its dashed partner is
    the assembly-bias signal in that direction, while their position relative
    to unity says which tail actually moves away from the full population.

    ``result`` is either a ``fit_anisotropic_A2`` output or an HDF5 path, in
    which case the fit is run here with ``sample``, ``r_range`` and any extra
    keyword arguments.
    """
    from ..ratio import fit_anisotropic_A2, DIRECTION_COLORS

    if not isinstance(result, dict):
        result = fit_anisotropic_A2(
            result, sample, r_range=r_range, **fit_kwargs
        )

    centers = result["mass_bin_centers"]
    markers = {"e1": "o", "e2": "s", "e3": "^"}
    styles = {"high": ("-", True), "low": ("--", False)}

    figure, axes = plt.subplots(
        2, 2, figsize=figsize, sharex=True, sharey=True,
        constrained_layout=True,
    )
    for axis, web in zip(axes.flat, result["webs"]):
        for tail in result["tails"]:
            linestyle, filled = styles.get(tail, ("-", True))
            for direction in result["directions"]:
                values = result["A"][tail][web][direction]
                errors = result["A_err"][tail][web][direction]
                color = DIRECTION_COLORS.get(direction, "0.4")
                axis.plot(
                    centers, values, color=color, linestyle=linestyle,
                    marker=markers.get(direction, "o"), ms=5,
                    markerfacecolor=color if filled else "none",
                    label=rf"{tail} $\mathbf{{e}}_{{{direction[-1]}}}$",
                )
                axis.fill_between(
                    centers, values - errors, values + errors,
                    color=color, alpha=0.12, lw=0,
                )
            if result.get("include_cube"):
                values = result["A"][tail][web]["cube"]
                axis.plot(
                    centers, values, color="k", linestyle=linestyle, lw=1.3,
                    marker="D", ms=4,
                    markerfacecolor="k" if filled else "none",
                    label=f"{tail} all 3D bins",
                )
        axis.axhline(1.0, color="k", ls=":", lw=1.0, alpha=0.6)
        axis.set_title(web.capitalize())
        axis.grid(alpha=0.25)
        if annotate_residual:
            text = "median shape residual (high, low)\n" + ",  ".join(
                rf"$\mathbf{{e}}_{{{direction[-1]}}}$="
                + "/".join(
                    f"{_safe_median(result['shape_residual'][tail][web][direction]):.2f}"
                    for tail in result["tails"]
                )
                for direction in result["directions"]
            )
            axis.text(
                0.03, 0.96, text, transform=axis.transAxes,
                ha="left", va="top", fontsize=7,
            )

    if ylim is not None:
        axes[0, 0].set_ylim(*ylim)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\log_{10}(M/M_\odot)$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$A^{S}=b^{S}/b^{\rm total}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, fontsize=8, ncol=4, frameon=False,
        loc="outside lower center",
    )

    r_lo, r_hi = result["r_range"]
    figure.suptitle(
        f"{result['sample']}: anisotropic relative bias of each tail "
        rf"$A^{{S}}=b^{{S}}/b^{{\rm total}}$, "
        rf"fitted over ${r_lo:g}\leq r\leq{r_hi:g}\ h^{{-1}}$Mpc",
        fontweight="bold",
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes
