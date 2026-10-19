"""Publication figures for the halo--cosmic-web correlation analysis.

Project-wide plotting conventions
---------------------------------
* Draw uncertainties with ``fill_between`` rather than error-bar caps.
* Use a tight layout and save with ``bbox_inches="tight"``.
* Do not add a figure-level title.
* Paper style (skill "paper-figure-style"; applied to the plot_1_main_*
  figures): 1.6 pt frames, large text, figure legend ABOVE the panels in
  <= 2 rows, inward major + minor ticks on all four sides, grid at major
  ticks only.
* Notation: log (not log_10), M_h for halo mass, delta for density,
  xi_3D(r) along e_i and xi_1D(r).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import functools
import math
import re

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import (AutoMinorLocator, FixedLocator, LogLocator,
                               NullFormatter)

from ..io import combined_h5_inventory, load_combined_h5


WEB_TYPES = ("knot", "filament", "sheet", "void")
XI_3D_LABEL = r"$\xi_{\rm 3D}(\mathbf{r})$"
XI_1D_LABEL = r"$\xi_{\rm 1D}(r)$"
DIRECTION_TITLES = (
    r"$\xi_{\rm 3D}(\mathbf{r})$ along $\mathbf{e}_1$",
    r"$\xi_{\rm 3D}(\mathbf{r})$ along $\mathbf{e}_2$",
    r"$\xi_{\rm 3D}(\mathbf{r})$ along $\mathbf{e}_3$",
)
DIRECTION_XLABELS = (
    r"$r_1\,[h^{-1}\mathrm{Mpc}]$",
    r"$r_2\,[h^{-1}\mathrm{Mpc}]$",
    r"$r_3\,[h^{-1}\mathrm{Mpc}]$",
)

# A deliberately muted blue--purple sequence for mass-bin encoding.
MASS_CMAP = LinearSegmentedColormap.from_list(
    "muted_blue_purple",
    ("#2A7B9B", "#3E9C91", "#57C785", "#A6D36B", "#EDDD53"),
)

STATISTIC_COLORS = ('#f33b1f', '#00ac38', '#5e76ff', "0.15")
STATISTIC_MARKERS = ("o", "s", "^", "x")
STATISTIC_LABELS = DIRECTION_TITLES + (XI_1D_LABEL,)


# ---------------------------------------------------------------------------
# paper figure style
# ---------------------------------------------------------------------------
PAPER_RC = {
    "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 14,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12,
    "axes.linewidth": 1.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 6.0, "ytick.major.size": 6.0,
    "xtick.minor.size": 3.2, "ytick.minor.size": 3.2,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
    "grid.alpha": 0.25, "grid.linewidth": 0.7,
    "legend.frameon": False,
}


def style_axis(ax):
    """Paper tick/grid/frame rules on one axis, whatever its scale.  Call
    after the scale, limits and any FixedLocator minor ticks are set."""
    for axis, scale in ((ax.xaxis, ax.get_xscale()),
                        (ax.yaxis, ax.get_yscale())):
        if scale == "linear" and not isinstance(
            axis.get_minor_locator(), FixedLocator
        ):
            axis.set_minor_locator(AutoMinorLocator())
        elif scale == "log":
            axis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
            axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=PAPER_RC["xtick.labelsize"])
    ax.tick_params(which="major", length=6.0, width=1.3)
    ax.tick_params(which="minor", length=3.2, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(PAPER_RC["axes.linewidth"])
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.grid(False, which="minor")


def top_legend(figure, handles, labels=None, max_rows=2, **kw):
    """Figure legend above the panels: one row when it fits the figure
    width, otherwise the fewest rows (<= max_rows) that do.  Call after
    tight_layout; savefig(bbox_inches="tight") keeps it."""
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    renderer = figure.canvas.get_renderer()
    legend = None
    for rows in range(1, max(1, max_rows) + 1):
        if legend is not None:
            legend.remove()
        legend = figure.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            ncol=max(1, math.ceil(len(labels) / rows)), frameon=False,
            columnspacing=1.4, handlelength=2.2,
            fontsize=PAPER_RC["legend.fontsize"], **kw,
        )
        if legend.get_window_extent(renderer).width <= figure.bbox.width:
            break
    return legend


def _paper_figure(func):
    """Run a figure function inside the paper rc context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(PAPER_RC):
            return func(*args, **kwargs)
    return wrapper


def _finish_figure(figure, handles, save_path, show):
    """Style every axis, tight layout, legend on top, save, show."""
    for axis in figure.axes:
        style_axis(axis)
    figure.tight_layout(pad=0.8)
    top_legend(figure, handles)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()


def _available_total_mass_bins(h5_file: Path) -> tuple[int, ...]:
    """Return 3D total-sample mass bins shared by all four web types."""
    sets = []
    with h5py.File(h5_file, "r") as stream:
        for web in WEB_TYPES:
            group_path = f"data/3d/total/all/{web}"
            if group_path not in stream:
                raise KeyError(
                    f"{h5_file} has no total 3D group /{group_path}"
                )
            indices = {
                int(name.removeprefix("mass_bin_"))
                for name in stream[group_path]
                if name.startswith("mass_bin_")
            }
            sets.append(indices)

    shared = set.intersection(*sets) if sets else set()
    if not shared:
        raise ValueError(
            f"{h5_file} contains no total 3D mass bin shared by all web types"
        )
    return tuple(sorted(shared))


def _normalize_mass_bins(
    h5_file: Path,
    mass_bins: int | Iterable[int] | None,
) -> tuple[int, ...]:
    available = _available_total_mass_bins(h5_file)
    if mass_bins is None:
        return available
    if isinstance(mass_bins, (int, np.integer)):
        selected = (int(mass_bins),)
    else:
        selected = tuple(dict.fromkeys(int(index) for index in mass_bins))
    if not selected:
        raise ValueError("mass_bins must contain at least one index")

    missing = [index for index in selected if index not in available]
    if missing:
        raise ValueError(
            f"Requested mass bins {missing} are unavailable; "
            f"available total 3D bins are {list(available)}"
        )
    return selected


def _directional_slice(
    xi: np.ndarray,
    xi_sd: np.ndarray,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice one tidal axis while fixing the other axes at index zero."""
    if direction == 0:
        return xi[:, 0, 0], xi_sd[:, 0, 0]
    if direction == 1:
        return xi[0, :, 0], xi_sd[0, :, 0]
    if direction == 2:
        return xi[0, 0, :], xi_sd[0, 0, :]
    raise ValueError(f"direction must be 0, 1, or 2; got {direction}")


def _mass_label(container_attrs: dict, fallback_index: int) -> str:
    lower = container_attrs.get("mass_bin_lower_log10_msun_h")
    upper = container_attrs.get("mass_bin_upper_log10_msun_h")
    if lower is None or upper is None:
        return f"mass bin {fallback_index}"
    if not np.isfinite(float(upper)):                 # open-ended bin
        return (
            rf"$\log(M_{{\rm h}}/h^{{-1}}{{\rm M}}_\odot)"
            rf"\geq {float(lower):g}$"
        )
    return (
        rf"${float(lower):g}\leq\log(M_{{\rm h}}/h^{{-1}}{{\rm M}}_\odot)"
        rf"<{float(upper):g}$"
    )


@_paper_figure
def plot_1_main_xi(
    h5_file,
    mass_bins=None,
    *,
    skip_underflow=True,
    xscale="linear",
    yscale="split",
    xlim=None,
    ylim=None,
    ylim_by_web="default",
    linthresh=0.1,
    neg_fraction=None,
    figsize=(15, 14),
    uncertainty_alpha=0.18,
    save_path=None,
    show=True,
):
    """Plot total-sample directional 3D-WCCFs from one combined HDF5 file.

    The output follows the original ``func.py::plot_xi_3d_directions``
    layout: rows are knot, filament, sheet, and void cells; columns are
    slices along the three ordered tidal eigenvectors.  Along each slice,
    the other two separation components remain fixed at their index-zero
    (smallest-separation/underflow) bins.

    Parameters
    ----------
    h5_file : path-like
        Combined HDF5 file readable by ``pywccf.io.load_combined_h5``.
    mass_bins : int, iterable of int, or None
        Mass-bin indices to draw. An integer selects one bin. ``None`` draws
        every total 3D mass bin shared by all four web types.
    skip_underflow : bool
        Omit the underflow bin only along the displayed separation axis.
    xlim, ylim : two-tuples or None
        Optional common limits for all panels.
    yscale : str
        "split" (default): logarithmic for xi > ``linthresh``, linear for
        xi < 0, the negative part taking ``neg_fraction`` of the height
        (default 1/3, module constant NEG_FRACTION); "symlog"; "linear";
        or any matplotlib scale.
    ylim_by_web : dict, "default" or None
        y-range of each web-type row for the "split" and "symlog" scales
        ("default" = ``XI_YLIM_BY_WEB``; None = from the data).  ``ylim``
        overrides it.  Not used with a linear scale.
    linthresh : float
        Start of the logarithmic part (default 0.1); [0, linthresh] is a
        short linear stretch.
    save_path : path-like or None
        If provided, save at 300 dpi with a tight bounding box.

    Returns
    -------
    figure, axes
        Matplotlib figure and the ``(4, 3)`` array of axes.
    """
    if neg_fraction is None:
        neg_fraction = NEG_FRACTION
    h5_file = Path(h5_file)
    if not h5_file.is_file():
        raise FileNotFoundError(h5_file)

    selected_bins = _normalize_mass_bins(h5_file, mass_bins)
    color_positions = np.linspace(0.08, 0.92, len(selected_bins))
    colors = {
        index: MASS_CMAP(position)
        for index, position in zip(selected_bins, color_positions)
    }

    figure, axes = plt.subplots(
        4,
        3,
        figsize=figsize,
        sharex="col",
        squeeze=False,
    )
    labels = {}

    for row, web in enumerate(WEB_TYPES):
        for column in range(3):
            axis = axes[row, column]
            for mass_bin_index in selected_bins:
                result = load_combined_h5(
                    h5_file,
                    dimension="3d",
                    sample="total",
                    tail="all",
                    web=web,
                    mass_bin_index=mass_bin_index,
                )
                xi = np.asarray(result["xi"], dtype=np.float64)
                xi_sd = np.asarray(result["xi_sd"], dtype=np.float64)
                radius = np.asarray(
                    result["centers"][column], dtype=np.float64
                )
                profile, profile_sd = _directional_slice(
                    xi, xi_sd, column
                )
                if skip_underflow:
                    radius = radius[1:]
                    profile = profile[1:]
                    profile_sd = profile_sd[1:]

                color = colors[mass_bin_index]
                axis.plot(
                    radius,
                    profile,
                    color=color,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.5,
                    label=f"mass bin {mass_bin_index}",
                    zorder=3,
                )
                axis.fill_between(
                    radius,
                    profile - profile_sd,
                    profile + profile_sd,
                    color=color,
                    alpha=uncertainty_alpha,
                    linewidth=0,
                    zorder=2,
                )
                labels[mass_bin_index] = _mass_label(
                    result["container_attrs"], mass_bin_index
                )

            axis.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
            axis.set_xscale(xscale)
            if yscale in ("split", "symlog"):
                web_ylim = None
                if ylim is not None:
                    web_ylim = tuple(ylim)
                elif ylim_by_web is not None:
                    web_ylim = (
                        XI_YLIM_BY_WEB if isinstance(ylim_by_web, str)
                        else ylim_by_web
                    ).get(web)
                _apply_xi_yscale(axis, yscale, web_ylim, linthresh,
                                 neg_fraction)
            else:
                axis.set_yscale(yscale)
                if ylim is not None:
                    axis.set_ylim(*ylim)
            if xlim is not None:
                axis.set_xlim(*xlim)
            axis.grid(alpha=0.22, linewidth=0.7)
            if row == 0:
                axis.set_title(DIRECTION_TITLES[column])
            if row == len(WEB_TYPES) - 1:
                axis.set_xlabel(DIRECTION_XLABELS[column])
            if column == 0:
                axis.set_ylabel(f"{web.capitalize()}\n" + XI_3D_LABEL)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[index],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=labels[index],
        )
        for index in selected_bins
    ]
    _finish_figure(figure, legend_handles, save_path, show)
    return figure, axes


def _mass_label_short(container_attrs: dict, fallback_index: int) -> str:
    lower = container_attrs.get("mass_bin_lower_log10_msun_h")
    upper = container_attrs.get("mass_bin_upper_log10_msun_h")
    if lower is None or upper is None:
        return f"mass bin {fallback_index}"
    if not np.isfinite(float(upper)):                 # open-ended bin
        return (
            rf"$\log(M_{{\rm h}}/h^{{-1}}{{\rm M}}_\odot)"
            rf"\geq {float(lower):g}$"
        )
    return (
        rf"$\log(M_{{\rm h}}/h^{{-1}}{{\rm M}}_\odot)\in"
        rf"[{float(lower):g},\,{float(upper):g})$"
    )


# y-range of each web type in plot_1_main_xi_by_mass (symlog axes)
XI_YLIM_BY_WEB = {
    "knot": (-1.2, 1.0e4),
    "filament": (-1.2, 1.0e3),
    "sheet": (-1.2, 20.0),
    "void": (-1.2, 2.0),
}


def _style_symlog_axis(axis, lower, upper, linthresh):
    """Symmetric-log y axis: linear for |xi| < linthresh, logarithmic
    beyond, so xi = -1 (no halo) and xi >> 1 fit on one axis.  Major ticks
    at -1, 0, 1 and the decades below ``upper``."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    axis.set_yscale("symlog", linthresh=linthresh)
    axis.set_ylim(lower, upper)
    ticks = [0.0]
    if lower <= -1.0:
        ticks.append(-1.0)
    if linthresh < 1.0:
        ticks += [linthresh] + ([-linthresh] if lower < -linthresh else [])
    decade = 1.0
    while decade <= upper * (1 + 1e-9):
        ticks.append(decade)
        decade *= 10.0

    def label(value, _):
        if value == 0:
            return "0"
        if abs(value) >= 10:
            power = int(round(np.log10(abs(value))))
            sign = "-" if value < 0 else ""
            return rf"${sign}10^{{{power}}}$"
        return f"{value:g}".replace("-", "\u2212")

    axis.yaxis.set_major_locator(FixedLocator(sorted(set(ticks))))
    axis.yaxis.set_major_formatter(FuncFormatter(label))
    minor = []                         # 2..9 x each decade beyond linthresh
    base = 10.0 ** np.floor(np.log10(linthresh))
    while base < max(upper, abs(lower)):
        for k in range(2, 10):
            value = k * base
            if value > linthresh and value < upper:
                minor.append(value)
            if value > linthresh and -value > lower:
                minor.append(-value)
        base *= 10.0
    axis.yaxis.set_minor_locator(FixedLocator(sorted(minor)))


# fraction of the height given to xi < 0 on "split" axes
NEG_FRACTION = 1.0 / 3.0


def _split_transform(lower, upper, linthresh, neg_fraction, linscale):
    """Forward / inverse maps of the "split" y axis.

    y < 0           : linear, [lower, 0] -> [-D, 0]
    0 <= y < y0     : linear, [0, y0] -> [0, linscale]        (y0 = linthresh)
    y >= y0         : logarithmic, linscale + log10(y / y0)
    D is chosen so that [lower, 0] takes ``neg_fraction`` of [lower, upper].
    """
    y0, c0 = float(linthresh), float(linscale)

    def positive(y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y < y0, y / y0 * c0,
                            c0 + np.log10(np.maximum(y, y0) / y0))

    pos_span = float(positive(np.asarray(float(upper))))
    depth = neg_fraction / (1.0 - neg_fraction) * pos_span
    scale_neg = depth / abs(float(lower))

    def forward(values):
        y = np.asarray(values, dtype=np.float64)
        return np.where(y < 0.0, y * scale_neg, positive(y))

    def inverse(values):
        t = np.asarray(values, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(
                t < 0.0, t / scale_neg,
                np.where(t < c0, t / c0 * y0,
                         y0 * 10.0 ** (np.minimum(t, 300.0) - c0)),
            )

    return forward, inverse


def _style_split_axis(axis, lower, upper, linthresh, neg_fraction=NEG_FRACTION,
                      linscale=0.5):
    """y axis that is logarithmic for xi > linthresh and linear below,
    with the negative part [lower, 0] taking ``neg_fraction`` of the axis
    height (default 1/3) whatever the decades above.  [0, linthresh] is a
    short linear stretch of ``linscale`` decades joining the two (log10 of
    xi -> 0 diverges).  Major ticks: linear ones on [lower, 0], linthresh
    and the decades above."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    lower, upper = float(lower), float(upper)
    if not lower < 0.0 < upper:
        raise ValueError(
            f"split y axis needs lower < 0 < upper, got ({lower}, {upper})"
        )
    if not 0.0 < neg_fraction < 1.0:
        raise ValueError("neg_fraction must lie in (0, 1)")
    forward, inverse = _split_transform(
        lower, upper, linthresh, neg_fraction, linscale
    )
    axis.set_yscale("function", functions=(forward, inverse))
    axis.set_ylim(lower, upper)

    negative = [
        tick for tick in MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10])
        .tick_values(lower, 0.0)
        if lower <= tick < 0.0
    ]
    ticks = negative + [0.0]
    if linthresh < upper:
        ticks.append(float(linthresh))
    decade = 10.0 ** np.ceil(np.log10(linthresh) + 1e-9)
    minor = []
    while decade <= upper * (1 + 1e-9):
        if decade > linthresh * (1 + 1e-9):
            ticks.append(decade)
        decade *= 10.0
    base = 10.0 ** np.floor(np.log10(linthresh))
    while base < upper:
        minor += [k * base for k in range(2, 10)
                  if linthresh < k * base < upper]
        base *= 10.0

    def label(value, _):
        if value == 0:
            return "0"
        if abs(value) >= 10:
            power = int(round(np.log10(abs(value))))
            sign = "-" if value < 0 else ""
            return rf"${sign}10^{{{power}}}$"
        return f"{value:g}".replace("-", "−")

    # minor ticks on the linear negative part: halfway between the majors
    neg_major = sorted(t for t in ticks if t <= 0.0)
    if len(neg_major) > 1:
        step = 0.5 * (neg_major[1] - neg_major[0])
        value = -step
        while value >= lower - 1e-12:
            if all(abs(value - t) > 1e-9 for t in neg_major):
                minor.append(value)
            value -= step
    axis.yaxis.set_major_locator(FixedLocator(sorted(set(ticks))))
    axis.yaxis.set_major_formatter(FuncFormatter(label))
    axis.yaxis.set_minor_locator(FixedLocator(sorted(minor)))


def _apply_xi_yscale(axis, scale, web_ylim, linthresh, neg_fraction):
    """y scale of one xi panel: "split" (log above linthresh, linear below,
    negative part = neg_fraction of the height), "symlog", or any
    matplotlib scale.  web_ylim None = automatic range."""
    if scale == "split":
        if web_ylim is None:                  # from the drawn data + bands
            lo, hi = axis.dataLim.y0, axis.dataLim.y1
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = -1.0, 1.0
            web_ylim = (min(1.05 * lo, -linthresh),
                        max(1.5 * hi, 10.0 * linthresh))
        _style_split_axis(axis, *web_ylim, linthresh, neg_fraction)
    elif scale == "symlog":
        if web_ylim is None:
            axis.set_yscale("symlog", linthresh=linthresh)
        else:
            _style_symlog_axis(axis, *web_ylim, linthresh)
    else:
        axis.set_yscale(scale)
        if web_ylim is not None:
            axis.set_ylim(*web_ylim)


@_paper_figure
def plot_1_main_xi_by_mass(
    h5_file,
    select=None,
    *,
    skip_underflow=True,
    xscale="linear",
    yscale="split",
    xlim=None,
    ylim=None,
    ylim_by_web="default",
    linthresh=0.1,
    neg_fraction=NEG_FRACTION,
    sharey=False,
    figsize=None,
    mass_bins=None,
    uncertainty_alpha=0.18,
    save_path=None,
    show=True,
):
    """Directional 3D-WCCFs of the total sample, one row per halo mass.

    Alternative layout of :func:`plot_1_main_xi`: one row per selected mass
    bin and four columns (knot, filament, sheet, void).  Each panel overlays the slices along the three tidal
    eigenvectors e1, e2 and e3 (the other two separation components fixed at
    their index-zero bin, as in :func:`plot_1_main_xi`), so the anisotropy
    is read within a panel and its mass and web dependence across panels.

    Parameters
    ----------
    h5_file : path-like
        Combined HDF5 file readable by ``pywccf.io.load_combined_h5``.
    select : int or iterable of int
        Mass-bin indices to draw, one row each, top to bottom in the given
        order, e.g. ``select=[6, 3, 1]`` (most massive on top).  ``None`` =
        every total 3D bin shared by all web types, in index order.
        (``mass_bins=`` is accepted as an alias.)
    yscale : str or one str per row
        "split" (default): logarithmic for xi > ``linthresh``, linear for
        xi < 0, the negative part taking ``neg_fraction`` of the panel
        height whatever the number of decades above; "symlog"
        (logarithmic beyond ``linthresh`` on both signs); "linear"; or a
        list with one scale per row.
    neg_fraction : float
        Share of the panel height given to xi < 0 on "split" axes
        (default 1/3, module constant ``NEG_FRACTION``).
    ylim_by_web : dict, "default" or None
        y-range of each column (the same for every mass row).  "default" =
        ``XI_YLIM_BY_WEB``: knot (-1.2, 10^3), filament (-1.2, 10^2), sheet
        (-1.2, 20), void (-1.2, 2).  Used with symlog axes; ``ylim`` (one
        range for all panels) overrides it; None = automatic.
    linthresh : float
        Start of the logarithmic part (default 0.1).  "split": [0,
        linthresh] is a short linear stretch (half a decade wide) joining
        the linear negative part to the log part; "symlog": half-width of
        the linear part on both signs.
    sharey : bool or {"row", "col", "all"}
        Shared y-axis; default ``False`` (each panel scaled on its own, so
        the e1/e2/e3 comparison is not compressed by the knot amplitude).
    figsize : (width, height) or None
        Default: 15 x (3.2 per row).
    skip_underflow, xscale, xlim, ylim, uncertainty_alpha,
    save_path, show : as in :func:`plot_1_main_xi`.

    Returns
    -------
    figure, axes
        Matplotlib figure and the ``(len(select), 4)`` array of axes.
    """
    h5_file = Path(h5_file)
    if not h5_file.is_file():
        raise FileNotFoundError(h5_file)

    if select is None:
        select = mass_bins
    if select is not None and not isinstance(select, (int, np.integer)):
        select = list(select)
        if len(set(select)) != len(select):
            raise ValueError(f"select has repeated mass bins: {select}")
    selected = _normalize_mass_bins(h5_file, select)   # keeps the order
    n_rows = len(selected)

    row_scales = (
        [yscale] * n_rows if isinstance(yscale, str) else list(yscale)
    )
    if len(row_scales) != n_rows:
        raise ValueError(
            f"yscale must be one string or {n_rows} (one per row)"
        )
    if figsize is None:
        figsize = (15, 3.6 * n_rows)

    figure, axes = plt.subplots(
        n_rows,
        len(WEB_TYPES),
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    for row, mass_bin_index in enumerate(selected):
        row_label = f"mass bin {mass_bin_index}"
        for column, web in enumerate(WEB_TYPES):
            axis = axes[row, column]
            result = load_combined_h5(
                h5_file,
                dimension="3d",
                sample="total",
                tail="all",
                web=web,
                mass_bin_index=mass_bin_index,
            )
            row_label = _mass_label_short(
                result["container_attrs"], mass_bin_index
            )
            xi = np.asarray(result["xi"], dtype=np.float64)
            xi_sd = np.asarray(result["xi_sd"], dtype=np.float64)
            for direction in range(3):
                radius = np.asarray(
                    result["centers"][direction], dtype=np.float64
                )
                profile, profile_sd = _directional_slice(
                    xi, xi_sd, direction
                )
                if skip_underflow:
                    radius = radius[1:]
                    profile = profile[1:]
                    profile_sd = profile_sd[1:]
                color = STATISTIC_COLORS[direction]
                axis.plot(
                    radius,
                    profile,
                    color=color,
                    marker=STATISTIC_MARKERS[direction],
                    markersize=3.2,
                    linewidth=1.5,
                    zorder=3,
                )
                axis.fill_between(
                    radius,
                    profile - profile_sd,
                    profile + profile_sd,
                    color=color,
                    alpha=uncertainty_alpha,
                    linewidth=0,
                    zorder=2,
                )

            axis.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
            axis.set_xscale(xscale)
            web_ylim = None
            if ylim is not None:
                web_ylim = tuple(ylim)
            elif ylim_by_web is not None:
                table = (
                    XI_YLIM_BY_WEB if isinstance(ylim_by_web, str)
                    else ylim_by_web
                )
                web_ylim = table.get(web)
            _apply_xi_yscale(axis, row_scales[row], web_ylim, linthresh,
                             neg_fraction)
            if xlim is not None:
                axis.set_xlim(*xlim)
            axis.grid(alpha=0.22, linewidth=0.7)
            if row == 0:
                axis.set_title(web.capitalize())
            if row == len(selected) - 1:
                axis.set_xlabel(r"$r_i\,[h^{-1}\mathrm{Mpc}]$")
        axes[row, 0].set_ylabel(row_label + "\n" + XI_3D_LABEL)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STATISTIC_COLORS[direction],
            marker=STATISTIC_MARKERS[direction],
            markersize=3.5,
            linewidth=1.6,
            label=DIRECTION_TITLES[direction],
        )
        for direction in range(3)
    ]
    _finish_figure(figure, legend_handles, save_path, show)
    return figure, axes


def _mass_sort_key(container_attrs: dict, index: int) -> tuple:
    """Order of mass bins by their edges (open upper edge = most massive)."""
    lower = container_attrs.get("mass_bin_lower_log10_msun_h")
    upper = container_attrs.get("mass_bin_upper_log10_msun_h")
    if lower is None or upper is None:
        return (float(index), float(index))
    return (float(lower), float(upper))


def _normalize_mass_pairs(h5_file: Path, pairs, select) -> list[tuple]:
    """Rows of one or two mass-bin indices, top to bottom."""
    if pairs is not None and select is not None:
        raise ValueError("give either pairs= or select=, not both")
    if pairs is None:
        if select is None:                       # all bins, most massive first
            flat = list(_available_total_mass_bins(h5_file))[::-1]
        elif isinstance(select, (int, np.integer)):
            flat = [int(select)]
        else:
            flat = [int(index) for index in select]
        rows = [tuple(flat[i:i + 2]) for i in range(0, len(flat), 2)]
    else:
        rows = []
        for row in pairs:
            row = ((int(row),) if isinstance(row, (int, np.integer))
                   else tuple(int(index) for index in row))
            if not 1 <= len(row) <= 2:
                raise ValueError(f"each pair needs one or two bins, got {row}")
            rows.append(row)
    if not rows:
        raise ValueError("no mass bin selected")
    for row in rows:
        if len(set(row)) != len(row):
            raise ValueError(f"a row repeats a mass bin: {row}")
        _normalize_mass_bins(h5_file, row)       # raises on unavailable bins
    return rows


def _load_or_skip(h5_file: Path, **where):
    """load_combined_h5, or None (with a message) if the record is absent."""
    try:
        return load_combined_h5(h5_file, **where)
    except (KeyError, ValueError) as error:
        print(
            f"[skip {where['dimension'].upper()}] {h5_file.name}, "
            f"{where['sample']}/{where['tail']}, {where['web']}, "
            f"mass bin {where['mass_bin_index']}: {error}"
        )
        return None


def _main_and_ratio_axes(n_rows, figsize, sharey, height_ratios):
    """(n_rows, 4) main axes, each with a ratio axis glued below it."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    figure = plt.figure(figsize=figsize)
    outer = GridSpec(n_rows, len(WEB_TYPES), figure=figure)
    axes = np.empty((n_rows, len(WEB_TYPES)), dtype=object)
    ratio_axes = np.empty_like(axes)
    for row in range(n_rows):
        for column in range(len(WEB_TYPES)):
            inner = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[row, column],
                height_ratios=height_ratios, hspace=0.06,
            )
            share_x = axes[0, 0]
            share_y = None
            if sharey in (True, "all") and (row or column):
                share_y = axes[0, 0]
            elif sharey == "row" and column:
                share_y = axes[row, 0]
            elif sharey == "col" and row:
                share_y = axes[0, column]
            main = figure.add_subplot(inner[0], sharex=share_x,
                                      sharey=share_y)
            axes[row, column] = main
            ratio_axes[row, column] = figure.add_subplot(inner[1],
                                                         sharex=main)
    return figure, axes, ratio_axes


def _ratio_profile(num, den, mode, min_snr, rmin=10.0):
    """Ratio of two profiles with diagonal delta-method errors, only at
    separations r >= rmin.

    mode "xi": xi_num / xi_den (optionally only where
    |xi_den| > min_snr * sigma_den);  mode "1+xi": (1 + xi_num) /
    (1 + xi_den), defined wherever 1 + xi_den > 0.
    High and low tails share the jackknife regions, but the combined file
    stores no covariance, so the errors are added in quadrature (as in
    pywccf.plotting.ratio._mean_ratio_and_error).
    """
    r_n, x_n, s_n = num
    r_d, x_d, s_d = den
    if len(r_n) != len(r_d) or not np.allclose(r_n, r_d):
        x_d = np.interp(r_n, r_d, x_d)
        s_d = np.interp(r_n, r_d, s_d)
    in_range = np.asarray(r_n, dtype=np.float64) >= rmin
    if mode == "1+xi":
        a, b = 1.0 + x_n, 1.0 + x_d
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0.0) & in_range
    elif mode == "xi":
        a, b = x_n, x_d
        ok = np.isfinite(a) & np.isfinite(b) & (b != 0.0) & in_range
        if min_snr:
            ok &= np.abs(b) > min_snr * np.abs(s_d)
    else:
        raise ValueError("ratio mode must be '1+xi' or 'xi'")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(ok, a / b, np.nan)
        error = np.where(
            ok, np.sqrt((s_n / b) ** 2 + (a * s_d / b**2) ** 2), np.nan
        )
    return r_n, value, error


def _changes_sign(profile, rmin):
    """True if xi changes sign somewhere at r >= rmin (a zero crossing)."""
    radius, xi, _ = profile
    xi = np.asarray(xi, dtype=np.float64)[np.asarray(radius) >= rmin]
    signs = np.sign(xi[np.isfinite(xi) & (xi != 0.0)])
    return signs.size > 0 and (signs.min() != signs.max())


def _draw_ratio_panel(axis, drawn, ratio, web, uncertainty_alpha):
    """Curve ratio["num"] over curve ratio["den"] for every direction drawn
    in the main panel (3D: colours, 1D: dark grey), at r >= ratio["rmin"].
    In mode "xi", a direction whose xi changes sign at r >= rmin in either
    tail (e.g. void along e3) is not drawn at all."""
    num, den = drawn.get(ratio.get("num", 0)), drawn.get(ratio.get("den", 1))
    rmin = float(ratio.get("rmin", 10.0))
    mode = ratio.get("mode", "xi")
    values = []
    if num and den:
        for key in (0, 1, 2, 3):
            if key not in num or key not in den:
                continue
            if mode == "xi" and (_changes_sign(num[key], rmin)
                                 or _changes_sign(den[key], rmin)):
                name = ("1D" if key == 3 else f"e{key + 1}")
                print(f"[ratio] {web}: {name} not drawn (xi changes sign "
                      f"at r >= {rmin:g})")
                continue
            radius, value, error = _ratio_profile(
                num[key], den[key], mode, ratio.get("min_snr"), rmin,
            )
            color = STATISTIC_COLORS[key]
            axis.plot(
                radius, value, color=color,
                marker=STATISTIC_MARKERS[key], markersize=2.8,
                markeredgewidth=0.8, linewidth=1.3,
                zorder=5 if key == 3 else 4,
            )
            axis.fill_between(
                radius, value - error, value + error, color=color,
                alpha=(0.5 if key == 3 else 1.0) * uncertainty_alpha,
                linewidth=0, zorder=2,
            )
            values.append(value[np.isfinite(value)])
    axis.axhline(1.0, color="0.25", linewidth=0.8, linestyle=":")
    axis.grid(alpha=0.22, linewidth=0.7)
    lim = ratio.get("ylim")
    if isinstance(lim, dict):
        lim = lim.get(web)
    if lim is None:
        flat = np.concatenate(values) if values else np.array([])
        if flat.size:
            lo, hi = np.nanpercentile(flat, [3.0, 97.0])
            lo, hi = min(lo, 1.0), max(hi, 1.0)
            pad = 0.1 * (hi - lo) or 0.1
            lim = (lo - pad, hi + pad)
    if lim is not None:
        axis.set_ylim(*lim)


def _plot_xi_overlay_rows(
    h5_file: Path,
    rows: list[dict],
    *,
    skip_underflow,
    xscale,
    yscale,
    xlim,
    ylim,
    ylim_by_web,
    linthresh,
    sharey,
    figsize,
    uncertainty_alpha,
    show_1d,
    neg_fraction=NEG_FRACTION,
    legend_web,
    legend_loc,
    save_path,
    show,
    ratio=None,
):
    """Shared engine of the overlay figures: one row per entry of ``rows``,
    four web-type columns, several records ("curves") per panel.

    Each row is ``{"curves": [...], "ylabel": str or None}`` and each curve
    ``{"sample", "tail", "index", "linestyle", "filled", "label"}``: the 3D
    directional slices (colour = direction) and, with ``show_1d``, the 1D
    WCCF (dark grey) of that record, all in the curve's line style, filled
    markers if ``filled`` else open ones.  The curve labels go into a
    line-style legend in the ``legend_web`` panel of the row (or into the
    row's y label when ``legend_web`` is None).

    ``ratio`` (optional dict) adds a ratio panel under every panel:
    ``{"num": i, "den": j, "mode": "1+xi" | "xi", "ylim", "label",
    "height_ratios", "min_snr", "rmin"}`` -- curve i over curve j of
    the row, per
    direction (and 1D).  The axes are then returned as a
    ``(2 * n_rows, 4)`` array, main and ratio rows interleaved.
    """
    if legend_web is not None and legend_web not in WEB_TYPES:
        raise ValueError(f"legend web must be one of {WEB_TYPES}")
    n_rows = len(rows)
    row_scales = (
        [yscale] * n_rows if isinstance(yscale, str) else list(yscale)
    )
    if len(row_scales) != n_rows:
        raise ValueError(
            f"yscale must be one string or {n_rows} (one per row)"
        )
    if figsize is None:
        figsize = (15, (4.8 if ratio else 3.6) * n_rows)

    if ratio is None:
        figure, axes = plt.subplots(
            n_rows,
            len(WEB_TYPES),
            figsize=figsize,
            sharex=True,
            sharey=sharey,
            squeeze=False,
        )
        ratio_axes = None
    else:
        figure, axes, ratio_axes = _main_and_ratio_axes(
            n_rows, figsize, sharey, ratio.get("height_ratios", (3.0, 1.0))
        )

    any_1d = False
    for row, spec in enumerate(rows):
        curves = spec["curves"]
        row_1d = False
        for column, web in enumerate(WEB_TYPES):
            axis = axes[row, column]
            drawn = {}                   # curve number -> profiles drawn
            for number, curve in enumerate(curves):
                drawn[number] = {}
                where = dict(
                    sample=curve["sample"],
                    tail=curve["tail"],
                    web=web,
                    mass_bin_index=curve["index"],
                )
                linestyle, filled = curve["linestyle"], curve["filled"]
                result = load_combined_h5(h5_file, dimension="3d", **where)
                xi = np.asarray(result["xi"], dtype=np.float64)
                xi_sd = np.asarray(result["xi_sd"], dtype=np.float64)
                for direction in range(3):
                    radius = np.asarray(
                        result["centers"][direction], dtype=np.float64
                    )
                    profile, profile_sd = _directional_slice(
                        xi, xi_sd, direction
                    )
                    if skip_underflow:
                        radius = radius[1:]
                        profile = profile[1:]
                        profile_sd = profile_sd[1:]
                    drawn[number][direction] = (radius, profile, profile_sd)
                    color = STATISTIC_COLORS[direction]
                    axis.plot(
                        radius,
                        profile,
                        color=color,
                        linestyle=linestyle,
                        marker=STATISTIC_MARKERS[direction],
                        markersize=3.2,
                        markerfacecolor=color if filled else "white",
                        markeredgewidth=0.9,
                        linewidth=1.5 if filled else 1.3,
                        zorder=4 if filled else 3,
                    )
                    axis.fill_between(
                        radius,
                        profile - profile_sd,
                        profile + profile_sd,
                        color=color,
                        alpha=uncertainty_alpha,
                        linewidth=0,
                        zorder=2,
                    )
                if not show_1d:
                    continue
                result_1d = _load_or_skip(h5_file, dimension="1d", **where)
                if result_1d is None:
                    continue
                any_1d = row_1d = True
                r_1d = np.asarray(result_1d["centers"], dtype=np.float64)
                xi_1d = np.asarray(result_1d["xi"], dtype=np.float64)
                sd_1d = np.asarray(result_1d["xi_sd"], dtype=np.float64)
                if skip_underflow:
                    r_1d, xi_1d, sd_1d = r_1d[1:], xi_1d[1:], sd_1d[1:]
                drawn[number][3] = (r_1d, xi_1d, sd_1d)
                axis.plot(
                    r_1d,
                    xi_1d,
                    color=STATISTIC_COLORS[3],
                    linestyle=linestyle,
                    marker=STATISTIC_MARKERS[3],
                    markersize=3.5,
                    markeredgewidth=0.9,
                    linewidth=1.5 if filled else 1.3,
                    zorder=5 if filled else 4.5,
                )
                axis.fill_between(
                    r_1d,
                    xi_1d - sd_1d,
                    xi_1d + sd_1d,
                    color=STATISTIC_COLORS[3],
                    alpha=0.5 * uncertainty_alpha,
                    linewidth=0,
                    zorder=2,
                )

            axis.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
            axis.set_xscale(xscale)
            web_ylim = None
            if ylim is not None:
                web_ylim = tuple(ylim)
            elif ylim_by_web is not None:
                table = (
                    XI_YLIM_BY_WEB if isinstance(ylim_by_web, str)
                    else ylim_by_web
                )
                web_ylim = table.get(web)
            _apply_xi_yscale(axis, row_scales[row], web_ylim, linthresh,
                             neg_fraction)
            if xlim is not None:
                axis.set_xlim(*xlim)
            axis.grid(alpha=0.22, linewidth=0.7)
            if row == 0:
                axis.set_title(web.capitalize())
            if ratio_axes is not None:
                axis.tick_params(labelbottom=False)
                _draw_ratio_panel(
                    ratio_axes[row, column], drawn, ratio, web,
                    uncertainty_alpha,
                )
                if row != n_rows - 1:
                    ratio_axes[row, column].tick_params(labelbottom=False)
                if column == 0:
                    ratio_axes[row, column].set_ylabel(
                        ratio.get("label") or "ratio", fontsize=13
                    )
            if row == n_rows - 1:
                (axis if ratio_axes is None else ratio_axes[row, column]
                 ).set_xlabel(r"$r_i\,[h^{-1}\mathrm{Mpc}]$")

        head = spec.get("ylabel")
        y_label = (
            r"$\xi_{\rm 3D}(\mathbf{r})$ or $\xi_{\rm 1D}(r)$" if row_1d
            else XI_3D_LABEL
        )
        if legend_web is None:
            lines = [
                f"{'solid' if c['filled'] else 'dashed'}: {c['label']}"
                for c in curves
            ]
            if head:
                lines.insert(0, head)
            axes[row, 0].set_ylabel(
                "\n".join(lines + [y_label]), fontsize=12
            )
        else:
            axes[row, 0].set_ylabel((head + "\n" if head else "")
                                    + y_label)
            handles = [
                Line2D(
                    [0], [0],
                    color="0.2",
                    linestyle=c["linestyle"],
                    marker="o",
                    markersize=3.2,
                    markerfacecolor="0.2" if c["filled"] else "white",
                    markeredgewidth=0.9,
                    linewidth=1.5 if c["filled"] else 1.3,
                )
                for c in curves
            ]
            axes[row, WEB_TYPES.index(legend_web)].legend(
                handles,
                [c["label"] for c in curves],
                loc=legend_loc,
                fontsize=11,
                frameon=False,
                handlelength=2.6,
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STATISTIC_COLORS[direction],
            marker=STATISTIC_MARKERS[direction],
            markersize=3.5,
            linewidth=1.6,
            label=DIRECTION_TITLES[direction],
        )
        for direction in range(3)
    ]
    if ratio_axes is not None:
        stacked = np.empty((2 * n_rows, len(WEB_TYPES)), dtype=object)
        stacked[0::2], stacked[1::2] = axes, ratio_axes
        axes = stacked
    if any_1d:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=STATISTIC_COLORS[3],
                marker=STATISTIC_MARKERS[3],
                markersize=3.5,
                linewidth=1.6,
                label=STATISTIC_LABELS[3],
            )
        )
    _finish_figure(figure, legend_handles, save_path, show)
    return figure, axes


@_paper_figure
def plot_1_main_xi_by_mass_pairs(
    h5_file,
    pairs=None,
    *,
    select=None,
    skip_underflow=True,
    xscale="linear",
    yscale="split",
    xlim=None,
    ylim=None,
    ylim_by_web="default",
    linthresh=0.1,
    neg_fraction=NEG_FRACTION,
    sharey=False,
    figsize=None,
    uncertainty_alpha=0.15,
    high_linestyle="-",
    low_linestyle="--",
    show_1d=True,
    mass_legend_web="void",
    mass_legend_loc="upper right",
    save_path=None,
    show=True,
):
    """Directional 3D-WCCFs of the total sample, two mass bins per row.

    Variant of :func:`plot_1_main_xi_by_mass` for six (or any even number
    of) mass bins in half the rows: each row overlays two mass bins, the
    more massive one solid with filled markers and the less massive one
    dashed with open markers; colours still give the direction (e1, e2, e3).
    Every panel also shows the 1D (isotropic-shell) WCCF of the same sample,
    web type and mass bin, read from the same file (dark grey, "x" markers,
    same line style as its mass bin).  Four columns (knot, filament, sheet,
    void) as before.  Which bin is "more massive" is read from the stored
    mass edges (an open upper edge counts as the largest), not from the
    order given.

    Parameters
    ----------
    h5_file : path-like
        Combined HDF5 file readable by ``pywccf.io.load_combined_h5``.
    pairs : list of (int, int), optional
        Mass-bin indices per row, top to bottom, e.g.
        ``[(5, 4), (3, 2), (1, 0)]``.  A row may hold a single bin.
    select : iterable of int, optional
        Alternative to ``pairs``: a flat list grouped two by two in the
        given order, e.g. ``select=[5, 4, 3, 2, 1, 0]`` -> rows (5, 4),
        (3, 2), (1, 0).  With neither: every available bin, most massive
        first, paired in that order.
    high_linestyle, low_linestyle : str
        Line styles of the more / less massive bin of a row ("-", "--").
    show_1d : bool
        Overlay the 1D WCCF (``dimension="1d"`` records of the file;
        default True).  A missing 1D record is reported and skipped.
    mass_legend_web, mass_legend_loc : str
        Panel (web type) and location of each row's line-style legend,
        which names the two mass bins (default: void panel, upper right,
        where the void profiles leave room).  ``mass_legend_web=None``
        puts the two mass ranges in the row's y label instead.
    yscale, neg_fraction, ylim_by_web, linthresh, sharey, skip_underflow,
    xscale, xlim, ylim, uncertainty_alpha, save_path, show
        As in :func:`plot_1_main_xi_by_mass` (default "split": log for
        xi > 0, linear for xi < 0 on 1/3 of the height; ``yscale`` may be
        one string per row).
    figsize : (width, height) or None
        Default: 15 x (3.2 per row).

    Returns
    -------
    figure, axes
        Matplotlib figure and the ``(n_rows, 4)`` array of axes.
    """
    h5_file = Path(h5_file)
    if not h5_file.is_file():
        raise FileNotFoundError(h5_file)
    if mass_legend_web is not None and mass_legend_web not in WEB_TYPES:
        raise ValueError(f"mass_legend_web must be one of {WEB_TYPES}")

    rows = []
    for bins in _normalize_mass_pairs(h5_file, pairs, select):
        attrs = {
            index: load_combined_h5(
                h5_file,
                dimension="3d",
                sample="total",
                tail="all",
                web=WEB_TYPES[0],
                mass_bin_index=index,
            )["container_attrs"]
            for index in bins
        }
        ordered = sorted(
            bins, key=lambda index: _mass_sort_key(attrs[index], index),
            reverse=True,
        )                                         # most massive first
        rows.append({
            "ylabel": None,
            "curves": [
                {
                    "sample": "total",
                    "tail": "all",
                    "index": index,
                    "linestyle": high_linestyle if rank == 0
                    else low_linestyle,
                    "filled": rank == 0,
                    "label": _mass_label_short(attrs[index], index),
                }
                for rank, index in enumerate(ordered)
            ],
        })

    return _plot_xi_overlay_rows(
        h5_file,
        rows,
        skip_underflow=skip_underflow,
        xscale=xscale,
        yscale=yscale,
        xlim=xlim,
        ylim=ylim,
        ylim_by_web=ylim_by_web,
        linthresh=linthresh,
        sharey=sharey,
        figsize=figsize,
        uncertainty_alpha=uncertainty_alpha,
        show_1d=show_1d,
        neg_fraction=neg_fraction,
        legend_web=mass_legend_web,
        legend_loc=mass_legend_loc,
        save_path=save_path,
        show=show,
    )


# TeX names of the sub-properties used to split the samples
SUBPROPERTY_TEX = {
    "zf": r"$z_{\rm f}$",
    "z_f": r"$z_{\rm f}$",
    "concentration": r"$c$",
    "c": r"$c$",
    "lambda": r"$\lambda$",
    "spin": r"$\lambda$",
}


def _available_mass_bins(
    h5_file: Path, sample: str, tails: Iterable[str]
) -> tuple[int, ...]:
    """3D mass bins present for every web type and every tail of a sample."""
    sets = []
    with h5py.File(h5_file, "r") as stream:
        if f"data/3d/{sample}" not in stream:
            present = sorted(stream["data/3d"]) if "data/3d" in stream else []
            raise KeyError(
                f"{h5_file} has no 3D sample {sample!r}; "
                f"available: {present}"
            )
        for tail in tails:
            for web in WEB_TYPES:
                group_path = f"data/3d/{sample}/{tail}/{web}"
                if group_path not in stream:
                    raise KeyError(f"{h5_file} has no group /{group_path}")
                sets.append({
                    int(name.removeprefix("mass_bin_"))
                    for name in stream[group_path]
                    if name.startswith("mass_bin_")
                })
    shared = set.intersection(*sets) if sets else set()
    if not shared:
        raise ValueError(
            f"{h5_file}: no 3D mass bin of sample {sample!r} is shared by "
            "all web types and tails"
        )
    return tuple(sorted(shared))


def _tex_subscript(label: str) -> str:
    """A legend label as mathtext for a subscript: plain text in roman
    (spaces kept, underscores escaped), $...$ parts passed through, e.g.
    "old" -> \\mathrm{old}, "high $\\lambda$" -> \\mathrm{high\\ }\\lambda."""
    parts = str(label).split("$")
    out = []
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2:                                   # inside $...$
            out.append(part)
        else:
            text = part.replace("_", r"\_").replace(" ", r"\ ")
            out.append(r"\mathrm{" + text + "}")
    return "".join(out) or r"\mathrm{?}"


def _subproperty_tail_label(attrs: dict, sample: str, tail: str) -> str:
    """Legend text of one tail from the record attributes, e.g.
    'old (upper 20% in $z_f$)'; falls back to 'high age'."""
    label = attrs.get("assembly_label")
    if label is None:
        return f"{tail} {sample}"
    label = str(label)
    prop = attrs.get("assembly_subproperty",
                     attrs.get("assembly_scalar_property"))
    pct = attrs.get("assembly_cutoff_percent")
    side = attrs.get("assembly_tail",
                     "upper" if tail == "high" else "lower")
    if prop is None or pct is None:
        return label
    prop_tex = SUBPROPERTY_TEX.get(str(prop), str(prop).replace("_", r"\_"))
    return f"{label} ({side} {float(pct):g}% in {prop_tex})"


@_paper_figure
def plot_1_main_xi_by_subproperty(
    h5_file,
    sample="age",
    select=None,
    *,
    tails=("high", "low"),
    labels=None,
    skip_underflow=True,
    xscale="linear",
    yscale="split",
    xlim=None,
    ylim=None,
    ylim_by_web="default",
    linthresh=0.1,
    neg_fraction=NEG_FRACTION,
    sharey=False,
    figsize=None,
    uncertainty_alpha=0.15,
    high_linestyle="-",
    low_linestyle="--",
    show_1d=True,
    legend_web="void",
    legend_loc="upper right",
    ratio_panel=False,
    ratio_mode="xi",
    ratio_rmin=10.0,
    ratio_ylim=None,
    ratio_label=None,
    ratio_height_ratios=(3.0, 1.0),
    ratio_min_snr=None,
    save_path=None,
    show=True,
):
    """Directional 3D-WCCFs of the two tails of a secondary property, one
    mass bin per row.

    Same layout as :func:`plot_1_main_xi_by_mass_pairs`, but the two curves
    of a panel are the high and low tail of one sub-sample split (``sample``
    = the selection name of the run: "age", "concentration", ...) in the
    same mass bin: high tail solid with filled markers, low tail dashed with
    open markers; colours = direction (e1, e2, e3), dark grey = 1D WCCF.
    The legend in the void panel names the tails from the stored record
    attributes, e.g. "old (upper 20% in z_f)" / "young (lower 20% in z_f)".

    Parameters
    ----------
    h5_file : path-like
        Combined HDF5 file readable by ``pywccf.io.load_combined_h5``.
    sample : str
        Sub-sample split, i.e. the group name under ``data/3d`` ("age",
        "concentration", ...; whatever ``assembly_bias.selections`` named
        in the run).
    select : int or iterable of int
        Mass-bin indices, one row each, top to bottom in the given order,
        as in :func:`plot_1_main_xi_by_mass`.  None = every bin present for
        all webs and both tails, most massive first.
    tails : (str, str)
        (solid, dashed) tails; default ("high", "low").
    labels : (str, str), optional
        Legend text of the two tails, overriding the stored labels.
    high_linestyle, low_linestyle : str
        Line styles of ``tails[0]`` and ``tails[1]``.
    ratio_panel : bool
        Add under every panel a ratio panel tails[0] / tails[1] (e.g. old /
        young) for e1, e2, e3 (colours) and the 1D WCCF (dark grey), with
        a dotted line at 1.  Default False.
    ratio_mode : {"xi", "1+xi"}
        "xi" (default): xi_high / xi_low; "1+xi": (1 + xi_high) /
        (1 + xi_low).
    ratio_rmin : float
        The ratio is drawn only at r_i >= ratio_rmin [h^-1 Mpc] (default
        10).  In mode "xi", a direction whose xi changes sign at
        r_i >= ratio_rmin in either tail (a transition, e.g. void along
        e3) is left out of the ratio panel altogether; a message says
        which.
    ratio_min_snr : float or None
        Optionally also require |xi_low| > ratio_min_snr * sigma(xi_low).
        Errors: high and low added in quadrature (delta method; the file
        stores no covariance between them), as in pywccf.plotting.ratio.
    ratio_ylim : (lo, hi), dict {web: (lo, hi)} or None
        y-range of the ratio panels; None = automatic per panel (3-97 %
        range of the drawn ratios, always including 1).
    ratio_label : str, optional
        y label of the ratio panels; default xi_A / xi_B with A, B the
        legend labels of the two tails (``labels`` when given, else the
        stored ones, e.g. old / young).
    ratio_height_ratios : (main, ratio)
        Relative heights of a main panel and its ratio panel (3:1).
    show_1d, legend_web, legend_loc, yscale, neg_fraction, ylim_by_web,
    linthresh, sharey, skip_underflow, xscale, xlim, ylim,
    uncertainty_alpha, figsize, save_path, show
        As in :func:`plot_1_main_xi_by_mass_pairs` (``legend_web=None``
        writes the tail labels into the row's y label).  figsize defaults
        to 15 x (4.3 per row) with ratio panels.

    Returns
    -------
    figure, axes
        Matplotlib figure and the ``(len(select), 4)`` array of axes; with
        ``ratio_panel=True`` a ``(2 * len(select), 4)`` array, main and
        ratio rows interleaved (``axes[0::2]`` main, ``axes[1::2]`` ratio).
    """
    h5_file = Path(h5_file)
    if not h5_file.is_file():
        raise FileNotFoundError(h5_file)
    tails = tuple(str(tail) for tail in tails)
    if len(tails) != 2 or tails[0] == tails[1]:
        raise ValueError(f"tails must be two different tails, got {tails}")
    if labels is not None and len(labels) != 2:
        raise ValueError("labels must give two strings (one per tail)")

    available = _available_mass_bins(h5_file, sample, tails)
    if select is None:
        selected = list(available)[::-1]
    else:
        selected = (
            [int(select)] if isinstance(select, (int, np.integer))
            else [int(index) for index in select]
        )
        if not selected:
            raise ValueError("select must contain at least one mass bin")
        if len(set(selected)) != len(selected):
            raise ValueError(f"select has repeated mass bins: {selected}")
        missing = [index for index in selected if index not in available]
        if missing:
            raise ValueError(
                f"Requested mass bins {missing} are unavailable for sample "
                f"{sample!r}; available bins are {list(available)}"
            )

    rows = []
    for index in selected:
        records = {
            tail: load_combined_h5(
                h5_file,
                dimension="3d",
                sample=sample,
                tail=tail,
                web=WEB_TYPES[0],
                mass_bin_index=index,
            )
            for tail in tails
        }
        curves = []
        for rank, tail in enumerate(tails):
            label = (
                labels[rank] if labels is not None
                else _subproperty_tail_label(records[tail]["attrs"], sample,
                                             tail)
            )
            curves.append({
                "sample": sample,
                "tail": tail,
                "index": index,
                "linestyle": high_linestyle if rank == 0 else low_linestyle,
                "filled": rank == 0,
                "label": label,
            })
        rows.append({
            "ylabel": _mass_label_short(
                records[tails[0]]["container_attrs"], index
            ),
            "curves": curves,
        })
        short = (
            [str(label) for label in labels] if labels is not None
            else [str(records[tail]["attrs"].get("assembly_label", tail))
                  for tail in tails]
        )

    ratio = None
    if ratio_panel:
        if ratio_mode not in ("1+xi", "xi"):
            raise ValueError("ratio_mode must be '1+xi' or 'xi'")
        if ratio_label is None:
            a, b = ("{" + _tex_subscript(name) + "}" for name in short)
            ratio_label = (
                rf"$(1+\xi_{a})/(1+\xi_{b})$" if ratio_mode == "1+xi"
                else rf"$\xi_{a}/\xi_{b}$"
            )
        ratio = {
            "num": 0,
            "den": 1,
            "mode": ratio_mode,
            "ylim": ratio_ylim,
            "label": ratio_label,
            "height_ratios": tuple(ratio_height_ratios),
            "min_snr": ratio_min_snr,
            "rmin": float(ratio_rmin),
        }

    return _plot_xi_overlay_rows(
        h5_file,
        rows,
        skip_underflow=skip_underflow,
        xscale=xscale,
        yscale=yscale,
        xlim=xlim,
        ylim=ylim,
        ylim_by_web=ylim_by_web,
        linthresh=linthresh,
        sharey=sharey,
        figsize=figsize,
        uncertainty_alpha=uncertainty_alpha,
        show_1d=show_1d,
        neg_fraction=neg_fraction,
        legend_web=legend_web,
        legend_loc=legend_loc,
        save_path=save_path,
        show=show,
        ratio=ratio,
    )


def _combined_run_priority(inventory: dict) -> tuple[int, int, int]:
    """Rank duplicate combined files by completeness of their metadata."""
    return (
        int(str(inventory.get("run_status", "")).lower() == "complete"),
        int(
            str(inventory.get("validation_status", "")).lower()
            == "complete"
        ),
        int(inventory.get("lambda_threshold") is not None),
    )


def _register_preferred_combined_run(
    runs: dict,
    inventories: dict,
    key: float,
    file_path: Path,
    inventory: dict,
):
    """Register one run, preferring a complete file for duplicate metadata."""
    duplicate = next(
        (
            known
            for known in runs
            if np.isclose(known, key, rtol=0.0, atol=1.0e-12)
        ),
        None,
    )
    if duplicate is None:
        runs[key] = file_path
        inventories[key] = inventory
        return

    old_priority = _combined_run_priority(inventories[duplicate])
    new_priority = _combined_run_priority(inventory)
    if new_priority > old_priority:
        runs[duplicate] = file_path
        inventories[duplicate] = inventory
    elif new_priority == old_priority:
        raise ValueError(
            f"Ambiguous combined runs for parameter={key:g}: "
            f"{runs[duplicate]} and {file_path}"
        )


def _density_line_styles(values) -> dict[float, object]:
    """Map increasing parameter values to increasingly dense line styles.

    The largest value is always solid. Lower values are dotted, with the gap
    shrinking monotonically as the parameter increases.
    """
    ordered = tuple(sorted({float(value) for value in values}))
    if not ordered:
        raise ValueError("At least one value is required for line styles")
    if len(ordered) == 1:
        return {ordered[0]: "-"}

    styles = {}
    dotted_count = len(ordered) - 1
    gaps = np.linspace(5.2, 1.35, dotted_count)
    for value, gap in zip(ordered[:-1], gaps):
        styles[value] = (0, (1.0, float(gap)))
    styles[ordered[-1]] = "-"
    return styles


def _make_wccf_ratio_axes(figsize):
    """Create four web panels, each with a main and a ratio axis."""
    figure, raw_axes = plt.subplots(
        4,
        2,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"height_ratios": (3.0, 1.25, 3.0, 1.25)},
    )
    axes = np.empty((2, 2, 2), dtype=object)
    for row in range(2):
        for column in range(2):
            main_axis = raw_axes[2 * row, column]
            ratio_axis = raw_axes[2 * row + 1, column]
            ratio_axis.sharex(main_axis)
            axes[row, column, 0] = main_axis
            axes[row, column, 1] = ratio_axis
    return figure, axes


def _draw_wccf_1d_3d_pair(
    main_axis,
    ratio_axis,
    *,
    xi_3d,
    sd_3d,
    centers_3d,
    xi_1d,
    sd_1d,
    r_1d,
    line_style,
    skip_underflow,
    show_uncertainty,
):
    """Draw one parameter value on a compound 3D/1D comparison panel."""
    xi_1d = np.asarray(xi_1d, dtype=np.float64)
    sd_1d = np.asarray(sd_1d, dtype=np.float64)
    r_1d = np.asarray(r_1d, dtype=np.float64)
    if skip_underflow:
        xi_1d = xi_1d[1:]
        sd_1d = sd_1d[1:]
        r_1d = r_1d[1:]

    for direction in range(3):
        profile, profile_sd = _directional_slice(
            xi_3d, sd_3d, direction
        )
        radius = np.asarray(centers_3d[direction], dtype=np.float64)
        if skip_underflow:
            profile = profile[1:]
            profile_sd = profile_sd[1:]
            radius = radius[1:]

        if len(r_1d) == len(radius) and np.allclose(
            r_1d, radius, rtol=5.0e-6, atol=5.0e-8
        ):
            xi_1d_at_r = xi_1d
            sd_1d_at_r = sd_1d
        else:
            xi_1d_at_r = np.interp(radius, r_1d, xi_1d)
            sd_1d_at_r = np.interp(radius, r_1d, sd_1d)

        color = STATISTIC_COLORS[direction]
        marker = STATISTIC_MARKERS[direction]
        main_axis.plot(
            radius,
            profile,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=3,
            markersize=3.0,
            linewidth=1.35,
            alpha=0.90,
        )
        if show_uncertainty:
            main_axis.fill_between(
                radius,
                profile - profile_sd,
                profile + profile_sd,
                color=color,
                alpha=0.035,
                linewidth=0,
            )

        denominator = 1.0 + xi_1d_at_r
        valid = (
            np.isfinite(profile)
            & np.isfinite(denominator)
            & (np.abs(denominator) > 1.0e-10)
        )
        ratio = np.full_like(profile, np.nan, dtype=np.float64)
        ratio[valid] = (1.0 + profile[valid]) / denominator[valid]
        ratio_axis.plot(
            radius,
            ratio,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=3,
            markersize=2.8,
            linewidth=1.25,
            alpha=0.90,
        )
        if show_uncertainty:
            ratio_sd = np.full_like(ratio, np.nan, dtype=np.float64)
            ratio_sd[valid] = np.sqrt(
                (profile_sd[valid] / denominator[valid]) ** 2
                + (
                    (1.0 + profile[valid])
                    * sd_1d_at_r[valid]
                    / denominator[valid] ** 2
                )
                ** 2
            )
            ratio_axis.fill_between(
                radius,
                ratio - ratio_sd,
                ratio + ratio_sd,
                color=color,
                alpha=0.035,
                linewidth=0,
            )

    main_axis.plot(
        r_1d,
        xi_1d,
        color=STATISTIC_COLORS[3],
        linestyle=line_style,
        marker=STATISTIC_MARKERS[3],
        markevery=3,
        markersize=3.5,
        linewidth=1.55,
        alpha=0.95,
        zorder=5,
    )
    if show_uncertainty:
        main_axis.fill_between(
            r_1d,
            xi_1d - sd_1d,
            xi_1d + sd_1d,
            color=STATISTIC_COLORS[3],
            alpha=0.045,
            linewidth=0,
        )


def _format_wccf_ratio_panel(
    main_axis,
    ratio_axis,
    web,
    *,
    xscale,
    yscale,
    xlim,
    ratio_ylim,
):
    main_axis.axhline(0.0, color="k", linewidth=0.8, linestyle=":", alpha=0.65)
    ratio_axis.axhline(1.0, color="k", linewidth=0.9, linestyle=":", alpha=0.75)
    main_axis.set_xscale(xscale)
    ratio_axis.set_xscale(xscale)
    main_axis.set_yscale(yscale)
    if xlim is not None:
        main_axis.set_xlim(*xlim)
    if ratio_ylim is not None:
        ratio_axis.set_ylim(*ratio_ylim)
    main_axis.grid(alpha=0.23)
    ratio_axis.grid(alpha=0.23)
    main_axis.set_title(web.capitalize(), fontsize=12, fontweight="bold")
    main_axis.set_ylabel(r"$\xi$", fontsize=11)
    ratio_axis.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$", fontsize=10)
    ratio_axis.set_ylabel(r"$(1+\xi_i)/(1+\xi_{\rm 1D})$", fontsize=9)
    ratio_axis.tick_params(axis="both", labelsize=8)
    plt.setp(main_axis.get_xticklabels(), visible=False)


def _statistic_legend_handles():
    return [
        Line2D(
            [0],
            [0],
            color=STATISTIC_COLORS[index],
            marker=STATISTIC_MARKERS[index],
            linestyle="-",
            linewidth=1.8,
            markersize=4,
            label=STATISTIC_LABELS[index],
        )
        for index in range(4)
    ]


def plot_2_main_contour(
    h5_file,
    mass_bin_idx=None,
    *,
    skip_underflow=True,
    mirror=True,
    cmap="viridis",
    vlim=None,
    symmetric=True,
    levels=12,
    filled=True,
    show_lines=True,
    bold_zero_line=False,
    line_color="k",
    line_width=0.55,
    line_alpha=0.55,
    figsize=(12, 15),
    save_path=None,
    show=True,
):
    """Plot total-sample 3D-WCCF contour slices from a combined HDF5 file.

    The file must follow the combined layout read by
    ``pywccf.io.load_combined_h5``. The output and four-row by three-column
    layout follow ``func.py::plot_xi_3d_contour``. The columns show the
    ``(e1, e2)``, ``(e1, e3)``, and ``(e2, e3)`` planes, respectively; the
    remaining separation component is fixed to its index-zero bin. Each web
    row has an independent contour normalization and colorbar.
    """
    if mass_bin_idx is None:
        raise TypeError("mass_bin_idx must be specified")
    h5_file = Path(h5_file)
    if not h5_file.is_file():
        raise FileNotFoundError(h5_file)
    _normalize_mass_bins(h5_file, int(mass_bin_idx))

    def mirror_slice(slab, first_centers, second_centers):
        n_first, n_second = slab.shape
        mirrored = np.empty(
            (2 * n_first, 2 * n_second), dtype=slab.dtype
        )
        mirrored[:n_first, :n_second] = slab[::-1, ::-1]
        mirrored[:n_first, n_second:] = slab[::-1, :]
        mirrored[n_first:, :n_second] = slab[:, ::-1]
        mirrored[n_first:, n_second:] = slab
        return (
            mirrored,
            np.concatenate([-first_centers[::-1], first_centers]),
            np.concatenate([-second_centers[::-1], second_centers]),
        )

    panel_data = {}
    finite_values = {}
    for web_name in WEB_TYPES:
        result = load_combined_h5(
            h5_file,
            dimension="3d",
            sample="total",
            tail="all",
            web=web_name,
            mass_bin_index=int(mass_bin_idx),
        )
        xi = np.asarray(result["xi"], dtype=np.float64)
        r1, r2, r3 = (
            np.asarray(axis, dtype=np.float64)
            for axis in result["centers"]
        )
        views = [
            (xi[:, :, 0], r1, r2),
            (xi[:, 0, :], r1, r3),
            (xi[0, :, :], r2, r3),
        ]
        if skip_underflow:
            views = [
                (slab[1:, 1:], first[1:], second[1:])
                for slab, first, second in views
            ]
        if mirror:
            views = [
                mirror_slice(slab, first, second)
                for slab, first, second in views
            ]
        panel_data[web_name] = views

        row_values = []
        for slab, _, _ in views:
            values = slab[np.isfinite(slab)]
            if values.size:
                row_values.append(values)
        if not row_values:
            raise RuntimeError(
                f"All {web_name} slice values are non-finite; "
                "nothing to plot."
            )
        finite_values[web_name] = np.concatenate(row_values)

    row_styles = {}
    for web_name in WEB_TYPES:
        values = finite_values[web_name]
        requested_vlim = vlim.get(web_name) if isinstance(vlim, dict) else vlim
        if requested_vlim is None:
            if symmetric:
                web_vmax = float(np.max(np.abs(values)))
                web_vmin = -web_vmax
            else:
                web_vmin = float(np.min(values))
                web_vmax = float(np.max(values))
        else:
            if len(requested_vlim) != 2:
                raise ValueError(
                    f"vlim for {web_name} must be a (vmin, vmax) pair"
                )
            web_vmin, web_vmax = map(float, requested_vlim)
            if symmetric:
                web_vmax = max(abs(web_vmin), abs(web_vmax))
                web_vmin = -web_vmax
        if (
            not np.isfinite(web_vmin)
            or not np.isfinite(web_vmax)
            or web_vmin >= web_vmax
        ):
            raise ValueError(
                f"Invalid contour limits for {web_name}: "
                f"({web_vmin}, {web_vmax})"
            )

        if isinstance(levels, dict):
            if web_name not in levels:
                raise ValueError(
                    f"levels dictionary is missing web={web_name!r}"
                )
            requested_levels = levels[web_name]
        else:
            requested_levels = levels
        if np.isscalar(requested_levels):
            n_levels = int(requested_levels)
            if n_levels < 2:
                raise ValueError(f"levels for {web_name} must be at least 2")
            if symmetric and n_levels % 2 == 0:
                n_levels += 1
            web_levels = np.linspace(web_vmin, web_vmax, n_levels)
            if symmetric:
                web_levels[n_levels // 2] = 0.0
        else:
            web_levels = np.asarray(requested_levels, dtype=np.float64)
            if (
                web_levels.ndim != 1
                or len(web_levels) < 2
                or not np.all(np.isfinite(web_levels))
                or np.any(np.diff(web_levels) <= 0)
            ):
                raise ValueError(
                    f"Explicit contour levels for {web_name} must be "
                    "finite and increasing"
                )

        if isinstance(cmap, dict):
            if web_name not in cmap:
                raise ValueError(
                    f"cmap dictionary is missing web={web_name!r}"
                )
            web_cmap = cmap[web_name]
        else:
            web_cmap = cmap
        row_styles[web_name] = (
            web_cmap,
            web_vmin,
            web_vmax,
            web_levels,
        )

    figure, axes = plt.subplots(
        4,
        3,
        figsize=figsize,
        squeeze=False,
    )
    column_titles = (
        r"$(\mathbf{e}_1,\mathbf{e}_2)$ plane; $r_3$ fixed",
        r"$(\mathbf{e}_1,\mathbf{e}_3)$ plane; $r_2$ fixed",
        r"$(\mathbf{e}_2,\mathbf{e}_3)$ plane; $r_1$ fixed",
    )
    x_labels = (
        r"$r_1\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_1\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_2\,[h^{-1}\mathrm{Mpc}]$",
    )
    y_labels = (
        r"$r_2\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_3\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_3\,[h^{-1}\mathrm{Mpc}]$",
    )

    def draw(
        axis,
        slab,
        first_centers,
        second_centers,
        web_cmap,
        web_vmin,
        web_vmax,
        web_levels,
    ):
        plot_values = np.ma.masked_invalid(slab.T)
        if filled:
            artist = axis.contourf(
                first_centers,
                second_centers,
                plot_values,
                levels=web_levels,
                cmap=web_cmap,
                vmin=web_vmin,
                vmax=web_vmax,
                extend="both",
            )
            if show_lines:
                axis.contour(
                    first_centers,
                    second_centers,
                    plot_values,
                    levels=web_levels,
                    colors=line_color,
                    linewidths=line_width,
                    alpha=line_alpha,
                )
        else:
            artist = axis.contour(
                first_centers,
                second_centers,
                plot_values,
                levels=web_levels,
                cmap=web_cmap,
                vmin=web_vmin,
                vmax=web_vmax,
                linewidths=max(line_width, 0.8),
            )

        finite_plot_values = plot_values.compressed()
        if (
            bold_zero_line
            and finite_plot_values.size
            and np.min(finite_plot_values) < 0.0
            and np.max(finite_plot_values) > 0.0
        ):
            axis.contour(
                first_centers,
                second_centers,
                plot_values,
                levels=[0.0],
                colors=line_color,
                linewidths=max(1.5, 2.5 * line_width),
                alpha=max(0.9, line_alpha),
            )
        axis.set_aspect("equal", adjustable="box")
        axis.tick_params(axis="both", labelsize=8)
        if mirror:
            axis.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
            axis.axvline(0.0, color="k", linewidth=0.6, alpha=0.5)
        return artist

    row_artists = []
    for row, web_name in enumerate(WEB_TYPES):
        web_cmap, web_vmin, web_vmax, web_levels = row_styles[web_name]
        row_artist = None
        for column, (slab, first, second) in enumerate(panel_data[web_name]):
            axis = axes[row, column]
            row_artist = draw(
                axis,
                slab,
                first,
                second,
                web_cmap,
                web_vmin,
                web_vmax,
                web_levels,
            )
            axis.set_xlabel(x_labels[column], fontsize=9)
            axis.set_ylabel(y_labels[column], fontsize=9)
            if row == 0:
                axis.set_title(column_titles[column], fontsize=11)
        axes[row, 0].annotate(
            web_name.capitalize(),
            xy=(-0.27, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
        )
        row_artists.append(row_artist)

    # Reserve a narrow strip on the right, then place one colorbar beside
    # each already-tightened row. This avoids colorbar overlap while keeping
    # the requested tight-layout behavior.
    figure.tight_layout(rect=(0.0, 0.0, 0.97, 1.0), pad=0.9)
    for row, (web_name, row_artist) in enumerate(
        zip(WEB_TYPES, row_artists)
    ):
        row_positions = [axis.get_position() for axis in axes[row, :]]
        row_bottom = min(position.y0 for position in row_positions)
        row_top = max(position.y1 for position in row_positions)
        row_height = row_top - row_bottom
        colorbar_axis = figure.add_axes(
            [0.945, row_bottom + 0.06 * row_height, 0.014, 0.88 * row_height]
        )
        colorbar = figure.colorbar(
            row_artist,
            cax=colorbar_axis,
        )
        colorbar.set_label(rf"$\xi_{{\mathrm{{{web_name}}}}}$", fontsize=11)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return figure, axes


def plot_a1_ss_impact(
    data_folder,
    mass_bin_idx,
    scales=(1, 2, 3, 4, 5),
    *,
    lambda_threshold=0.0,
    file_pattern="*_s*_lth*_combined.h5",
    xscale="linear",
    yscale="linear",
    xlim=None,
    ratio_ylim=None,
    skip_underflow=True,
    show_uncertainty=False,
    figsize=(14, 14),
    save_path=None,
    show=True,
):
    """Compare directional 3D and isotropic 1D WCCFs across smoothing scales.

    Data are read from ``sample='total'`` and ``tail='all'`` records in the
    combined HDF5 files supported by ``pywccf.io.load_combined_h5``. Line
    patterns become denser with increasing smoothing scale, and the largest
    requested scale is represented by a solid line.
    """
    data_folder = Path(data_folder)
    scales = tuple(float(scale) for scale in scales)
    if not scales:
        raise ValueError("scales must contain at least one value")

    selected_runs = _select_smoothing_runs(
        data_folder,
        scales,
        lambda_threshold=lambda_threshold,
        file_pattern=file_pattern,
    )

    styles = _density_line_styles(scales)
    figure, axes = _make_wccf_ratio_axes(figsize)
    pair_count = 0

    for web_index, web in enumerate(WEB_TYPES):
        row, column = divmod(web_index, 2)
        main_axis = axes[row, column, 0]
        ratio_axis = axes[row, column, 1]
        for scale in scales:
            file_path = selected_runs[scale]
            try:
                result_1d = load_combined_h5(
                    file_path,
                    dimension="1d",
                    sample="total",
                    tail="all",
                    web=web,
                    mass_bin_index=int(mass_bin_idx),
                )
                result_3d = load_combined_h5(
                    file_path,
                    dimension="3d",
                    sample="total",
                    tail="all",
                    web=web,
                    mass_bin_index=int(mass_bin_idx),
                )
            except (KeyError, ValueError) as error:
                print(
                    f"[skip] {file_path.name}, {web}, "
                    f"mass bin {mass_bin_idx}: {error}"
                )
                continue

            _draw_wccf_1d_3d_pair(
                main_axis,
                ratio_axis,
                xi_3d=np.asarray(result_3d["xi"], dtype=np.float64),
                sd_3d=np.asarray(result_3d["xi_sd"], dtype=np.float64),
                centers_3d=result_3d["centers"],
                xi_1d=np.asarray(result_1d["xi"], dtype=np.float64),
                sd_1d=np.asarray(result_1d["xi_sd"], dtype=np.float64),
                r_1d=np.asarray(result_1d["centers"], dtype=np.float64),
                line_style=styles[scale],
                skip_underflow=skip_underflow,
                show_uncertainty=show_uncertainty,
            )
            pair_count += 1

        _format_wccf_ratio_panel(
            main_axis,
            ratio_axis,
            web,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ratio_ylim=ratio_ylim,
        )

    if pair_count == 0:
        plt.close(figure)
        raise FileNotFoundError(
            f"No paired 3D/1D smoothing runs for mass bin {mass_bin_idx} "
            f"in {data_folder}"
        )

    scale_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=styles[scale],
            linewidth=1.8,
            label=rf"$R_{{\rm s}}={scale:g}\,h^{{-1}}\mathrm{{Mpc}}$",
        )
        for scale in scales
    ]
    figure.legend(
        handles=_statistic_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=4,
        fontsize=9,
        frameon=True,
    )
    figure.legend(
        handles=scale_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.950),
        ncol=min(len(scale_handles), 5),
        fontsize=9,
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.8)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return figure, axes


def _lambda_threshold_from_file(file_path: Path) -> float:
    """Read a web-classification threshold from HDF5 or its filename."""
    with h5py.File(file_path, "r") as stream:
        if "lambda_threshold" in stream.attrs:
            return float(stream.attrs["lambda_threshold"])
    match = re.search(
        r"_lth(?P<tag>m?\d+(?:p\d+)?)_combined\.h5$", file_path.name
    )
    if match is None:
        raise ValueError(f"Cannot determine lambda_threshold from {file_path}")
    tag = match.group("tag")
    sign = -1.0 if tag.startswith("m") else 1.0
    if tag.startswith("m"):
        tag = tag[1:]
    return sign * float(tag.replace("p", "."))


def _select_smoothing_runs(
    data_folder,
    scales,
    *,
    lambda_threshold,
    file_pattern,
) -> dict[float, Path]:
    """Find one preferred combined file for every requested smoothing scale."""
    data_folder = Path(data_folder)
    scales = tuple(float(scale) for scale in scales)
    candidate_files = sorted(data_folder.glob(file_pattern))
    if not candidate_files:
        raise FileNotFoundError(
            f"No combined HDF5 files match {file_pattern!r} in "
            f"{data_folder}"
        )

    runs_by_scale = {}
    inventories = {}
    for file_path in candidate_files:
        inventory = combined_h5_inventory(file_path)
        threshold = _lambda_threshold_from_file(file_path)
        if not np.isclose(
            threshold,
            float(lambda_threshold),
            rtol=0.0,
            atol=1.0e-12,
        ):
            continue
        scale = float(inventory["smoothing_scale_mpc_h"])
        _register_preferred_combined_run(
            runs_by_scale,
            inventories,
            scale,
            file_path,
            inventory,
        )

    selected_runs = {}
    for requested_scale in scales:
        match = next(
            (
                known
                for known in runs_by_scale
                if np.isclose(
                    known,
                    requested_scale,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ),
            None,
        )
        if match is None:
            raise FileNotFoundError(
                f"No combined run for smoothing_scale={requested_scale:g}, "
                f"lambda_threshold={float(lambda_threshold):g}"
            )
        selected_runs[requested_scale] = runs_by_scale[match]
    return selected_runs


def plot_a2_ss_evolution(
    data_folder,
    mass_bin_idx,
    scales=(1, 2, 3, 4, 5),
    *,
    web,
    lambda_threshold=0.0,
    file_pattern="*_s*_lth*_combined.h5",
    skip_underflow=True,
    mirror=True,
    cmap="viridis",
    vlim=None,
    symmetric=True,
    levels=12,
    filled=True,
    show_lines=True,
    bold_zero_line=False,
    line_color="k",
    line_width=0.55,
    line_alpha=0.55,
    figsize=(19, 11),
    save_path=None,
    show=True,
):
    """Show the evolution of three 3D-WCCF planes with smoothing scale.

    Rows are the ``(e1, e2)``, ``(e1, e3)``, and ``(e2, e3)`` planes;
    columns are the requested smoothing scales. All panels share a single
    contour normalization so that their colors can be compared directly.
    Data are loaded from the total/all records in combined HDF5 files.
    """
    data_folder = Path(data_folder)
    scales = tuple(float(scale) for scale in scales)
    if not scales:
        raise ValueError("scales must contain at least one value")
    if web not in WEB_TYPES:
        raise ValueError(f"web={web!r} is invalid; expected one of {WEB_TYPES}")

    selected_runs = _select_smoothing_runs(
        data_folder,
        scales,
        lambda_threshold=lambda_threshold,
        file_pattern=file_pattern,
    )

    def mirror_slice(slab, first_centers, second_centers):
        n_first, n_second = slab.shape
        mirrored = np.empty(
            (2 * n_first, 2 * n_second), dtype=slab.dtype
        )
        mirrored[:n_first, :n_second] = slab[::-1, ::-1]
        mirrored[:n_first, n_second:] = slab[::-1, :]
        mirrored[n_first:, :n_second] = slab[:, ::-1]
        mirrored[n_first:, n_second:] = slab
        return (
            mirrored,
            np.concatenate([-first_centers[::-1], first_centers]),
            np.concatenate([-second_centers[::-1], second_centers]),
        )

    scale_views = {}
    finite_values = []
    for scale in scales:
        file_path = selected_runs[scale]
        result = load_combined_h5(
            file_path,
            dimension="3d",
            sample="total",
            tail="all",
            web=web,
            mass_bin_index=int(mass_bin_idx),
        )
        xi = np.asarray(result["xi"], dtype=np.float64)
        r1, r2, r3 = (
            np.asarray(axis, dtype=np.float64)
            for axis in result["centers"]
        )
        views = [
            (xi[:, :, 0], r1, r2),
            (xi[:, 0, :], r1, r3),
            (xi[0, :, :], r2, r3),
        ]
        if skip_underflow:
            views = [
                (slab[1:, 1:], first[1:], second[1:])
                for slab, first, second in views
            ]
        if mirror:
            views = [
                mirror_slice(slab, first, second)
                for slab, first, second in views
            ]
        scale_views[scale] = views
        for slab, _, _ in views:
            values = slab[np.isfinite(slab)]
            if values.size:
                finite_values.append(values)

    if not finite_values:
        raise RuntimeError(
            f"All slices are non-finite for web={web!r}, "
            f"mass_bin_idx={mass_bin_idx}"
        )
    finite_values = np.concatenate(finite_values)

    if vlim is None:
        if symmetric:
            vmax = float(np.max(np.abs(finite_values)))
            vmin = -vmax
        else:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
    else:
        if len(vlim) != 2:
            raise ValueError("vlim must be a (vmin, vmax) pair")
        vmin, vmax = map(float, vlim)
        if symmetric:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"Invalid contour limits: ({vmin}, {vmax})")

    if np.isscalar(levels):
        n_levels = int(levels)
        if n_levels < 2:
            raise ValueError("levels must be at least 2")
        if symmetric and n_levels % 2 == 0:
            n_levels += 1
        contour_levels = np.linspace(vmin, vmax, n_levels)
        if symmetric:
            contour_levels[n_levels // 2] = 0.0
    else:
        contour_levels = np.asarray(levels, dtype=np.float64)
        if (
            contour_levels.ndim != 1
            or len(contour_levels) < 2
            or not np.all(np.isfinite(contour_levels))
            or np.any(np.diff(contour_levels) <= 0)
        ):
            raise ValueError(
                "Explicit contour levels must be finite and increasing"
            )

    figure, axes = plt.subplots(
        3,
        len(scales),
        figsize=figsize,
        squeeze=False,
    )
    x_labels = (
        r"$r_1\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_1\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_2\,[h^{-1}\mathrm{Mpc}]$",
    )
    y_labels = (
        r"$r_2\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_3\,[h^{-1}\mathrm{Mpc}]$",
        r"$r_3\,[h^{-1}\mathrm{Mpc}]$",
    )
    plane_labels = (
        r"$(\mathbf{e}_1,\mathbf{e}_2)$ plane; $r_3$ fixed",
        r"$(\mathbf{e}_1,\mathbf{e}_3)$ plane; $r_2$ fixed",
        r"$(\mathbf{e}_2,\mathbf{e}_3)$ plane; $r_1$ fixed",
    )

    artist = None
    for column, scale in enumerate(scales):
        for row, (slab, first, second) in enumerate(scale_views[scale]):
            axis = axes[row, column]
            values = np.ma.masked_invalid(slab.T)
            if filled:
                artist = axis.contourf(
                    first,
                    second,
                    values,
                    levels=contour_levels,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extend="both",
                )
                if show_lines:
                    axis.contour(
                        first,
                        second,
                        values,
                        levels=contour_levels,
                        colors=line_color,
                        linewidths=line_width,
                        alpha=line_alpha,
                    )
            else:
                artist = axis.contour(
                    first,
                    second,
                    values,
                    levels=contour_levels,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=max(line_width, 0.8),
                )

            finite = values.compressed()
            if (
                bold_zero_line
                and finite.size
                and np.min(finite) < 0.0
                and np.max(finite) > 0.0
            ):
                axis.contour(
                    first,
                    second,
                    values,
                    levels=[0.0],
                    colors=line_color,
                    linewidths=max(1.5, 2.5 * line_width),
                    alpha=max(0.9, line_alpha),
                )

            axis.set_aspect("equal", adjustable="box")
            axis.tick_params(axis="both", labelsize=7)
            axis.set_xlabel(x_labels[row], fontsize=8)
            if column == 0:
                axis.set_ylabel(
                    y_labels[row] + "\n" + plane_labels[row],
                    fontsize=8,
                )
            if mirror:
                axis.axhline(0.0, color="k", linewidth=0.5, alpha=0.45)
                axis.axvline(0.0, color="k", linewidth=0.5, alpha=0.45)
        axes[0, column].set_title(
            rf"$R_{{\rm s}}={scale:g}\,h^{{-1}}\mathrm{{Mpc}}$",
            fontsize=11,
            fontweight="bold",
        )

    figure.tight_layout(rect=(0.0, 0.0, 0.96, 1.0), pad=0.8)
    panel_positions = [axis.get_position() for axis in axes.ravel()]
    bottom = min(position.y0 for position in panel_positions)
    top = max(position.y1 for position in panel_positions)
    colorbar_axis = figure.add_axes([0.968, bottom, 0.012, top - bottom])
    colorbar = figure.colorbar(artist, cax=colorbar_axis)
    colorbar.set_label(rf"$\xi_{{\mathrm{{{web}}}}}$", fontsize=11)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return figure, axes


def plot_b2_lth_impact(
    data_folder,
    *,
    mass_bin_index=1,
    smoothing_scale=3.0,
    lambda_thresholds=None,
    file_pattern="*_s{smoothing_scale:.0f}_lth*_combined.h5",
    skip_underflow=True,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ratio_ylim=None,
    show_uncertainty=False,
    figsize=(14, 14),
    save_path=None,
    show=True,
):
    """Compare total-sample 1D/3D WCCFs across web thresholds.

    Inputs and outputs match ``pywccf.ratio.plot_lambda_threshold_robustness``.
    Line patterns become denser as ``lambda_threshold`` increases, and the
    largest requested threshold is represented by a solid line.
    """
    data_folder = Path(data_folder)
    search_pattern = file_pattern.format(
        smoothing_scale=float(smoothing_scale)
    )
    candidate_files = sorted(data_folder.glob(search_pattern))
    if not candidate_files:
        raise FileNotFoundError(
            f"No combined HDF5 files match {search_pattern!r} in "
            f"{data_folder}"
        )

    runs_by_threshold = {}
    inventories = {}
    for file_path in candidate_files:
        inventory = combined_h5_inventory(file_path)
        file_scale = float(inventory["smoothing_scale_mpc_h"])
        if not np.isclose(file_scale, smoothing_scale):
            continue
        threshold = _lambda_threshold_from_file(file_path)
        _register_preferred_combined_run(
            runs_by_threshold,
            inventories,
            threshold,
            file_path,
            inventory,
        )

    if lambda_thresholds is None:
        thresholds = tuple(sorted(runs_by_threshold))
    else:
        requested = tuple(float(value) for value in lambda_thresholds)
        selected = []
        for value in requested:
            match = next(
                (
                    known
                    for known in runs_by_threshold
                    if np.isclose(known, value, rtol=0.0, atol=1.0e-12)
                ),
                None,
            )
            if match is None:
                raise FileNotFoundError(
                    f"No combined run for lambda_threshold={value:g}"
                )
            selected.append(match)
        thresholds = tuple(selected)
    if not thresholds:
        raise FileNotFoundError(
            f"No usable lambda-threshold runs in {data_folder}"
        )

    styles = _density_line_styles(thresholds)
    figure, axes = _make_wccf_ratio_axes(figsize)
    pair_count = 0

    for web_index, web in enumerate(WEB_TYPES):
        row, column = divmod(web_index, 2)
        main_axis = axes[row, column, 0]
        ratio_axis = axes[row, column, 1]

        for threshold in thresholds:
            file_path = runs_by_threshold[threshold]
            try:
                result_1d = load_combined_h5(
                    file_path,
                    dimension="1d",
                    sample="total",
                    tail="all",
                    web=web,
                    mass_bin_index=mass_bin_index,
                )
                result_3d = load_combined_h5(
                    file_path,
                    dimension="3d",
                    sample="total",
                    tail="all",
                    web=web,
                    mass_bin_index=mass_bin_index,
                )
            except (KeyError, ValueError) as error:
                print(
                    f"[skip] {file_path.name}, {web}, "
                    f"mass bin {mass_bin_index}: {error}"
                )
                continue

            _draw_wccf_1d_3d_pair(
                main_axis,
                ratio_axis,
                xi_3d=np.asarray(result_3d["xi"], dtype=np.float64),
                sd_3d=np.asarray(result_3d["xi_sd"], dtype=np.float64),
                centers_3d=result_3d["centers"],
                xi_1d=np.asarray(result_1d["xi"], dtype=np.float64),
                sd_1d=np.asarray(result_1d["xi_sd"], dtype=np.float64),
                r_1d=np.asarray(result_1d["centers"], dtype=np.float64),
                line_style=styles[threshold],
                skip_underflow=skip_underflow,
                show_uncertainty=show_uncertainty,
            )
            pair_count += 1

        _format_wccf_ratio_panel(
            main_axis,
            ratio_axis,
            web,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ratio_ylim=ratio_ylim,
        )

    if pair_count == 0:
        plt.close(figure)
        raise FileNotFoundError(
            f"No plottable total-sample records for mass bin "
            f"{mass_bin_index}"
        )

    threshold_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=styles[threshold],
            linewidth=1.8,
            label=rf"$\lambda_{{\rm th}}={threshold:g}$",
        )
        for threshold in thresholds
    ]
    figure.legend(
        handles=_statistic_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=4,
        fontsize=9,
        frameon=True,
    )
    figure.legend(
        handles=threshold_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.950),
        ncol=min(len(threshold_handles), 6),
        fontsize=9,
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.8)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    return figure, axes


# Python identifiers cannot start with a digit.  This string-key alias keeps
# the requested figure number available through getattr(plot, "1_main_xi").
globals()["1_main_xi"] = plot_1_main_xi


__all__ = [
    "plot_1_main_xi",
    "plot_2_main_contour",
    "plot_a1_ss_impact",
    "plot_a2_ss_evolution",
    "plot_b2_lth_impact",
]
