#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_stats_plot.py
Create per-(noise, target) PDF plots comparing the time-evolution of *local* stats
computed on-board by robots (one line per arena), shaded by std across runs.
Also generate final-window violin plots per target.

Inputs:
  - A Pogosim-like CSV/Feather with per-robot rows and columns including at least:
      time (s), arena_file, robot_category, run or runs
    Optionally: noise (str), target (str).
    Required local stat columns: pol_local, wall_local, pers_local, nb_local,
                                ang4_local, vort_local, loss_led, motor_bias.
    For genotypic diversity: beta, sigma, speed, phi_norm, crowd_depth_norm.

Assumptions:
  - Only rows with robot_category == "robots" are considered.
  - If a column 'runs' exists, it will be used (and aliased to 'run').

CLI:
  -i / --input          CSV or Feather file
  -o / --out            Output directory
  --final-window-s      Window (s) before the last timestamp for violin plots; default 25.0
  --nb-max              Maximum neighbors used to normalize nb_local to [0,1]; default 20
  --stats               Comma list from: pol,ang4,vort,pers,nb,wall,loss,geno_div

Outputs:
  - For each (noise, target, stat): one PDF with one line per arena (mean across runs)
    and a shaded std band across runs.
  - For each (target, stat): one PDF violin of final-window distribution across runs
    with one violin per (arena, noise) combination.

Plot style:
  - No subplots; one plot per PDF.
  - Seaborn + LaTeX header; all text uses LaTeX mode.
"""

from __future__ import annotations
import os, argparse, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import math

ARENA_ORDER = ['empty', 'disk', 'star', 'H']

# List of stats used to create the boxplots
#COMBINED_BOX_STATS = ['pol', 'wall', 'pers', 'nb', 'ang4']
COMBINED_BOX_STATS = ['pol', 'wall', 'pers', 'nb']

STAT_LABELS = {
    'pol':  'Polarization',
    'wall': 'Wall-avoidance + U-Turns',
    'pers': 'Neighbor persistence',
    'nb':   'Neighbor count / 20',
    #'ang4': 'Angular balance (S4)',
}

# Per-stat y-limits for VIOLIN plots only (leave empty or comment keys to use autoscale)
VIOLIN_YLIMS = {
     'pol':      (0.0, 1.0),
     'wall':     (0.0, 0.9),
     'pers':     (0.2, 0.7),
     'nb':       (0.10, 0.4),
     'ang4':     (0.3, 0.75),
     'vort':     (0.0, 1.0),
     'loss':     (0.0, 0.7),
     'geno_div': (0.0, 0.4),
     'motor_bias': (-0.25, 0.25),
}

# ------------------------- Plot header ----------------
sns.set(font_scale=1.4)
sns.set_context("talk")
sns.set_style("white")
sns.set_palette("colorblind")
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r''.join([
    r'\usepackage{amsmath}',
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{helvet}",
    r"\renewcommand{\familydefault}{\sfdefault}",
    r"\usepackage[helvet]{sfmath}",
    r"\everymath={\sf}",
    r'\centering',
]))

# ------------------------------ Helpers -------------------------------------

def _arena_disp(a: str) -> str:
    return "torus" if str(a) == "empty" else str(a)


def _arena_noise_to_name(arena_file: str, noise: str) -> str:
    a = "torus" if str(arena_file) == "empty" else str(arena_file)
    n = "" if str(noise) == "noiseless" else "(bias)"
    return f"{a}" if n == "" else f"{a}\n{n}"


def _sanitize(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_') or "NA"


def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".feather", ".ft", ".fth"):
        return pd.read_feather(path)
    return pd.read_csv(path)


def _ensure_columns(df: pd.DataFrame, need: set[str]):
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in dataframe: {sorted(miss)}")


# ------------------------------ Aggregation ---------------------------------

# NOTE: geno_div is handled specially (no single backing column)
STAT_MAP = {
    'pol':      'pol_local',
    'wall':     'wall_local',
    'pers':     'pers_local',
    'nb':       'nb_local',
    'ang4':     'ang4_local',
    'vort':     'vort_local',
    'loss':     'loss_led',
    'geno_div': None,
    'motor_bias': 'motor_bias',
}

GENO_COLS = ['beta', 'sigma', 'speed', 'phi_norm', 'crowd_depth_norm']

def clean_float16_genotype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean float16 genotype columns in-place.

    - Casts to float32.
    - Treats non-finite or out-of-domain values as NaN.
    - Clips to [0,1] (expected genotype domain).
    """
    df = df.copy()
    for c in GENO_COLS:
        if c not in df.columns:
            continue
        col = df[c].astype('float32')

        # 1) Non-finite → NaN
        col[~np.isfinite(col)] = np.nan

        # 2) Domain filter: we expect genotypes in [0,1]
        #    Anything outside [-1, 2] is "obviously wrong" → NaN
        col[(col < -1.0) | (col > 2.0)] = np.nan

        # 3) Clip to [0,1] to be safe
        col = col.clip(0.0, 1.0)

        df[c] = col
    return df

def _normalize_nb(df: pd.DataFrame, nb_max: float) -> pd.Series:
    # Ensure [0,1] with a hard cap
    x = df['nb_local'].astype(float)
    nb_max = max(float(nb_max), 1e-9)
    return (x / nb_max).clip(0.0, 1.0)


def _y_limits_for(stat: str):
    # Special case for motor_bias (domain ~[-0.2,0.2])
    if stat == 'motor_bias':
        return (-0.25, 0.25)
    # Keep everything in [0,1] for line plots
    return (0.0, 1.0)


def _ylabel_for(stat: str) -> str:
    names = dict(
        pol      = r'\textbf{Polarization (local)}',
        ang4     = r'\textbf{Angular entropy $S_4$ (local)}',
        vort     = r'\textbf{Vortex strength $|\hat v\cdot \hat t|$ (local)}',
        pers     = r'\textbf{Neighbor-age persistence (local)}',
        nb       = r'\textbf{Mean degree (norm., local)}',
        wall     = r'\textbf{Wall-contact fraction (local)}',
        loss     = r'\textbf{loss (local)}',
        geno_div = r'\textbf{Genotypic diversity (norm.)}',
        motor_bias = r'\textbf{Motor bias (per-robot)}',
    )
    return names.get(stat, rf'\\textbf{{{stat} — Local}}')

def _gene_label_for(gene: str) -> str:
    labels = dict(
        beta       = r'$\beta$',
        sigma      = r'$\sigma$',
        speed      = r'$v$',
        phi_norm   = r'$\phi_\mathrm{norm}$',
        crowd_depth_norm = r'$d_\mathrm{crowd}$',
        motor_bias = r'\textbf{Motor bias}',
    )
    return labels.get(gene, rf'\textbf{{{gene}}}')


def _make_palette(df: pd.DataFrame) -> dict:
    arenas_all = df['arena_file'].dropna().astype(str).unique().tolist()
    palette_colors = sns.color_palette("colorblind", n_colors=max(3, len(arenas_all)))
    return {a: palette_colors[i % len(palette_colors)] for i, a in enumerate(sorted(arenas_all))}


# --------- Genotypic diversity helper ---------------------------------

def _add_scaled_genotype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add globally scaled genotype columns in [0,1] for GENO_COLS:
    each column c -> c+"_scaled".

    Scaling is global (over entire df), so diversity is comparable across time/runs.
    If a column is constant or NaN, its scaled value is set to 0.
    """
    df = df.copy()
    for c in GENO_COLS:
        col = df[c].astype(float)
        cmin = col.min()
        cmax = col.max()
        if np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin:
            df[c + "_scaled"] = (col - cmin) / (cmax - cmin)
        else:
            df[c + "_scaled"] = 0.0
    return df

def _compute_genotypic_diversity(geno_df: pd.DataFrame) -> float:
    """
    Compute normalized genotypic diversity in [0,1] for a group of robots,
    given only the scaled genotype columns (each in [0,1]).

    Diversity = 4 * mean(variance_per_dimension), clipped to [0,1].
    (Max variance for a [0,1] variable is ~0.25, hence the factor 4.)
    """
    X = geno_df.to_numpy(dtype=float)  # shape: (n_robots, n_dims)
    if X.shape[0] <= 1:
        # Only one robot in this (arena,run,time) group ⇒ no diversity
        return 0.0
    var = X.var(axis=0, ddof=0)        # per-dimension variance
    mean_var = np.nanmean(var)         # average over genotype dimensions
    return float(np.clip(4.0 * mean_var, 0.0, 1.0))


# ------------------------------ Plotting ------------------------------------

def _one_plot_per_combo(A: pd.DataFrame, stat: str, out_dir: str, noise: str, target: str, arena_palette: dict):
    pdf_name = f"{_sanitize(stat)}__noise={_sanitize(noise)}__target={_sanitize(target)}.pdf"
    out_path = os.path.join(out_dir, pdf_name)
    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        for arena, g in A.groupby('arena_file', sort=False):
            g = g.sort_values('time')
            col = arena_palette[str(arena)]
            ax.plot(g['time'], g['mean'], label=_arena_disp(arena), color=col)
            if 'std' in g and np.isfinite(g['std']).any():
                lo = g['mean'] - g['std']
                hi = g['mean'] + g['std']
                ax.fill_between(g['time'], lo, hi, alpha=0.15, color=col)
        ax.set_xlabel(r'\textbf{Time (s)}')
        ax.set_ylabel(_ylabel_for(stat))
        ax.set_ylim(*_y_limits_for(stat))
        ax.grid(True, alpha=0.3)
        ax.legend(title=r'\textbf{Arenas}', ncols=2, fontsize=13, title_fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _violin_final_window(df_series: pd.DataFrame, stat: str, target: str, noise: str, out_dir: str,
                         arena_palette: dict, final_window_s: float, per_robot: bool):
    """
    Aggregate values over the final time window for violin plots.

    per_robot = True:
        - group by (arena_file, run, robot_id)
        - one point per robot per run (e.g. 60 robots × 8 runs = 480 points)

    per_robot = False:
        - group by (arena_file, run)
        - one point per run (previous behavior)
    """
    if df_series.empty:
        return None

    tmax = df_series['time'].max()
    t0 = max(float(tmax) - float(final_window_s), df_series['time'].min())
    dfw = df_series[df_series['time'] >= t0].copy()
    if dfw.empty:
        return None

    # Decide grouping keys
    if per_robot and 'robot_id' in dfw.columns:
        # One point per robot per run (time-averaged)
        group_keys = ['arena_file', 'run', 'robot_id']
    else:
        # Fallback / old behavior: one point per run
        group_keys = ['arena_file', 'run']

    g = (
        dfw
        .groupby(group_keys, sort=False)['val']
        .mean()
        .reset_index(name='val')
    )
    if g.empty:
        return None

    g['target'] = target
    g['noise'] = noise
    g['combo'] = g.apply(lambda r: _arena_noise_to_name(r['arena_file'], r['noise']), axis=1)

    # Keep run (and robot_id if present) for possible debugging; _draw_violin
    # only uses arena_file, noise, target, combo, val.
    cols = ['arena_file', 'noise', 'target', 'run', 'combo', 'val']
    cols = [c for c in cols if c in g.columns]
    return g[cols]



def _draw_violin(dfv: pd.DataFrame, stat: str, target: str, out_dir: str, arena_palette: dict, final_window_s: float):
    if dfv is None or dfv.empty:
        return
    dfv = dfv.copy()
    dfv['arena_disp'] = dfv['arena_file'].map(lambda a: "torus" if a == "empty" else a)
    palette_disp = {("torus" if a == "empty" else a): arena_palette[a]
                    for a in dfv['arena_file'].unique()}

    pdf_name = f"{_sanitize(stat)}__target={_sanitize(target)}__final{int(final_window_s)}s_violin.pdf"
    out_path = os.path.join(out_dir, pdf_name)

    # Ordering to match legend/lines
    uniq_arenas = dfv['arena_file'].astype(str).unique().tolist()
    arenas = [a for a in ARENA_ORDER if a in uniq_arenas] + [a for a in uniq_arenas if a not in ARENA_ORDER]
    noises = sorted(dfv['noise'].unique().tolist())
    arena_disp_order = [('torus' if a == 'empty' else a) for a in arenas]

    ordered = []
    present = set(dfv['combo'])
    for a in arenas:
        for n in noises:
            name = _arena_noise_to_name(a, n)
            if name in present:
                ordered.append(name)

    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=(7, 5.0))

        import matplotlib as _mpl
        sns.violinplot(
            data=dfv,
            x='combo', y='val', hue='arena_disp',
            order=ordered, dodge=False, ax=ax,
            palette=palette_disp, cut=0, inner=None,
            hue_order=arena_disp_order
        )
        for c in ax.collections:
            if isinstance(c, _mpl.collections.PolyCollection):
                c.set_alpha(0.6)

        # Jittered sample markers by noise, colored by arena
        markers = { 'noiseless': 'o', 'strong_noise': 'X' }
        pos_map = {name: i for i, name in enumerate(ordered)}
        jitter = 0.10
        for nz in dfv['noise'].unique():
            sub = dfv[dfv['noise'] == nz]
            if sub.empty:
                continue
            xs = sub['combo'].map(pos_map).astype(float).to_numpy()
            xs = xs + np.random.uniform(-jitter, jitter, size=xs.size)
            colors = [arena_palette[a] for a in sub['arena_file']]
            mk = markers.get(nz, 'D')
            ax.scatter(xs, sub['val'], c=colors, marker=mk, s=20, alpha=0.9, linewidths=0.0, zorder=3)

        # Overlay per-combo median as a short horizontal bar (not a plus)
        medians = dfv.groupby('combo', sort=False)['val'].median()
        xs_median = np.array([pos_map[c] for c in medians.index], dtype=float)
        half_width = 0.20
        for x, y in zip(xs_median, medians.to_numpy()):
            ax.hlines(y, x - half_width, x + half_width, colors='k', linewidth=2, zorder=5)

        ax.set_xlabel(r'\textbf{Arena}')
        ax.set_ylabel(_ylabel_for(stat))
        ylims = VIOLIN_YLIMS.get(stat, None)
        if ylims is not None:
            ax.set_ylim(*ylims)
        ax.grid(True, axis='y', alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_ha('center')
        ax.legend(title=r'\textbf{Arenas}', fontsize=13, title_fontsize=12)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _draw_combined_boxplot(
    df_combined: pd.DataFrame,
    target: str,
    out_dir: str,
    final_window_s: float,
):
    """
    One figure per target:
      - x = (arena, noise/bias) "combo"
      - hue = stat in COMBINED_BOX_STATS
      - y = value in final window

    df_combined must contain columns:
      ['arena_file', 'noise', 'target', 'combo', 'val', 'stat']
    """
    if df_combined is None or df_combined.empty:
        return

    # Only keep the stats we want, in the order of COMBINED_BOX_STATS
    present_set = set(df_combined['stat'])
    stats_present = [s for s in COMBINED_BOX_STATS if s in present_set]
    if not stats_present:
        return

    # Order of x-axis: same arena/noise order as in violins
    uniq_arenas = df_combined['arena_file'].astype(str).unique().tolist()
    arenas = [a for a in ARENA_ORDER if a in uniq_arenas] + [
        a for a in uniq_arenas if a not in ARENA_ORDER
    ]
    noises = sorted(df_combined['noise'].unique().tolist())
    present = set(df_combined['combo'])

    ordered = []
    arena_for_pos = []  # arena name for each x-position
    for a in arenas:
        for n in noises:
            name = _arena_noise_to_name(a, n)
            if name in present:
                ordered.append(name)
                arena_for_pos.append(a)

    if not ordered:
        return

    pdf_name = (
        f"combined_stats_boxplot__target={_sanitize(target)}"
        f"__final{int(final_window_s)}s.pdf"
    )
    out_path = os.path.join(out_dir, pdf_name)

    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=(7, 6.0))

        # Thin boxes: much narrower than violins
        sns.boxplot(
            data=df_combined,
            x='combo',
            y='val',
            hue='stat',
            order=ordered,
            hue_order=stats_present,
            palette="colorblind",
            width=0.70,
            dodge=True,
            fliersize=1.5,
            linewidth=0.8,
            ax=ax,
        )

        ax.set_xlabel(r'\textbf{Arena / bias}')
        ax.set_ylabel(r'\textbf{Local statistic value}')
        ax.set_ylim(0.0, 1.0)  # all these stats are normalized / in [0,1]
        ax.grid(True, axis='y', alpha=0.3)

        # --- vertical separators between arenas ---
        ymin, ymax = ax.get_ylim()
        for i in range(1, len(arena_for_pos)):
            # separator between position i-1 and i
            ax.axvline(
                x=i - 0.5,
                linestyle='--',
                linewidth=0.8,
                alpha=0.4,
            )
        ax.set_ylim(ymin, ymax)  # restore limits (vlines can slightly change them)

        for label in ax.get_xticklabels():
            label.set_ha('center')

        # Legend with full stat names
        handles, labels = ax.get_legend_handles_labels()
        nice_labels = [STAT_LABELS.get(l, l) for l in labels]
        ax.legend(
            handles,
            nice_labels,
            title=r'\textbf{Statistics (mean over all robots in the last ' + str(int(final_window_s)) + 's of each run)}',
            fontsize=12,
            title_fontsize=12,
            ncols=2,
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.2,
        )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[plot] wrote {out_path}")


def _draw_combined_boxplot_per_target(
    df_summary: pd.DataFrame,
    out_dir: str,
    final_window_s: float,
):
    """
    One figure for all targets:
      - x = target
      - hue = stat in COMBINED_BOX_STATS
      - y = mean value over (arena,bias) combos (one sample per combo)

    df_summary must contain:
      ['target', 'stat', 'combo', 'val']
      where 'val' is one number per (target, stat, combo),
      typically the mean over robots/runs in the final window.
    """
    if df_summary is None or df_summary.empty:
        return

    # Keep only the stats we care about, in a fixed order
    present_set = set(df_summary['stat'])
    stats_present = [s for s in COMBINED_BOX_STATS if s in present_set]
    if not stats_present:
        return

    targets = sorted(df_summary['target'].astype(str).unique().tolist())

    pdf_name = (
        f"combined_stats_boxplot__per_target__final{int(final_window_s)}s.pdf"
    )
    out_path = os.path.join(out_dir, pdf_name)

    with PdfPages(out_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6.0))

        sns.boxplot(
            data=df_summary,
            x='target',
            y='val',
            hue='stat',
            order=targets,
            hue_order=stats_present,
            palette="colorblind",
            width=0.70,
            dodge=True,
            fliersize=1.5,
            linewidth=0.8,
            ax=ax,
        )

        ax.set_xlabel(r'\textbf{Goal}')
        ax.set_ylabel(r'\textbf{Local statistic value}')
        ax.set_ylim(0.0, 1.0)  # these stats are normalized / in [0,1]
        ax.grid(True, axis='y', alpha=0.3)

        # Legend with full stat names (same style as combined_stats_boxplot)
        handles, labels = ax.get_legend_handles_labels()
        nice_labels = [STAT_LABELS.get(l, l) for l in labels]
        ax.legend(
            handles,
            nice_labels,
            title=(
                r'\textbf{Statistics '
                r'(variation across $(\mathrm{arena},\mathrm{bias})$ couples; '
                r'final ' + str(int(final_window_s)) + r'\,s)}'
            ),
            fontsize=12,
            title_fontsize=12,
            ncols=2,
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.2,
        )

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[plot] wrote {out_path}")



def _plot_genotype_heatmaps(
    G: pd.DataFrame,
    out_dir: str,
    noise: str,
    target: str,
    time_bins: int,
    val_bins: int,
):
    """
    For each genotype component, create a 2D histogram heatmap:
    - x: time
    - y: value of the component (in [0,1] for GENO_COLS, [-0.25,0.25] for motor_bias)
    - color: proportion of robot values in that (time, value) bin.

    Aggregation is over all arenas and runs for the given (noise, target).
    """
    # Which genome components to plot
    genes = list(GENO_COLS)
    if 'motor_bias' in G.columns:
        genes.append('motor_bias')

    for gene in genes:
        if gene not in G.columns:
            continue

        sub = G[['time', gene]].dropna().copy()
        if sub.empty:
            continue

        # Time range
        tmin = float(sub['time'].min())
        tmax = float(sub['time'].max())
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            continue

        # Value range
        if gene == 'motor_bias':
            vmin, vmax = -0.25, 0.25
            vals = sub[gene].astype(float).to_numpy()
        else:
            vmin, vmax = 0.0, 1.0
            vals = sub[gene].astype(float).clip(0.0, 1.0).to_numpy()

        times = sub['time'].astype(float).to_numpy()

        # 2D histogram: H[time_bin, value_bin]
        time_edges = np.linspace(tmin, tmax, time_bins + 1)
        val_edges = np.linspace(vmin, vmax, val_bins + 1)

        H, xedges, yedges = np.histogram2d(
            times, vals,
            bins=[time_edges, val_edges]
        )  # H shape: (time_bins, val_bins)

        # Normalize each time column to sum to 1 (proportions)
        # Axis 1 is values; axis 0 is time
        col_sums = H.sum(axis=1, keepdims=True)
        col_sums[col_sums == 0.0] = 1.0
        H_norm = H / col_sums

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 5))

        # imshow expects [row, col] with row -> y, col -> x
        # Transpose so that y corresponds to value bins
        im = ax.imshow(
            H_norm.T,
            origin='lower',
            aspect='auto',
            extent=[tmin, tmax, vmin, vmax],
            interpolation='nearest',
            cmap="cividis"
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'\textbf{Proportion of robots}', rotation=90)

        ax.set_xlabel(r'\textbf{Time (s)}')
        ax.set_ylabel(_gene_label_for(gene))
        ax.set_ylim(vmin, vmax)
        ax.grid(False)

        fig.tight_layout()

        pdf_name = (
            f"geno_heatmap__{_sanitize(gene)}"
            f"__noise={_sanitize(noise)}"
            f"__target={_sanitize(target)}.pdf"
        )
        out_path = os.path.join(out_dir, pdf_name)
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


def _plot_genotype_heatmaps_grid(
    G: pd.DataFrame,
    out_dir: str,
    noise: str,
    target: str,
    time_bins: int,
    val_bins: int,
):
    """
    For a given (noise, target), build a single figure with a grid of heatmaps
    (one per genotype component):
      - columns = ceil(n_genes / 3)
      - rows    = 3
    x: time
    y: gene value (in [0,1] for GENO_COLS, [-0.25,0.25] for motor_bias)
    color: proportion of robots in that (time, value) bin.
    """
    # Which genes we want to show
    genes = list(GENO_COLS)
    if 'motor_bias' in G.columns:
        genes.append('motor_bias')

    # Filter to genes that actually exist and have data
    gene_data = []
    for gene in genes:
        if gene not in G.columns:
            continue
        sub = G[['time', gene]].dropna().copy()
        if sub.empty:
            continue
        gene_data.append((gene, sub))

    if not gene_data:
        return

    # Precompute histograms for each gene
    histograms = []
    for gene, sub in gene_data:
        tmin = float(sub['time'].min())
        tmax = float(sub['time'].max())
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            continue

        if gene == 'motor_bias':
            vmin, vmax = -0.25, 0.25
            vals = sub[gene].astype(float).to_numpy()
        else:
            vmin, vmax = 0.0, 1.0
            vals = sub[gene].astype(float).clip(0.0, 1.0).to_numpy()

        times = sub['time'].astype(float).to_numpy()

        time_edges = np.linspace(tmin, tmax, time_bins + 1)
        val_edges = np.linspace(vmin, vmax, val_bins + 1)

        H, xedges, yedges = np.histogram2d(
            times, vals,
            bins=[time_edges, val_edges],
        )

        col_sums = H.sum(axis=1, keepdims=True)
        col_sums[col_sums == 0.0] = 1.0
        H_norm = H / col_sums

        histograms.append(
            dict(
                gene=gene,
                H_norm=H_norm,
                tmin=tmin,
                tmax=tmax,
                vmin=vmin,
                vmax=vmax,
            )
        )

    if not histograms:
        return

    n = len(histograms)
    nrows = 3
    ncols = math.ceil(n / nrows)

    # Compact but readable figure size
    fig_width = 1.0 + 4.0 * ncols
    fig_height = 2.5 * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex='col',
    )

    # We’ll use the same colormap scale for all (proportions in [0,1])
    vmin_c = 0.0
    vmax_c = 0.2 # 1.0
    last_im = None

    for idx, info in enumerate(histograms):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        gene = info['gene']
        H_norm = info['H_norm']
        tmin = info['tmin']
        tmax = info['tmax']
        vmin = info['vmin']
        vmax = info['vmax']

        im = ax.imshow(
            H_norm.T,
            origin='lower',
            aspect='auto',
            extent=[tmin, tmax, vmin, vmax],
            interpolation='nearest',
            cmap='cividis',  # colorblind-friendly
            vmin=vmin_c,
            vmax=vmax_c,
        )
        last_im = im

        ax.set_title(_gene_label_for(gene), fontsize=18)
        ax.set_ylim(vmin, vmax)

        # Only bottom row gets x-labels for compactness
        if row == nrows - 1:
            ax.set_xlabel(r'\textbf{Time (s)}')
        else:
            ax.set_xticklabels([])

        # Only first column gets y-labels
        if col == 0:
            ax.set_ylabel(r'\textbf{Value}')
        else:
            ax.set_yticklabels([])

        ax.grid(False)

    # Hide any unused axes
    for idx in range(len(histograms), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    # Global title (small, to stay compact)
    fig.suptitle(
        rf"Genome distributions – noise={_sanitize(noise)}, goal={_sanitize(target)}",
        fontsize=11
    )

    # One shared colorbar on the right
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label(r'\textbf{Proportion of robots}', rotation=90)

    # Manual compact layout (no tight_layout to avoid warnings)
    fig.subplots_adjust(
        left=0.10,
        right=0.85,   # leave space for colorbar
        bottom=0.08,
        top=0.90,
        wspace=0.25,
        hspace=0.30,
    )

    pdf_name = (
        "geno_heatmaps_grid__"
        f"noise={_sanitize(noise)}__target={_sanitize(target)}.pdf"
    )

    out_path = os.path.join(out_dir, pdf_name)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _draw_phenotypic_diversity_violin(
    df_pheno: pd.DataFrame,
    out_dir: str,
    final_window_s: float,
):
    """
    Phenotypic diversity violin plots.

    - One PDF per bias/noise:
        x = target
        hue = stat (colored with colorblind cmap)
        y = phenotypic diversity in [0,1]
        each point = one run

    - One additional PDF where all biases/noise levels are merged.

    df_pheno must contain:
      ['target', 'noise', 'stat', 'run', 'diversity']
    """
    if df_pheno is None or df_pheno.empty:
        return

    df = df_pheno.copy()

    # Map noise → "bias" label (No / Yes) for display / filenames
    def _bias_label(n: str) -> str:
        n = str(n)
        if n == 'noiseless':
            return 'No'
        if n == 'strong_noise':
            return 'Yes'
        if n == 'NA':
            return 'NA'
        return n

    def _draw_subset(df_sub: pd.DataFrame, file_suffix: str, title: str | None):
        """Draw a single phenotypic diversity figure for the given subset."""
        if df_sub is None or df_sub.empty:
            return

        df_s = df_sub.copy()
        df_s['target'] = df_s['target'].astype(str)
        df_s['stat'] = df_s['stat'].astype(str)

        # Order of targets on x-axis
        target_order = sorted(df_s['target'].unique().tolist())

        # Stats order: COMBINED_BOX_STATS intersection with present ones
        stats_present = df_s['stat'].unique().tolist()
        stat_codes = [s for s in COMBINED_BOX_STATS if s in stats_present]
        if not stat_codes:
            # fall back to whatever is present
            stat_codes = sorted(stats_present)

        # Human-readable labels for stats (for legend)
        stat_labels = [STAT_LABELS.get(s, s) for s in stat_codes]

        # Map code → label
        code_to_label = {c: l for c, l in zip(stat_codes, stat_labels)}
        df_s['stat_label'] = df_s['stat'].map(code_to_label).fillna(df_s['stat'])

        # Colorblind palette for the stats
        palette_colors = sns.color_palette("colorblind", n_colors=len(stat_labels))
        palette = {label: color for label, color in zip(stat_labels, palette_colors)}

        pdf_name = (
            f"phenotypic_diversity_violin__{file_suffix}"
            f"__final{int(final_window_s)}s.pdf"
        )
        out_path = os.path.join(out_dir, pdf_name)

        with PdfPages(out_path) as pdf:
            fig_width = 7 + 0.6 * len(target_order) if len(target_order) > 1 else 3.0
            fig_height = 6.0 if len(target_order) > 1 else 5.0
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Handle special case: only one target "NA"
            only_na_target = (len(target_order) == 1 and target_order[0] == 'NA')

            sns.violinplot(
                data=df_s,
                x='target',
                y='diversity',
                hue='stat_label',
                order=target_order,
                hue_order=stat_labels,
                palette=palette,
                cut=0,
                inner='box',
                dodge=True,
                ax=ax,
                legend=not only_na_target,
            )

            if only_na_target:
                # No xtick, no xlabel
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel('')
            else:
                # Replace '_' by ' ' in xtick labels
                nice_labels = [t.replace('_', ' ') for t in target_order]
                ax.set_xticklabels(nice_labels)
                ax.set_xlabel(r'\textbf{Goal}')

#                # Rotate x tick labels if many targets
#                for label in ax.get_xticklabels():
#                    label.set_rotation(10)
#                    #label.set_ha('right')

            ax.set_ylabel(r'\textbf{Phenotypic diversity}')
            #ax.set_ylim(0.0, 1.0)
            ax.grid(True, axis='y', alpha=0.3)

            # Vertical separators between targets
            if (not only_na_target) and len(target_order) > 1:
                ymin, ymax = ax.get_ylim()
                for i in range(1, len(target_order)):
                    # separator between target i-1 and i
                    ax.axvline(
                        x=i - 0.5,
                        linestyle='--',
                        linewidth=0.8,
                        alpha=0.4,
                    )
                ax.set_ylim(ymin, ymax)  # restore limits (vlines can slightly change them)

            if title:
                ax.set_title(title)

            # Legend: stats with colors
            if not only_na_target:
                leg = ax.legend(
                    title=r'\textbf{Statistic}',
                    #loc='upper right',
                    frameon=True,
                    fontsize=13,
                    title_fontsize=15,
                    ncols=1,
                )
                leg._legend_box.align = "left"

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"[plot] wrote {out_path}")

    # --- One figure per noise/bias ---
    noises = sorted(df['noise'].astype(str).unique().tolist())
    for noise in noises:
        df_b = df[df['noise'] == noise]
        if df_b.empty:
            continue
        bias_lbl = _bias_label(noise)
        title_txt = rf"\textbf{{Bias: {bias_lbl}}}"
        file_suffix = f"bias-{bias_lbl}"
        _draw_subset(df_b, file_suffix=file_suffix, title=title_txt)

    # --- One merged figure over all biases/noise levels ---
    if len(noises) > 1:
        #_draw_subset(df, file_suffix="bias-All", title=r"\textbf{All biases}")
        _draw_subset(df, file_suffix="bias-All", title=r"")




# ------------------------------ Main ---------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Local stats plots across arenas, per (noise,target), plus final-window violins.")
    ap.add_argument("-i", "--input", required=True, help="Input CSV/Feather with per-robot logs.")
    ap.add_argument("-o", "--out", required=True, help="Output directory for PDFs.")
    ap.add_argument("--final-window-s", type=float, default=25.0, help="Final-window length in seconds for violin plots. Default 25.0")
    ap.add_argument("--nb-max", type=float, default=20.0, help="Max neighbors to normalize nb_local to [0,1]. Default 20")
    ap.add_argument(
        "--stats", type=str,
        default="pol,ang4,vort,pers,nb,wall,loss,geno_div,motor_bias",
        help="Comma list from: pol,ang4,vort,pers,nb,wall,loss,geno_div,motor_bias"
    )
    ap.add_argument(
        "--violin-per-robot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true (default), violin points are one mean value per robot (per run) over the final window. "
             "If false, points are one value per run (previous behavior)."
    )
    ap.add_argument(
        "--geno-heatmaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate time-vs-genotype distribution heatmaps per (noise,target). "
             "Use --no-geno-heatmaps to disable."
    )
    ap.add_argument(
        "--geno-heatmap-time-bins",
        type=int, default=80,
        help="Number of time bins for genotype heatmaps (default: 80)."
    )
    ap.add_argument(
        "--geno-heatmap-val-bins",
        type=int, default=40,
        help="Number of value bins for genotype heatmaps (default: 40)."
    )
    ap.add_argument(
        "--geno-heatmaps-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a compact multi-panel genotype heatmap per (noise,target). "
             "Use --no-geno-heatmaps-grid to disable."
    )
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    df = _read_any(args.input)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Only robots
    if 'robot_category' in df.columns:
        df = df[df['robot_category'] == 'robots'].copy()

    # Clean float16 genotype columns
    df = clean_float16_genotype(df)

    # Accept 'runs' or 'run'
    if 'runs' in df.columns and 'run' not in df.columns:
        df = df.rename(columns={'runs': 'run'})

    # Ensure noise/target exist (gracefully handle missing)
    if 'noise' not in df.columns:
        df['noise'] = 'NA'
    if 'target' not in df.columns:
        df['target'] = 'NA'

    # Parse requested stats
    stats_wanted = [s.strip() for s in args.stats.split(',') if s.strip()]
    valid = set(STAT_MAP.keys())
    for s in stats_wanted:
        if s not in valid:
            raise ValueError(f"Unknown stat '{s}'. Choose among: {sorted(valid)}")

    # Check only the columns actually needed
    need = {'time', 'arena_file', 'run'}
    # Local stat backing columns
    need |= {STAT_MAP[s] for s in stats_wanted if STAT_MAP[s] is not None}
    # Genotype columns if needed
    if 'geno_div' in stats_wanted:
        need |= set(GENO_COLS)
    _ensure_columns(df, need)

    # Normalize nb_local → nb_local_norm
    df = df.copy()
    df['nb_local_norm'] = _normalize_nb(df, args.nb_max)

    # Add globally scaled genotype if needed
    if 'geno_div' in stats_wanted:
        df = _add_scaled_genotype(df)

    arena_palette = _make_palette(df)

    # Collector for final-window violins (across noises)
    collector: dict[tuple[str,str], list[pd.DataFrame]] = {}
    # Collector for phenotypic diversity (across arenas, per run)
    pheno_records: list[dict] = []

    for (noise, target), G in df.groupby(['noise','target'], sort=False):
        if G.empty:
            continue

        # Final-window bounds for this (noise, target)
        tmax_group = float(G['time'].max())
        tmin_group = float(G['time'].min())
        t0_group = max(tmin_group, tmax_group - float(args.final_window_s))

        # Genome distribution heatmaps per (noise, target)
        if args.geno_heatmaps:
            _plot_genotype_heatmaps(
                G,
                out_dir=args.out,
                noise=noise,
                target=target,
                time_bins=args.geno_heatmap_time_bins,
                val_bins=args.geno_heatmap_val_bins,
            )

        if args.geno_heatmaps_grid:
            _plot_genotype_heatmaps_grid(
                G,
                out_dir=args.out,
                noise=noise,
                target=target,
                time_bins=args.geno_heatmap_time_bins,
                val_bins=args.geno_heatmap_val_bins,
            )

        # Time-evolution line plots per (noise,target,stat)
        for stat in stats_wanted:

            if stat == 'geno_div':
                # ----- geno_div: same as before (always per-run) -----
                cols_scaled = [c + "_scaled" for c in GENO_COLS]
                series = G[['arena_file', 'run', 'time'] + cols_scaled].dropna(subset=cols_scaled).copy()
                if series.empty:
                    continue

                g1 = (
                    series
                    .groupby(['arena_file', 'run', 'time'], sort=False)
                    .apply(
                        lambda grp: _compute_genotypic_diversity(grp[cols_scaled]),
                        include_groups=False
                    )
                    .reset_index(name='val')
                )

                # For lines, geno_div is already per (arena,run,time)
                series_for_lines = g1
                # For violins, geno_div conceptually stays per run
                series_for_violin = g1.assign(noise=noise)

            else:
                # ----- usual local stats: build per-robot series first -----
                col = STAT_MAP[stat]
                base_cols = ['arena_file', 'run', 'time', col]
                if 'robot_id' in G.columns:
                    base_cols.append('robot_id')

                series_raw = G[base_cols].dropna(subset=[col]).copy()
                if series_raw.empty:
                    continue

                if stat == 'nb':
                    # nb uses normalized degree
                    series_raw['val'] = G.loc[series_raw.index, 'nb_local_norm']
                elif stat == 'motor_bias':
                    # motor_bias lives in [-0.2,0.2], do NOT clip to [0,1]
                    series_raw['val'] = series_raw[col].astype(float)
                else:
                    # other stats are probabilities / normalized [0,1]
                    series_raw['val'] = series_raw[col].astype(float).clip(0.0, 1.0)

                # --- Phenotypic diversity per (noise, target, stat, run) ---
                # Only for selected stats (phenotypic stats in [0,1])
                if stat in COMBINED_BOX_STATS:  # ['pol', 'wall', 'pers', 'nb']
                    # Restrict to final window for this (noise, target)
                    series_win = series_raw[series_raw['time'] >= t0_group].copy()
                    if not series_win.empty:
                        # 1) Mean over time per (run, arena, robot)
                        group_keys = ['run', 'arena_file']
                        if 'robot_id' in series_win.columns:
                            group_keys.append('robot_id')

                        per_robot = (
                            series_win
                            .groupby(group_keys, sort=False)['val']
                            .mean()
                            .reset_index()
                        )

                        # 2) Mean over robots per (run, arena) → one value per arena
                        per_arena = (
                            per_robot
                            .groupby(['run', 'arena_file'], sort=False)['val']
                            .mean()
                            .reset_index()
                        )

                        # 3) For each run, variance between arenas of these per-arena means
                        for run_id, g_run in per_arena.groupby('run', sort=False):
                            vals = g_run['val'].to_numpy(dtype=float)
                            if vals.size <= 1:
                                # If only one arena, no between-arena variance
                                diversity = 0.0
                            else:
                                var = float(np.var(vals, ddof=0))
                                # stats in [0,1] ⇒ max var ≈ 0.25; rescale to [0,1]
                                diversity = float(np.clip(4.0 * var, 0.0, 1.0))

                            pheno_records.append(
                                dict(
                                    noise=noise,
                                    target=target,
                                    stat=stat,
                                    run=run_id,
                                    diversity=diversity,
                                )
                            )


                # For lines: average over robots within (arena,run,time)
                series_for_lines = (
                    series_raw
                    .groupby(['arena_file', 'run', 'time'], sort=False)['val']
                    .mean()
                    .reset_index()
                )

                # For violins:
                #   - per_robot=True -> keep per-robot data (needs robot_id)
                #   - per_robot=False or missing robot_id -> per-run data
                if args.violin_per_robot and 'robot_id' in series_raw.columns:
                    series_for_violin = series_raw.assign(noise=noise)
                else:
                    series_for_violin = series_for_lines.assign(noise=noise)

            # -------- line plot aggregation (unchanged in spirit) --------
            A = (
                series_for_lines
                .groupby(['arena_file', 'time'], sort=False)['val']
                .agg(mean='mean', std='std')
                .reset_index()
            )
            if A.empty:
                continue
            _one_plot_per_combo(A, stat=stat, out_dir=args.out, noise=noise, target=target, arena_palette=arena_palette)

            # -------- collect for violins --------
            dfv_part = _violin_final_window(
                series_for_violin,
                stat, target, noise, args.out, arena_palette,
                args.final_window_s,
                per_robot=args.violin_per_robot
            )
            if dfv_part is not None and not dfv_part.empty:
                collector.setdefault((target, stat), []).append(dfv_part)


    # Draw one violin PDF per (target, stat), concatenating all noises
    # and at the same time build combined boxplot data per target.
    combined_by_target: dict[str, list[pd.DataFrame]] = {}

    for (target, stat), parts in collector.items():
        dfv = pd.concat(parts, axis=0, ignore_index=True)

        # Existing per-stat violin
        _draw_violin(dfv, stat, target, args.out, arena_palette, args.final_window_s)

        # Feed combined boxplot for selected stats
        if stat in COMBINED_BOX_STATS:
            dfv2 = dfv.copy()
            dfv2['stat'] = stat
            combined_by_target.setdefault(target, []).append(dfv2)

    # Now draw one combined boxplot per target (no subplots)
    for target, lst in combined_by_target.items():
        df_all = pd.concat(lst, axis=0, ignore_index=True)
        _draw_combined_boxplot(
            df_all,
            target=target,
            out_dir=args.out,
            final_window_s=args.final_window_s,
        )

    # Summarize variation across (arena,bias) couples per target
    summary_frames = []
    for target, lst in combined_by_target.items():
        df_all = pd.concat(lst, axis=0, ignore_index=True)
        # One sample per (target, stat, combo): mean over robots/runs
        g = (
            df_all
            .groupby(['target', 'stat', 'combo'], sort=False)['val']
            .mean()
            .reset_index(name='val')
        )
        summary_frames.append(g)

    if summary_frames:
        df_summary = pd.concat(summary_frames, axis=0, ignore_index=True)
        _draw_combined_boxplot_per_target(
            df_summary,
            out_dir=args.out,
            final_window_s=args.final_window_s,
        )

    # --- Phenotypic diversity violin (across arenas, per run) ---
    if pheno_records:
        df_pheno = pd.DataFrame(pheno_records)
        _draw_phenotypic_diversity_violin(
            df_pheno,
            out_dir=args.out,
            final_window_s=args.final_window_s,
        )




if __name__ == "__main__":
    import sys
    sys.exit(main())

