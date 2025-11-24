#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
result_summary_tables.py

Given a directory that contains several Feather result files with names:

    result_{algorithm}_{fast_transmission_strategy}.feather

this script:

1) Assembles all files into a single dataframe, adding columns:
       - algorithm
       - fast_transmission_strategy
   inferred from the filename (using the last '_' as separator).

2) Builds a LaTeX table ("loss_targets_summary.tex") where each row is one
   combination:
       (algorithm, fast_transmission_strategy, arena_file, noise)
   and columns are times (seconds, mean ± std across runs) to first reach
   the run-averaged loss thresholds:

       L <= 0.3, 0.25, 0.2, 0.15, 0.10

   Loss is computed consistently with local_stats_plot.py:
   - use loss_led,
   - clean non-finite values,
   - clip to [0,1],
   - average over robots per (time, run),
   - time-to-target is computed on these run-averaged curves.

   For each (arena, noise) pair, the smallest mean time per column
   (over all (algorithm, fast_transmission_strategy)) is typeset in bold.

3) Builds a second LaTeX table ("geno_diversity_summary.tex") where each row is:
       (algorithm, fast_transmission_strategy, noise)
   and the single data column is final-window genotypic diversity
   (mean ± std across runs and arenas), over the last W seconds per run,
   where W is --final-window-s (default: 100).
"""

from __future__ import annotations
import os
import re
import argparse

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import cblind as cb

sns.set(font_scale=1.5)
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


# Arena ordering for nicer LaTeX tables
ARENA_ORDER_RAW = ['empty', 'disk', 'star', 'H']

FAST_TX_DISPLAY_MAP = {
    "never": "No",
    "prob": "Yes",
}

NOISE_DISPLAY_MAP = {
    "noiseless": "No",
    "strong_noise": "Yes",
}

ALGO_DISPLAY_MAP = {
    "es1p1": "1+1-ES",
    "hit": "HIT",
}

# Genotypic/Phenotypic diversity columns
GENO_COLS = ['beta', 'sigma', 'speed', 'phi_norm', 'crowd_depth_norm']
PHENO_COLS = ['pol_local', 'wall_local', 'pers_local', 'nb_local']

# Minimum number of runs that must reach a target for it to be considered
MIN_RUNS_PER_TARGET = 5



# ---------------------------------------------------------------------------
# Helpers for filenames and basic sanity
# ---------------------------------------------------------------------------

def _disp_algorithm(a: str) -> str:
    return ALGO_DISPLAY_MAP.get(str(a), str(a))

def _disp_fast_transmission(fts: str) -> str:
    return FAST_TX_DISPLAY_MAP.get(str(fts), str(fts))

def _disp_noise(n: str) -> str:
    return NOISE_DISPLAY_MAP.get(str(n), str(n))


def _parse_result_filename(fname: str) -> tuple[str | None, str | None]:
    """
    Parse "result_{algorithm}_{fast_transmission_strategy}.feather"
    using the last '_' as separator, so that 'algorithm' can contain underscores.

    Returns (algorithm, fast_transmission_strategy) or (None, None) if pattern
    does not match.
    """
    base = os.path.basename(fname)
    if not (base.startswith("result_") and base.endswith(".feather")):
        return None, None
    core = base[len("result_"):-len(".feather")]
    if "_" not in core:
        return None, None
    algo, fts = core.rsplit("_", 1)
    return algo, fts


def _arena_disp(a: str) -> str:
    """Display name for arenas; 'empty' is shown as 'torus'."""
    a = str(a)
    return "torus" if a == "empty" else a


def _sanitize(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_') or "NA"


def _ensure_columns(df: pd.DataFrame, need: set[str]):
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in dataframe: {sorted(miss)}")

def _fmt_count(n) -> str:
    """
    Format a run-count as a LaTeX string.
    If n is not finite or <= 0, return an em dash.
    """
    if n is None or not np.isfinite(n) or n <= 0:
        return r"\textemdash"
    return str(int(n))


# ---------------------------------------------------------------------------
# Genotype cleaning and diversity helpers
# ---------------------------------------------------------------------------


def clean_float16_genotype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean float16 genotype columns in-place:

    - Casts to float32.
    - Treats non-finite or obviously out-of-domain values as NaN.
    - Clips to [0,1] (assumed genotype domain).
    """
    df = df.copy()
    for c in GENO_COLS:
        if c not in df.columns:
            continue
        col = df[c].astype('float32')

        # Non-finite → NaN
        col[~np.isfinite(col)] = np.nan

        # Values far outside [-1, 2] are invalid → NaN
        col[(col < -1.0) | (col > 2.0)] = np.nan

        # Clip to [0,1]
        col = col.clip(0.0, 1.0)

        df[c] = col
    return df

def clean_float16_pheno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean float16 phenotypic statistic columns in-place:

    - Casts to float32.
    - Treats non-finite or obviously out-of-domain values as NaN.
    - Clips to [0,1] (assumed domain for pol, wall, pers, nb).

    This mirrors clean_float16_genotype, but uses PHENO_COLS instead.
    """
    df = df.copy()
    for c in PHENO_COLS:
        if c not in df.columns:
            continue
        col = df[c].astype('float32')

        # Non-finite → NaN
        col[~np.isfinite(col)] = np.nan

        # Values far outside [-1, 2] are invalid → NaN
        col[(col < -1.0) | (col > 2.0)] = np.nan

        # Clip to [0,1]
        col = col.clip(0.0, 1.0)

        df[c] = col

    return df



def _add_scaled_genotype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add globally scaled genotype columns in [0,1] for GENO_COLS:
    each column c -> c + "_scaled".

    Scaling is global over the entire dataframe.
    """
    df = df.copy()
    for c in GENO_COLS:
        if c not in df.columns:
            continue
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
    """
    X = geno_df.to_numpy(dtype=float)
    if X.shape[0] <= 1:
        return 0.0
    var = X.var(axis=0, ddof=0)
    mean_var = np.nanmean(var)
    return float(np.clip(4.0 * mean_var, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core loading / assembling
# ---------------------------------------------------------------------------

def load_all_results(indir: str) -> pd.DataFrame:
    """
    Load all ".feather" files in the given directory whose name matches the
    "result_{algo}_{fts}.feather" pattern, and assemble into one dataframe,
    adding columns 'algorithm' and 'fast_transmission_strategy'.
    """
    records = []
    for fname in sorted(os.listdir(indir)):
        if not fname.endswith(".feather"):
            continue
        algo, fts = _parse_result_filename(fname)
        if algo is None or fts is None:
            print(f"[warn] Skipping non-matching file: {fname}")
            continue
        path = os.path.join(indir, fname)
        print(f"[load] {path} (algo={algo}, fts={fts})")
        df = pd.read_feather(path)
        df['algorithm'] = algo
        df['fast_transmission_strategy'] = fts
        records.append(df)

    if not records:
        raise RuntimeError(f"No matching Feather files found in directory: {indir}")
    df_all = pd.concat(records, axis=0, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan)

    # Keep only robots if robot_category is present
    if 'robot_category' in df_all.columns:
        df_all = df_all[df_all['robot_category'] == 'robots'].copy()

    return df_all


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _fmt_mean_std(mu: float, sigma: float, is_time: bool, bold: bool = False) -> str:
    """
    Format "mu ± sigma" as a LaTeX string.
    If mu is NaN or not finite, returns '\textemdash'.

    If bold is True, wrap in \\mathbf{...}.
    """
    if mu is None or not np.isfinite(mu):
        return r"\textemdash"
    if sigma is None or not np.isfinite(sigma):
        sigma = 0.0
    if is_time:
        base = f"{mu:.1f} \\pm {sigma:.1f}"
    else:
        base = f"{mu:.3f} \\pm {sigma:.3f}"
    return f"$\\mathbf{{{base}}}$" if bold else f"${base}$"


# ---------------------------------------------------------------------------
# Loss targets summary table
# ---------------------------------------------------------------------------

def compute_per_run_loss_stats(
    df: pd.DataFrame,
    loss_targets: list[float],
) -> pd.DataFrame:
    """
    For each (algorithm, fast_transmission_strategy, arena_file, noise, run),
    compute:

    - time to first reach loss_led <= target for each target in loss_targets

    Loss is computed consistently with local_stats_plot.py:
    - clean non-finite values,
    - clip to [0,1],
    - average over robots per (time, run).
    """

    required = {
        'time', 'run', 'arena_file', 'noise', 'loss_led',
        'algorithm', 'fast_transmission_strategy',
    }
    _ensure_columns(df, required)

    # Make loss consistent with local_stats_plot.py
    series_raw = df[['algorithm', 'fast_transmission_strategy',
                     'arena_file', 'noise', 'run', 'time', 'loss_led']].copy()
    loss = series_raw['loss_led'].astype(float)
    loss[~np.isfinite(loss)] = np.nan
    loss = loss.clip(0.0, 1.0)
    series_raw['val'] = loss

    # Aggregate over robots: one value per (algo, fts, arena, noise, run, time)
    series_for_lines = (
        series_raw
        .groupby(
            ['algorithm', 'fast_transmission_strategy',
             'arena_file', 'noise', 'run', 'time'],
            sort=False
        )['val']
        .mean()
        .reset_index()
    )

    groups = series_for_lines.groupby(
        ['algorithm', 'fast_transmission_strategy', 'arena_file', 'noise', 'run'],
        sort=False
    )

    records = []
    for (algo, fts, arena, noise, run), g in groups:
        g = g.sort_values('time')
        times = g['time'].to_numpy(dtype=float)
        losses = g['val'].to_numpy(dtype=float)

        rec: dict[str, object] = {
            'algorithm': algo,
            'fast_transmission_strategy': fts,
            'arena_file': str(arena),
            'noise': str(noise),
            'run': run,
        }

        # time to target (on run-averaged loss)
        for thr in loss_targets:
            mask = losses <= thr
            if mask.any():
                t_hit = float(times[mask][0])
            else:
                t_hit = np.nan  # will show as '—' in the LaTeX table
            rec[f"t_le_{thr:g}"] = t_hit

        records.append(rec)

    return pd.DataFrame.from_records(records)


def aggregate_loss_per_combo(
    df_runs: pd.DataFrame,
    loss_targets: list[float],
) -> pd.DataFrame:
    """
    Aggregate per-run stats to
       (algorithm, fast_transmission_strategy, arena_file, noise)
    using mean, std and count (number of runs reaching the target) across runs.

    For each threshold thr, we create:
      - t_le_{thr}_mean
      - t_le_{thr}_std
      - t_le_{thr}_count   (non-NaN entries, i.e. runs that reached thr)
    """
    group_cols = ['algorithm', 'fast_transmission_strategy', 'arena_file', 'noise']

    agg_dict: dict[str, list[str]] = {}
    for thr in loss_targets:
        col = f"t_le_{thr:g}"
        # mean/std over runs, and count of non-NaN hits
        agg_dict[col] = ['mean', 'std', 'count']

    agg = df_runs.groupby(group_cols, sort=False).agg(agg_dict)
    # Flatten MultiIndex columns: (col, stat) -> "col_stat"
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()
    return agg



def build_loss_table_dataframe(
    agg: pd.DataFrame,
    loss_targets: list[float],
) -> pd.DataFrame:
    """
    From aggregated statistics, build a dataframe with:
      - algorithm
      - fast_transmission_strategy
      - arena_file
      - arena_disp
      - noise
      - for each threshold: mean, std, count, is_min flag per (arena, noise).

    Any (algorithm, ft, arena, noise, target) combination with fewer than
    MIN_RUNS_PER_TARGET runs reaching the target is ignored:
      - its mean/std are set to NaN, so it appears as an em-dash in LaTeX,
      - it is not considered when selecting the minimal time for boldface.
    """
    df = agg.copy()
    df['arena_disp'] = df['arena_file'].map(_arena_disp)

    # Apply minimum-run filter and compute per-(arena, noise) minima
    for thr in loss_targets:
        col_mean = f"t_le_{thr:g}_mean"
        col_std = f"t_le_{thr:g}_std"
        col_count = f"t_le_{thr:g}_count"

        # If fewer than MIN_RUNS_PER_TARGET runs reached the target, ignore this cell
        if col_count in df.columns:
            mask_low = df[col_count] < MIN_RUNS_PER_TARGET
            df.loc[mask_low, col_mean] = np.nan
            df.loc[mask_low, col_std] = np.nan

        # Minimal mean time per (arena, noise), ignoring NaNs
        group = df.groupby(['arena_file', 'noise'])[col_mean]
        mins = group.transform(lambda s: np.nan if s.isna().all() else s.min())

        mask_min = (
            np.isfinite(df[col_mean]) &
            np.isfinite(mins) &
            np.isclose(df[col_mean], mins, rtol=1e-5, atol=1e-8)
        )
        df[f"t_le_{thr:g}_is_min"] = mask_min

    # Sorting
    arena_order_map = {a: i for i, a in enumerate(ARENA_ORDER_RAW)}
    df['arena_order'] = df['arena_file'].map(
        lambda a: arena_order_map.get(a, len(arena_order_map))
    )
    df = df.sort_values(
        ['algorithm', 'fast_transmission_strategy', 'arena_order', 'arena_file', 'noise'],
        ignore_index=True,
    )

    return df



def write_loss_latex_table(
    df_table: pd.DataFrame,
    loss_targets: list[float],
    out_path: str,
):
    """
    Write LaTeX table "loss_targets_summary.tex" with rows:
        (algorithm, fast_transmission_strategy, arena_disp, noise)
    and, for each loss target, two columns:
        - time to target (mean ± std),
        - number of runs reaching the target.

    For each (arena, noise) pair, the smallest mean time per target (among
    combinations with at least MIN_RUNS_PER_TARGET runs reaching the target)
    is set in bold.

    Display mapping:
      - algorithm: es1p1 -> 1+1-ES, hit -> HIT
      - fast_transmission_strategy: never -> No, prob -> Yes
      - noise: noiseless -> No, strong_noise -> Yes
      - column 'Noise' is renamed to 'Biases'
    """
    if df_table.empty:
        print("[warn] No data for loss targets table; not writing LaTeX file.")
        return

    with open(out_path, "w") as f:
        f.write("% Auto-generated by result_summary_tables.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        # Resize to full text width
        f.write("  \\resizebox{1.00\\textwidth}{!}{%\n")

        # 3 text cols (algo, fts, arena) + 1 biases + 2 cols per loss target
        f.write("  \\begin{tabular}{llll" + "cc" * len(loss_targets) + "}\n")
        f.write("    \\toprule\n")

        # Header row
        headers = [
            "Algorithm",
            "Fast transmission",
            "Arena",
            "Biases",
        ]
        for thr in loss_targets:
            headers.append(fr"$t_{{L \le {thr:.2f}}}$ (s)")
            headers.append(fr"$n_{{L \le {thr:.2f}}}$")
        f.write("    " + " & ".join(headers) + " \\\\\n")
        f.write("    \\midrule\n")

        # Data rows
        for _, row in df_table.iterrows():
            cells = [
                _disp_algorithm(row['algorithm']),
                _disp_fast_transmission(row['fast_transmission_strategy']),
                str(row['arena_disp']),
                _disp_noise(row['noise']),
            ]
            for thr in loss_targets:
                mu = row.get(f"t_le_{thr:g}_mean", np.nan)
                sigma = row.get(f"t_le_{thr:g}_std", np.nan)
                bold = bool(row.get(f"t_le_{thr:g}_is_min", False))
                n_runs = row.get(f"t_le_{thr:g}_count", np.nan)

                # Time (mean ± std); will be '—' if < MIN_RUNS_PER_TARGET or NaN
                cells.append(_fmt_mean_std(mu, sigma, is_time=True, bold=bold))
                # Number of runs reaching the target
                cells.append(_fmt_count(n_runs))

            f.write("    " + " & ".join(cells) + " \\\\\n")

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("  }% end resizebox\n")
        f.write(
            "  \\caption{Time (in simulation seconds) to reach loss targets "
            + ", ".join(f"{thr:.2f}" for thr in loss_targets)
            + " (mean $\\pm$ std across runs), and number of runs $n$ reaching each target. "
              f"Only combinations with $\\geq {MIN_RUNS_PER_TARGET}$ runs reaching a target "
              "are considered when selecting the minimal time (bold).}\n"
        )
        f.write("  \\label{tab:loss_targets_summary}\n")
        f.write("\\end{table}\n")

    print(f"[table] wrote {out_path}")



# ---------------------------------------------------------------------------
# Genotypic diversity summary table
# ---------------------------------------------------------------------------

def compute_geno_diversity_per_run_arena(
    df: pd.DataFrame,
    final_window_s: float,
) -> pd.DataFrame:
    """
    Compute final-window genotypic diversity per
        (algorithm, fast_transmission_strategy, noise, arena_file, run).

    Returns a dataframe with columns:
        ['algorithm', 'fast_transmission_strategy', 'noise',
         'arena_file', 'run', 'geno_div_final']
    """
    required = {
        'time', 'run', 'arena_file', 'noise',
        'algorithm', 'fast_transmission_strategy',
    } | set(GENO_COLS)
    _ensure_columns(df, required)

    df_clean = clean_float16_genotype(df)
    df_scaled = _add_scaled_genotype(df_clean)

    geno_scaled_cols = [c + "_scaled" for c in GENO_COLS if (c + "_scaled") in df_scaled.columns]
    if not geno_scaled_cols:
        raise RuntimeError("No scaled genotype columns found; cannot compute diversity.")

    # group per run
    group_cols = ['algorithm', 'fast_transmission_strategy', 'noise', 'arena_file', 'run']

    # Per-row max time for each run (algo, fts, noise, arena, run)
    run_tmax = df_scaled.groupby(group_cols)['time'].transform('max')
    df_final = df_scaled[df_scaled['time'] >= (run_tmax - float(final_window_s))].copy()
    if df_final.empty:
        print("[warn] No data in final window for genotype diversity.")
        return pd.DataFrame()

    # Diversity per snapshot (algo, fts, noise, arena, run, time)
    gd = (
        df_final
        .groupby(group_cols + ['time'], sort=False)
        .apply(lambda g: _compute_genotypic_diversity(g[geno_scaled_cols]), include_groups=False)
        .reset_index(name='geno_div')
    )

    if gd.empty:
        print("[warn] Diversity snapshots empty.")
        return pd.DataFrame()

    # Final-window diversity per (algo, fts, noise, arena, run)
    gd_run = (
        gd
        .groupby(group_cols, sort=False)['geno_div']
        .mean()
        .reset_index(name='geno_div_final')
    )

    return gd_run


def compute_geno_diversity_summary(
    df: pd.DataFrame,
    final_window_s: float,
) -> pd.DataFrame:
    """
    Compute final-window genotypic diversity per run and arena, then aggregate
    to (algorithm, fast_transmission_strategy, noise).

    Steps:
      - clean genotype columns (float16 issues),
      - add scaled genotype columns,
      - for each (algo, fts, noise, arena, run), find last time and keep last W seconds,
      - per (algo, fts, noise, arena, run, time) snapshot, compute diversity across robots,
      - per (algo, fts, noise, arena, run): average over time (final-window),
      - aggregate per (algo, fts, noise) using mean ± std across runs and arenas.
    """
    gd_run = compute_geno_diversity_per_run_arena(df, final_window_s)
    if gd_run.empty:
        return pd.DataFrame()

    # Aggregate per (algorithm, fts, noise) across runs and arenas
    agg = (
        gd_run
        .groupby(['algorithm', 'fast_transmission_strategy', 'noise'], sort=False)['geno_div_final']
        .agg(['mean', 'std'])
        .reset_index()
    )
    return agg


def plot_geno_diversity_boxplot(
    df: pd.DataFrame,
    final_window_s: float,
    out_path: str,
):
    """
    Create a box plot of final-window genotypic diversity, with one box per
    (algorithm, fast_transmission_strategy, noise) combination across arenas
    and runs.

    Uses the same label reformulations as the genotypic diversity table and
    a colorblind-friendly seaborn palette.

    y-axis is limited to [0.0, 0.4].
    """
    gd_run = compute_geno_diversity_per_run_arena(df, final_window_s)
    if gd_run.empty:
        print("[warn] No data for genotype diversity boxplot; not writing figure.")
        return

    # Prepare display labels
    df_plot = gd_run.copy()
    df_plot['Algorithm'] = df_plot['algorithm'].map(_disp_algorithm)
    df_plot['Fast transmission'] = df_plot['fast_transmission_strategy'].map(_disp_fast_transmission)
    df_plot['Biases'] = df_plot['noise'].map(_disp_noise)

    # Combined condition label: (Fast transmission, Biases)
    df_plot['Condition'] = (
        "FT " + df_plot['Fast transmission'] + ", Bias " + df_plot['Biases']
    )

    # Plot
    sns.set_theme(style="whitegrid")
    # Ensure we only take as many colors as we have conditions
    cond_order = sorted(df_plot['Condition'].unique())
    palette = sns.color_palette("cb.iris", n_colors=len(cond_order))

    plt.figure(figsize=(5, 4))
    ax = sns.boxplot(
        data=df_plot,
        x="Algorithm",
        y="geno_div_final",
        hue="Condition",
        palette=palette,
    )

    #ax.set_ylim(0.0, 0.4)
    ax.set_ylabel(r"\textbf{Final " + rf"{int(final_window_s)}s genotypic diversity" + r"}")
    ax.set_xlabel(r"\textbf{Algorithm}")
    ax.legend(title=r"\textbf{Fast transmission / Biases}", loc="best")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")


def build_geno_table_dataframe(
    agg: pd.DataFrame,
    final_window_s: float,
) -> pd.DataFrame:
    """
    Build a dataframe with:
      - algorithm
      - fast_transmission_strategy
      - noise
      - formatted genotypic diversity (final window)
    """
    rows = []
    for _, row in agg.iterrows():
        mu = row.get('mean', np.nan)
        sigma = row.get('std', np.nan)
        rec = {
            'algorithm': row['algorithm'],
            'fast_transmission_strategy': row['fast_transmission_strategy'],
            'noise': row['noise'],
            'geno_div_final': _fmt_mean_std(mu, sigma, is_time=False, bold=False),
        }
        rows.append(rec)

    df_table = pd.DataFrame(rows)
    df_table = df_table.sort_values(
        ['algorithm', 'fast_transmission_strategy', 'noise'],
        ignore_index=True
    )
    return df_table


def write_geno_latex_table(
    df_table: pd.DataFrame,
    final_window_s: float,
    out_path: str,
):
    """
    Write LaTeX table "geno_diversity_summary.tex" with rows:
        (algorithm, fast_transmission_strategy, noise)
    and one column: final-window genotypic diversity (mean ± std).

    Display mapping:
      - algorithm: es1p1 -> 1+1-ES, hit -> HIT
      - fast_transmission_strategy: never -> No, prob -> Yes
      - noise: noiseless -> No, strong_noise -> Yes
      - column 'Noise' is renamed to 'Biases'
    """
    if df_table.empty:
        print("[warn] No data for genotype diversity table; not writing LaTeX file.")
        return

    with open(out_path, "w") as f:
        f.write("% Auto-generated by result_summary_tables.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write("  \\begin{tabular}{lllc}\n")
        f.write("    \\toprule\n")
        headers = [
            "Algorithm",
            "Fast transmission",
            "Biases",
            fr"$D_{{\mathrm{{geno}},\mathrm{{final}}~{int(final_window_s)},\mathrm{{s}}}}$",
        ]
        f.write("    " + " & ".join(headers) + " \\\\\n")
        f.write("    \\midrule\n")

        for _, row in df_table.iterrows():
            cells = [
                _disp_algorithm(row['algorithm']),
                _disp_fast_transmission(row['fast_transmission_strategy']),
                _disp_noise(row['noise']),
                str(row['geno_div_final']),
            ]
            f.write("    " + " & ".join(cells) + " \\\\\n")

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write(
            f"  \\caption{{Final-window genotypic diversity (last {int(final_window_s)}\\,s), "
            f"averaged across runs and arenas (mean $\\pm$ std).}}\n"
        )
        f.write("  \\label{tab:geno_diversity_summary}\n")
        f.write("\\end{table}\n")

    print(f"[table] wrote {out_path}")


def compute_pheno_diversity_per_run_arena(
    df: pd.DataFrame,
    final_window_s: float,
) -> pd.DataFrame:
    """
    Compute final-window phenotypic diversity per
        (algorithm, fast_transmission_strategy, noise, arena_file, run).

    We reuse the same diversity definition as for genotype:
        D = 4 * mean(variance_per_dimension), clipped to [0,1],

    but on the phenotypic stats (pol_local, wall_local, pers_local, nb_local).

    Steps:
      - clean phenotypic columns (float16 issues, clipping to [0,1]),
      - for each (algo, fts, noise, arena, run), find last time and keep last W seconds,
      - per (algo, fts, noise, arena, run, time) snapshot, compute diversity across robots,
      - per (algo, fts, noise, arena, run): average over time (final-window).

    Returns a dataframe with columns:
        ['algorithm', 'fast_transmission_strategy', 'noise',
         'arena_file', 'run', 'pheno_div_final']
    """
    required = {
        'time', 'run', 'arena_file', 'noise',
        'algorithm', 'fast_transmission_strategy',
    } | set(PHENO_COLS)
    _ensure_columns(df, required)

    # Clean float16 phenotypic stats
    df_clean = clean_float16_pheno(df)

    pheno_cols_present = [c for c in PHENO_COLS if c in df_clean.columns]
    if not pheno_cols_present:
        print("[warn] No phenotypic columns found in dataframe; cannot compute diversity.")
        return pd.DataFrame()

    # group per run
    group_cols = ['algorithm', 'fast_transmission_strategy', 'noise', 'arena_file', 'run']

    # Per-row max time for each run (algo, fts, noise, arena, run)
    run_tmax = df_clean.groupby(group_cols)['time'].transform('max')
    df_final = df_clean[df_clean['time'] >= (run_tmax - float(final_window_s))].copy()
    if df_final.empty:
        print("[warn] No data in final window for phenotypic diversity.")
        return pd.DataFrame()

    # Diversity per snapshot (algo, fts, noise, arena, run, time)
    # across robots, using the same _compute_genotypic_diversity formula
    gd_ph = (
        df_final
        .groupby(group_cols + ['time'], sort=False)
        .apply(
            lambda g: _compute_genotypic_diversity(g[pheno_cols_present]),
            include_groups=False,
        )
        .reset_index(name='pheno_div')
    )

    if gd_ph.empty:
        print("[warn] Phenotypic diversity snapshots empty.")
        return pd.DataFrame()

    # Final-window diversity per (algo, fts, noise, arena, run)
    ph_run = (
        gd_ph
        .groupby(group_cols, sort=False)['pheno_div']
        .mean()
        .reset_index(name='pheno_div_final')
    )

    return ph_run


def plot_pheno_diversity_boxplot(
    df: pd.DataFrame,
    final_window_s: float,
    out_path: str,
):
    """
    Box plot of final-window phenotypic diversity, with one box per
    (algorithm, fast_transmission_strategy, noise) combination across arenas
    and runs.

    Diversity is in [0,1], defined like genotypic diversity but on
    (pol_local, wall_local, pers_local, nb_local).

    Uses the same label reformulations (No/Yes, 1+1-ES/HIT) and a
    colorblind-friendly seaborn palette.

    Draws vertical separators between different (Algorithm, Fast transmission)
    pairs.
    """
    ph_run = compute_pheno_diversity_per_run_arena(df, final_window_s)
    if ph_run.empty:
        print("[warn] No data for phenotypic diversity boxplot; not writing figure.")
        return

    df_plot = ph_run.copy()
    df_plot['Algorithm'] = df_plot['algorithm'].map(_disp_algorithm)
    df_plot['Fast transmission'] = df_plot['fast_transmission_strategy'].map(_disp_fast_transmission)
    df_plot['Biases'] = df_plot['noise'].map(_disp_noise)

    # One x-category per full combination (Algorithm, FT, Biases)
    df_plot['Combo'] = (
        df_plot['Algorithm']
        + " / FT " + df_plot['Fast transmission']
        + " / Bias " + df_plot['Biases']
    )

    # Unique combos, ordered nicely by (Algorithm, FT, Biases)
    combos = (
        df_plot[['Combo', 'Algorithm', 'Fast transmission']]
        .drop_duplicates()
        .sort_values(['Algorithm', 'Fast transmission', 'Combo'])
    )
    combo_order = combos['Combo'].tolist()

    # For vertical separators: (Algorithm, Fast) per x-position
    pair_for_pos = list(zip(combos['Algorithm'], combos['Fast transmission']))

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", n_colors=len(combo_order))

    plt.figure(figsize=(max(7, 0.9 * len(combo_order)), 8))
    ax = sns.violinplot(
        data=df_plot,
        x="Combo",
        y="pheno_div_final",
        order=combo_order,
        palette=palette,
        hue="Combo",
        dodge=False,
        legend=False,
    )

    ax.set_ylabel(f"Final {int(final_window_s)}s phenotypic diversity")
    ax.set_xlabel("Algorithm / Fast transmission / Biases")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(25)
        lbl.set_ha('right')

    # Optional: same ylim as genotypic diversity, if you want direct comparison
    # ax.set_ylim(0.0, 0.4)

    # --- vertical separators between (Algorithm, Fast transmission) groups ---
    ymin, ymax = ax.get_ylim()
    for i in range(1, len(pair_for_pos)):
        if pair_for_pos[i] != pair_for_pos[i - 1]:
            ax.axvline(
                x=i - 0.5,
                linestyle='--',
                linewidth=0.8,
                alpha=0.4,
            )
    ax.set_ylim(ymin, ymax)  # restore limits

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plot] wrote {out_path}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Assemble optimization result Feather files and generate LaTeX "
            "tables for loss targets and genotypic diversity."
        )
    )
    ap.add_argument(
        "-i", "--indir", required=True,
        help="Input directory containing result_*.feather files."
    )
    ap.add_argument(
        "-o", "--out", required=True,
        help="Output directory for .tex tables."
    )
    ap.add_argument(
        "--loss-targets", type=str, default="0.3,0.25,0.2,0.15,0.1",
        help="Comma-separated list of loss thresholds, "
             "default: '0.3,0.25,0.2,0.15,0.1'."
    )
    ap.add_argument(
        "--final-window-s", type=float, default=100.0,
        help="Final-window length (in seconds) for genotype diversity stats. Default: 100."
    )

    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    loss_targets = [
        float(x.strip())
        for x in args.loss_targets.split(',')
        if x.strip()
    ]
    if not loss_targets:
        raise ValueError("No valid --loss-targets provided.")

    # 1) Load & assemble all results
    df_all = load_all_results(args.indir)

    # 2) Loss targets summary
    df_runs_loss = compute_per_run_loss_stats(
        df_all,
        loss_targets=loss_targets,
    )
    agg_loss = aggregate_loss_per_combo(df_runs_loss, loss_targets)
    df_loss_table = build_loss_table_dataframe(
        agg_loss,
        loss_targets,
    )
    loss_tex_path = os.path.join(args.out, "loss_targets_summary.tex")
    write_loss_latex_table(
        df_loss_table,
        loss_targets,
        out_path=loss_tex_path,
    )

    # 3) Genotypic diversity summary
    try:
        agg_geno = compute_geno_diversity_summary(
            df_all,
            final_window_s=args.final_window_s,
        )
    except Exception as e:
        print(f"[warn] Could not compute genotype diversity summary: {e}")
        return 0

    if not agg_geno.empty:
        df_geno_table = build_geno_table_dataframe(
            agg_geno,
            final_window_s=args.final_window_s,
        )
        geno_tex_path = os.path.join(args.out, "geno_diversity_summary.tex")
        write_geno_latex_table(
            df_geno_table,
            final_window_s=args.final_window_s,
            out_path=geno_tex_path,
        )

        # Boxplot of genotypic diversity across arenas and runs
        geno_boxplot_path = os.path.join(args.out, "geno_diversity_boxplot.pdf")
        plot_geno_diversity_boxplot(
            df_all,
            final_window_s=args.final_window_s,
            out_path=geno_boxplot_path,
        )

        # Boxplot of phenotypic diversity across arenas and runs
        pheno_boxplot_path = os.path.join(args.out, "pheno_diversity_boxplot.pdf")
        plot_pheno_diversity_boxplot(
            df_all,
            final_window_s=args.final_window_s,
            out_path=pheno_boxplot_path,
        )


    return 0



if __name__ == "__main__":
    import sys
    sys.exit(main())

