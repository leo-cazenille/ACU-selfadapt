#!/usr/bin/env python3
"""
traces_gif_batch.py
-------------------
Create per-run robot traces (PNG frames + optional GIF) from a dataframe that may
contain many simulations spanning different parameter combinations, arenas, and runs.

Adds:
- Periodic square handling for "empty"/"torus" arenas with side L = sqrt(arena_surface).
  Trail segments are drawn with minimal-image wrapping (no long lines).
- Arena boundary overlay (black):
    * periodic: square box [0, L] × [0, L]
    * polygonal arenas: loaded+scaled via pogosim.arenas using --arenas-dir
- CLI: --arenas-dir (default "arenas/")
"""

from __future__ import annotations

import os
import re
import math
import shutil
import subprocess
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from pogosim import utils
import pogosim.arenas as arenas  # polygon loader/scaler/plotter

# ─────────────────────────────── Helpers ─────────────────────────────── #

def _safe(s: str, maxlen: int = 120) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("_")
    return s[:maxlen] or "x"

def _short_label_from_arena(arena: str, maxlen: int = 50) -> str:
    name = Path(str(arena)).name
    name = name.rsplit(".", 1)[0]
    return _safe(name, maxlen=maxlen)

def _simset_label(row_like: Dict[str, object]) -> str:
    parts = []
    for k in sorted(row_like.keys()):
        v = row_like[k]
        parts.append(f"{_safe(k)}={_safe(v)}")
    return "sim__" + "__".join(parts) if parts else "sim__default"

def _build_boundary_spec(
    arena_value: Optional[str],
    arena_surface: float,
    arenas_dir: str | os.PathLike,
) -> Dict[str, object]:
    """
    Return a simple, picklable spec describing how to draw the arena boundary and (if needed)
    how to treat coordinates. Two modes:
      - periodic square for 'empty' or 'torus'
      - polygon (outer + holes) for anything else (resolved under arenas_dir if relative)
    """
    if isinstance(arena_value, str) and arena_value.lower() in ("empty", "torus"):
        L = float(math.sqrt(float(arena_surface)))
        return {
            "type": "periodic_square",
            "L": L,
            "bounds": (0.0, 0.0, L, L),
        }

    if arena_value is None or str(arena_value).strip() == "":
        return {"type": "none"}  # draw nothing special

    # Polygon arena
    afile = Path(arena_value + ".csv")
    if not afile.is_absolute():
        afile = Path(arenas_dir) / afile
    poly = arenas.build_scaled_arena_polygon(str(afile), float(arena_surface))
    minx, miny, maxx, maxy = arenas.get_bounds_from_polygon(poly)
    # Extract simple lists for exterior + holes
    ext_x, ext_y = poly.exterior.xy
    holes = []
    for interior in poly.interiors:
        hx, hy = interior.xy
        holes.append((list(hx), list(hy)))

    return {
        "type": "polygon",
        "bounds": (float(minx), float(miny), float(maxx), float(maxy)),
        "exterior": (list(ext_x), list(ext_y)),
        "holes": holes,
    }

def _wrap_delta_min_image(d: np.ndarray, L: float) -> np.ndarray:
    """Apply minimal-image convention to a delta array for period L."""
    return d - L * np.round(d / L)

# ───────────────────── GIF building (gifski required) ────────────────── #
def _compile_gif(frame_paths: List[str], gif_path: Path, fps: int, gifski_bin: str = "gifski") -> bool:
    gifski_exe = shutil.which(gifski_bin)
    if gifski_exe is None:
        print(f"[WARNING] gifski not found ('{gifski_bin}'). GIF skipped for {gif_path}.")
        return False
    try:
        subprocess.run(
            [gifski_exe, "-q", "-r", str(fps), "--output", str(gif_path), *frame_paths],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] gifski failed ({e}). GIF not produced for {gif_path}.")
        return False

# ──────────────────────── Single-run trace renderer ──────────────────── #
def _render_single_run(
    run_df: pd.DataFrame,
    run_output_dir: Path,
    *,
    boundary_spec: Dict[str, object] | None = None,
    k_steps: int = 20,
    robot_cmap_name: str = "tab20",
    point_size: int = 30,
    line_width: float = 2.0,
    fade_min_alpha: float = 0.1,
    dpi: int = 150,
    make_gif: bool = False,
    gif_fps: int = 20,
    gif_name: str = "trace.gif",
    gifski_bin: str = "gifski",
    margin_frac: float = 0.03,
) -> List[str]:
    """
    Render one run's fading-trail PNG frames (and optional GIF).

    If boundary_spec["type"] == "periodic_square":
        - force axis limits to [0, L] × [0, L]
        - draw boundary square
        - draw trail segments with minimal-image wrapping (no long lines)
    If boundary_spec["type"] == "polygon":
        - draw exterior + holes in black
        - axis limits from polygon bounds (+ small margin)
    """
    if run_df.empty:
        return []

    run_df = run_df.sort_values(["time", "robot_id"], ignore_index=True)

    # Determine axis limits and prepare boundary drawing
    periodic = False
    L = None
    if boundary_spec and boundary_spec.get("type") == "periodic_square":
        periodic = True
        L = float(boundary_spec["L"])  # type: ignore[assignment]
        x_min, y_min, x_max, y_max = 0.0, 0.0, L, L
    elif boundary_spec and boundary_spec.get("type") == "polygon":
        bminx, bminy, bmaxx, bmaxy = boundary_spec["bounds"]  # type: ignore[assignment]
        dx, dy = (bmaxx - bminx), (bmaxy - bminy)
        x_min, x_max = bminx - dx * margin_frac, bmaxx + dx * margin_frac  # type: ignore[assignment]
        y_min, y_max = bminy - dy * margin_frac, bmaxy + dy * margin_frac  # type: ignore[assignment]
    else:
        # Fallback to data-driven bounds
        x_min, x_max = run_df["x"].min(), run_df["x"].max()
        y_min, y_max = run_df["y"].min(), run_df["y"].max()
        dx, dy = max(1e-9, x_max - x_min), max(1e-9, y_max - y_min)
        x_min -= dx * margin_frac; x_max += dx * margin_frac
        y_min -= dy * margin_frac; y_max += dy * margin_frac

    times     = run_df["time"].unique()
    robot_ids = np.sort(run_df["robot_id"].unique())
    cmap      = get_cmap(robot_cmap_name)
    color_for = {rid: cmap(i % cmap.N)[:3] for i, rid in enumerate(robot_ids)}

    tail_times : List[float] = []
    frame_paths: List[str]   = []

    run_output_dir.mkdir(parents=True, exist_ok=True)

    for current_time in times:
        tail_times.append(current_time)
        if len(tail_times) > k_steps:
            tail_times.pop(0)

        window_df = run_df[run_df["time"].isin(tail_times)]
        t_old, t_new = tail_times[0], tail_times[-1]
        age_den = (t_new - t_old) or 1.0

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Draw arena boundary
        if periodic and L is not None:
            ax.add_patch(Rectangle((0.0, 0.0), L, L, fill=False, edgecolor="black", linewidth=1.5))
        elif boundary_spec and boundary_spec.get("type") == "polygon":
            ext_x, ext_y = boundary_spec["exterior"]  # type: ignore[assignment]
            ax.plot(ext_x, ext_y, color="black", linewidth=1.5)
            for hole_x, hole_y in boundary_spec.get("holes", []):  # type: ignore[assignment]
                ax.plot(hole_x, hole_y, color="black", linewidth=1.0)

        # Draw trails
        for rid, group in window_df.groupby("robot_id", sort=False):
            g = group.sort_values("time")
            xs, ys, ts = g["x"].to_numpy(), g["y"].to_numpy(), g["time"].to_numpy()

            if len(xs) > 1:
                if periodic and L is not None:
                    # Minimal-image wrapping for each segment
                    x0 = xs[:-1]
                    y0 = ys[:-1]
                    dx = _wrap_delta_min_image(xs[1:] - xs[:-1], L)
                    dy = _wrap_delta_min_image(ys[1:] - ys[:-1], L)
                    x1 = x0 + dx
                    y1 = y0 + dy
                else:
                    x0 = xs[:-1]; y0 = ys[:-1]
                    x1 = xs[1:];  y1 = ys[1:]

                segs = np.stack(
                    [np.column_stack([x0, y0]),
                     np.column_stack([x1, y1])],
                    axis=1
                )
                seg_ages   = (ts[1:] - t_old) / age_den
                seg_alphas = fade_min_alpha + (1 - fade_min_alpha) * seg_ages
                seg_rgba   = [(*color_for[rid], a) for a in seg_alphas]

                ax.add_collection(LineCollection(
                    segs,
                    colors     = seg_rgba,
                    linewidths = line_width,
                    capstyle   = "round",
                    joinstyle  = "round",
                ))

            # Heads (current positions)
            ax.scatter(xs[-1], ys[-1], s=point_size, c=[color_for[rid]], edgecolors="none")

        ax.set_title(f"time = {current_time:.3f}   (tail = {len(tail_times)} steps)")
        fig.tight_layout()

        fname = run_output_dir / f"trace_{current_time:.6f}.png"
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
        frame_paths.append(str(fname.resolve()))

    if make_gif and frame_paths:
        _compile_gif(frame_paths, run_output_dir / gif_name, fps=gif_fps, gifski_bin=gifski_bin)

    return frame_paths

# ───────────────────────────── Per-run worker ────────────────────────── #
def _process_run(args: Tuple[int, pd.DataFrame, str, dict]) -> Tuple[int, List[str]]:
    run_val, run_df, out_dir_str, kw = args
    paths = _render_single_run(run_df, Path(out_dir_str), **kw)
    return run_val, paths

# ────────────────────────── Public trace API ─────────────────────────── #
def generate_trace_images(
    df: pd.DataFrame,
    *,
    boundary_spec: Dict[str, object] | None = None,
    k_steps: int = 20,
    output_dir: str | os.PathLike = "trace_frames",
    run_id: int | None = None,
    robot_cmap_name: str = "tab20",
    point_size: int = 30,
    line_width: float = 2.0,
    fade_min_alpha: float = 0.1,
    dpi: int = 150,
    run_fmt: str = "run_{run}",
    # GIF options
    make_gif: bool = False,
    gif_fps: int = 20,
    gif_name: str = "trace.gif",
    gifski_bin: str = "gifski",
    # Parallelism
    n_jobs: int | None = None,
) -> Union[List[str], Dict[int, List[str]]]:
    """
    Render fading-trail PNGs (and optional GIFs) from a robot-trace dataframe.
    If run_id is None and 'run' exists, each run is processed (optionally in parallel).
    """
    df = df.copy()

    required = ["time", "robot_id", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns for traces: {missing}")

    # Single explicit run
    if run_id is not None:
        if "run" in df.columns:
            df = df[df["run"] == run_id]
        return _render_single_run(
            df,
            Path(output_dir),
            boundary_spec=boundary_spec,
            k_steps=k_steps,
            robot_cmap_name=robot_cmap_name,
            point_size=point_size,
            line_width=line_width,
            fade_min_alpha=fade_min_alpha,
            dpi=dpi,
            make_gif=make_gif,
            gif_fps=gif_fps,
            gif_name=gif_name,
            gifski_bin=gifski_bin,
        )

    # Automatic per-run processing
    if "run" in df.columns:
        runs = list(sorted(df["run"].unique()))
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        common_kw = dict(
            boundary_spec=boundary_spec,
            k_steps=k_steps,
            robot_cmap_name=robot_cmap_name,
            point_size=point_size,
            line_width=line_width,
            fade_min_alpha=fade_min_alpha,
            dpi=dpi,
            make_gif=make_gif,
            gif_fps=gif_fps,
            gif_name=gif_name,
            gifski_bin=gifski_bin,
        )

        tasks: List[Tuple[int, pd.DataFrame, str, dict]] = [
            (
                r,
                df[df["run"] == r],
                str(base_dir / run_fmt.format(run=r)),
                common_kw,
            )
            for r in runs
        ]

        if n_jobs == 1:
            results = [_process_run(t) for t in tasks]
        else:
            workers = n_jobs or os.cpu_count() or 1
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                results = pool.map(_process_run, tasks)

        return {run_val: paths for run_val, paths in results}

    # No 'run' column → treat as single run
    return _render_single_run(
        df,
        Path(output_dir),
        boundary_spec=boundary_spec,
        k_steps=k_steps,
        robot_cmap_name=robot_cmap_name,
        point_size=point_size,
        line_width=line_width,
        fade_min_alpha=fade_min_alpha,
        dpi=dpi,
        make_gif=make_gif,
        gif_fps=gif_fps,
        gif_name=gif_name,
        gifski_bin=gifski_bin,
    )

# ──────────────────────────────── Batch main ─────────────────────────── #
def create_traces_for_all_simsets(
    input_file: str,
    output_dir: str,
    *,
    arenas_dir: str | os.PathLike = "arenas",
    k_steps: int = 20,
    make_gif: bool = True,
    gif_fps: int = 20,
    gifski_bin: str = "gifski",
    robot_cmap_name: str = "tab20",
    point_size: int = 30,
    line_width: float = 2.0,
    fade_min_alpha: float = 0.1,
    dpi: int = 150,
    n_jobs: int | None = None,
) -> None:
    """
    Load dataframe + metadata, determine simulation sets from configuration["result_new_columns"],
    then for each simulation set, for each arena, for each run → render traces (+ GIFs).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataframe and metadata
    df, meta = utils.load_dataframe(input_file)
    config = meta.get("configuration", {})  # required by spec

    # Basic cleaning
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    if "run" not in df.columns:
        df["run"] = 0
    if "robot_id" not in df.columns:
        raise ValueError("Dataframe must contain a 'robot_id' column.")
    if "time" not in df.columns:
        raise ValueError("Dataframe must contain a 'time' column.")

    # Determine which columns define simulation sets
    sim_cols: List[str] = []
    if isinstance(config, dict) and "result_new_columns" in config:
        wanted = config.get("result_new_columns") or []
        if isinstance(wanted, (list, tuple)):
            sim_cols = [c for c in wanted if c in df.columns]

    # If none, treat as a single set
    if not sim_cols:
        sim_groups = [(("default",), df)]
    else:
        sim_groups = list(df.groupby(sim_cols, sort=False))

    arena_surface = float(config.get("arena_surface", 0.0))  # used for scaling periodic/polygon arenas

    # Iterate sets
    for key_vals, df_set in sim_groups:
        if sim_cols:
            key_dict = {k: v for k, v in zip(sim_cols, (key_vals if isinstance(key_vals, tuple) else (key_vals,)))}
        else:
            key_dict = {}

        sim_label = _simset_label(key_dict)
        base_sim_dir = Path(output_dir) / sim_label
        base_sim_dir.mkdir(parents=True, exist_ok=True)

        # Per arena
        arenas_in_set = list(df_set["arena_file"].dropna().unique()) if "arena_file" in df_set.columns else [None]

        for arena_value in arenas_in_set:
            if arena_value is None:
                sub = df_set
                arena_dir = base_sim_dir / "arena_unspecified"
                boundary_spec = {"type": "none"}
            else:
                sub = df_set[df_set["arena_file"] == arena_value]
                arena_dir = base_sim_dir / _short_label_from_arena(str(arena_value))
                boundary_spec = _build_boundary_spec(str(arena_value), arena_surface, arenas_dir)

            generate_trace_images(
                sub,
                boundary_spec=boundary_spec,
                k_steps=k_steps,
                output_dir=str(arena_dir),
                run_id=None,
                robot_cmap_name=robot_cmap_name,
                point_size=point_size,
                line_width=line_width,
                fade_min_alpha=fade_min_alpha,
                dpi=dpi,
                run_fmt="run_{run}",
                make_gif=make_gif,
                gif_fps=gif_fps,
                gif_name="trace.gif",
                gifski_bin=gifski_bin,
                n_jobs=n_jobs,
            )

# ──────────────────────────────── CLI entry ──────────────────────────── #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Batch traces/GIF generator across simulation sets, arenas, and runs.")
    p.add_argument("-i", "--inputFile", type=str, required=True, help="Path to input feather/parquet/csv (utils.load_dataframe compatible).")
    p.add_argument("-o", "--outputDir", type=str, default=".", help="Directory for output images/GIFs.")
    p.add_argument("--arenas-dir", type=str, default="arenas", help="Base directory for arena CSV files (used if arena_file is relative).")
    p.add_argument("--kSteps", type=int, default=20, help="Tail length in steps.")
    p.add_argument("--gif", action="store_true", help="Enable GIF creation (requires gifski in PATH).")
    p.add_argument("--gif-fps", type=int, default=20, help="GIF framerate.")
    p.add_argument("--gifski-bin", type=str, default="gifski", help="Path/name of gifski binary.")
    p.add_argument("--cmap", type=str, default="tab20", help="Matplotlib colormap for robot IDs.")
    p.add_argument("--point-size", type=int, default=30, help="Scatter size for robot heads.")
    p.add_argument("--line-width", type=float, default=2.0, help="Line width for trails.")
    p.add_argument("--fade-min-alpha", type=float, default=0.1, help="Minimum alpha for the oldest trail segment.")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    p.add_argument("--jobs", type=int, default=None, help="Workers for parallel per-run rendering (None → CPU count).")
    args = p.parse_args()

    create_traces_for_all_simsets(
        input_file=args.inputFile,
        output_dir=args.outputDir,
        arenas_dir=args.arenas_dir,
        k_steps=args.kSteps,
        make_gif=args.gif,
        gif_fps=args.gif_fps,
        gifski_bin=args.gifski_bin,
        robot_cmap_name=args.cmap,
        point_size=args.point_size,
        line_width=args.line_width,
        fade_min_alpha=args.fade_min_alpha,
        dpi=args.dpi,
        n_jobs=args.jobs,
    )

