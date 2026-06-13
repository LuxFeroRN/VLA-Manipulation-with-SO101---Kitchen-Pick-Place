"""
Generate static trajectory plots for all data*.rrd files.
One figure per run: 6 joint subplots (action vs obs) + per-episode MAE bar.
Style mirrors video_from_rrd.py.

Usage:
    python plot_runs.py               # all data*.rrd → plots/
    python plot_runs.py data3.rrd     # single file
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rerun.experimental as rr_exp

OUTPUT_DIR = Path("plots")

COLORS = {
    "action":   "#e94560",
    "obs":      "#4ecdc4",
    "error":    "#f5a623",
    "episode":  "#30363d",
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "text":     "#c9d1d9",
    "grid":     "#21262d",
    "subtitle": "#8b949e",
}

_GAP_MULTIPLIER = 5


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(path: str) -> tuple[dict, list[str]]:
    """Load full recording using rerun.experimental (legacy-RRD compatible)."""
    store = rr_exp.RrdReader(path).stream().collect()
    schema = store.schema()

    obs_keys = sorted(
        col.entity_path for col in schema.component_columns()
        if col.entity_path.startswith("/observation.")
        and col.component_type == "rerun.components.Scalar"
    )
    joints = []
    for ok in obs_keys:
        stem = ok[len("/observation."):]
        action_key = f"/action.{stem}"
        if any(c.entity_path == action_key for c in schema.component_columns()):
            joints.append(stem)

    if not joints:
        raise ValueError("No matching action/observation pairs in " + path)

    data = {}
    for chunk in store.stream():
        rb = chunk.to_record_batch()
        if "Scalars:scalars" not in rb.schema.names:
            continue
        d = rb.to_pydict()
        name = chunk.entity_path.lstrip("/")
        ticks = d["log_tick"]
        values = [v[0] for v in d["Scalars:scalars"]]
        timestamps = [t.timestamp() for t in d["log_time"]]
        data[f"vals.{name}"]  = pd.Series(values,     index=ticks)
        data[f"time.{name}"]  = pd.Series(timestamps, index=ticks)

    return data, joints


def episode_boundary_times(data: dict, joints: list[str], t0: float) -> list[float]:
    """Return list of times (relative to t0) where new episodes start."""
    key = f"time.observation.{joints[0]}"
    s = pd.Series(data[key].sort_index().values)
    dt = s.diff()
    median_step = dt.median()
    threshold = _GAP_MULTIPLIER * median_step
    gap_positions = s[dt > threshold].index.tolist()
    return [(s.iloc[p] - t0) for p in gap_positions]


def per_episode_mae(data: dict, joints: list[str]) -> tuple[list[float], list[float]]:
    """Return (episode_start_times_relative, mae_per_episode)."""
    key = f"time.observation.{joints[0]}"
    t_series = pd.Series(data[key].sort_index().values)
    dt = t_series.diff()
    median_step = dt.median()
    threshold = _GAP_MULTIPLIER * median_step

    gap_pos = [0] + t_series[dt > threshold].index.tolist()
    ends = gap_pos[1:] + [len(t_series)]
    t0 = t_series.iloc[0]

    ep_starts, ep_maes = [], []
    for i, (b, e) in enumerate(zip(gap_pos, ends)):
        slices = {
            j: (
                data[f"vals.action.{j}"].sort_index().reset_index(drop=True).iloc[b:e].values,
                data[f"vals.observation.{j}"].sort_index().reset_index(drop=True).iloc[b:e].values,
            )
            for j in joints
        }
        mae = np.mean([np.abs(a - o).mean() for a, o in slices.values()])
        ep_starts.append(float(t_series.iloc[b] - t0))
        ep_maes.append(float(mae))

    return ep_starts, ep_maes


# ── Styling helpers ───────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.5)


# ── Plot rendering ────────────────────────────────────────────────────────────

def render_run_plot(data: dict, joints: list[str], stem: str, out_path: Path) -> None:
    t0 = data[f"time.observation.{joints[0]}"].sort_index().values[0]
    t_end = max(
        data[f"time.observation.{j}"].sort_index().values[-1] for j in joints
    )
    duration = t_end - t0

    ep_boundaries = episode_boundary_times(data, joints, t0)
    ep_starts, ep_maes = per_episode_mae(data, joints)
    n_eps = len(ep_maes)

    n_joints = len(joints)
    n_cols = 3
    n_rows = (n_joints + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 3.5 * n_rows + 2.8), facecolor=COLORS["bg"])
    fig.suptitle(
        f"{stem}   —   {duration:.1f} s   |   {n_eps} episodios",
        color=COLORS["text"], fontsize=12, y=0.98,
    )

    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        figure=fig,
        height_ratios=[2.5] * n_rows + [1.5],
        hspace=0.55, wspace=0.35,
        top=0.93, bottom=0.07,
    )

    # ── Joint trajectory subplots ──
    for idx, j in enumerate(joints):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        _style_ax(ax)

        t_o = data[f"time.observation.{j}"].sort_index().values - t0
        t_a = data[f"time.action.{j}"].sort_index().values - t0
        o_vals = data[f"vals.observation.{j}"].sort_index().values
        a_vals = data[f"vals.action.{j}"].sort_index().values

        # Shaded error envelope
        # Interpolate action onto obs timestamps for fill_between
        a_interp = np.interp(t_o, t_a, a_vals)
        ax.fill_between(t_o, a_interp, o_vals,
                        alpha=0.15, color=COLORS["error"], linewidth=0)

        ax.plot(t_a, a_vals, color=COLORS["action"], lw=1.0, alpha=0.9, label="action")
        ax.plot(t_o, o_vals, color=COLORS["obs"],    lw=1.0, alpha=0.9, label="obs")

        # Episode boundaries
        for bt in ep_boundaries:
            ax.axvline(bt, color=COLORS["episode"], lw=0.8, alpha=0.7, linestyle="--")

        ax.set_title(j.replace("_", " "), fontsize=8)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.set_xlim(0, duration)

        if idx == 0:
            ax.legend(fontsize=6, loc="upper right",
                      facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
                      labelcolor=COLORS["text"])

    # ── Per-episode MAE bar chart ──
    ax_mae = fig.add_subplot(gs[n_rows, :])
    _style_ax(ax_mae)

    bar_w = (ep_starts[1] - ep_starts[0]) * 0.7 if len(ep_starts) > 1 else 1.0
    bar_colors = [
        "#e94560" if m > 4.0 else "#f5a623" if m > 2.0 else "#4ecdc4"
        for m in ep_maes
    ]
    ax_mae.bar(ep_starts, ep_maes, width=bar_w, color=bar_colors, alpha=0.85)

    # Threshold lines
    ax_mae.axhline(2.0, color=COLORS["obs"],    lw=0.8, linestyle=":", alpha=0.6)
    ax_mae.axhline(4.0, color=COLORS["action"], lw=0.8, linestyle=":", alpha=0.6)
    ax_mae.text(duration * 0.01, 2.05, "2°", color=COLORS["obs"],    fontsize=6)
    ax_mae.text(duration * 0.01, 4.05, "4°", color=COLORS["action"], fontsize=6)

    ax_mae.set_xlabel("t inicio episodio (s)", fontsize=7)
    ax_mae.set_ylabel("MAE medio (°)", fontsize=7)
    ax_mae.set_title("Error de seguimiento por episodio", fontsize=8)
    ax_mae.set_xlim(0, duration)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def process(rrd_path: str) -> None:
    stem = Path(rrd_path).stem
    print(f"\nLoading {rrd_path} ...")
    data, joints = load_run(rrd_path)
    out = OUTPUT_DIR / f"{stem}_trajectories.png"
    render_run_plot(data, joints, stem, out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            process(arg)
    else:
        rrds = sorted(Path(".").glob("data*.rrd"))
        if not rrds:
            sys.exit("No data*.rrd files found.")
        for rrd in rrds:
            process(str(rrd))
    print("\nDone.")
