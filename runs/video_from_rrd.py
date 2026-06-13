"""
Reconstruct a playback video for a Rerun .rrd recording.

Each video shows:
  - Static camera frame (top)
  - Animated joint-position plots (action vs observation) with a moving playhead

Usage:
  python video_from_rrd.py data1.rrd
  python video_from_rrd.py data1.rrd --fps 30 --output-dir videos/
  python video_from_rrd.py --all --output-dir videos/   # all data*.rrd in runs/
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import rerun as rr
import rerun_bindings as rrb


COLORS = {
    "action": "#e94560",
    "obs":    "#4ecdc4",
    "vline":  "#f5f5f5",
    "bg":     "#0d1117",
    "panel":  "#161b22",
    "text":   "#c9d1d9",
    "grid":   "#21262d",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(path: str) -> tuple[dict, list[str]]:
    """Return (data, joints) for the full recording without splitting into episodes."""
    store = rrb.load_recording(path)
    schema = store.schema()

    action_entities = sorted(
        col.entity_path for col in schema.component_columns()
        if col.entity_path.startswith("/action.")
        and col.component_type == "rerun.components.Scalar"
    )
    obs_entities = sorted(
        col.entity_path for col in schema.component_columns()
        if col.entity_path.startswith("/observation.")
        and col.component_type == "rerun.components.Scalar"
    )

    joints = []
    for aep in action_entities:
        stem = aep[len("/action."):]
        if f"/observation.{stem}" in obs_entities:
            joints.append(stem)

    if not joints:
        raise ValueError("No matching action/observation joint pairs found in " + path)

    data = {}
    for entity_path in action_entities + obs_entities:
        view = store.view(index="log_time", contents=entity_path)
        rb = view.select().read_all()
        df = rb.to_pandas()
        scalar_col = [c for c in df.columns if c not in ("log_tick", "log_time")][0]
        name = entity_path.lstrip("/")
        data[f"vals.{name}"] = pd.Series(
            [v[0] for v in df[scalar_col]],
            index=df["log_tick"].values,
        )
        data[f"time.{name}"] = pd.Series(
            df["log_time"].values.astype("int64") / 1e9,
            index=df["log_tick"].values,
        )

    return data, joints


def extract_camera_frame(path: str) -> np.ndarray | None:
    store = rrb.load_recording(path)
    schema = store.schema()
    has_camera = any(
        col.entity_path == "/observation.camera1"
        and col.component_type == "rerun.components.ImageBuffer"
        for col in schema.component_columns()
    )
    if not has_camera:
        return None

    view = store.view(index="log_time", contents="/observation.camera1")
    rb = view.select_static().read_all()
    if len(rb) == 0:
        return None

    fmt = rb["/observation.camera1:Image:format"][0][0].as_py()
    w, h = fmt["width"], fmt["height"]
    raw = np.frombuffer(
        bytes(bytearray(rb["/observation.camera1:Image:buffer"][0][0].as_py())),
        dtype=np.uint8,
    )
    img = raw.reshape(h, w, 3)
    # color_model 2 = RGB (already correct for matplotlib)
    if fmt.get("color_model") == rr.ColorModel.BGR.value:
        img = img[:, :, ::-1]
    return img


# ── Video rendering ───────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.5)


def render_run(data: dict, joints: list[str], camera: np.ndarray | None,
               out_path: str, fps: int, stem: str) -> None:
    ref_key = f"time.observation.{joints[0]}"
    t_abs = data[ref_key].sort_index().values
    t0 = t_abs[0]
    duration = float(t_abs[-1] - t0)
    n_frames = max(2, int(duration * fps))

    n_joints = len(joints)
    n_cols = 3
    n_rows = (n_joints + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(16, 4 + 3 * n_rows), facecolor=COLORS["bg"])

    if camera is not None:
        gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                                     height_ratios=[2.5, n_rows * 2.5],
                                     hspace=0.35)
        ax_cam = fig.add_subplot(gs_outer[0])
        ax_cam.imshow(camera)
        ax_cam.set_title(f"{stem}  |  {duration:.1f} s",
                         color=COLORS["text"], fontsize=11)
        ax_cam.axis("off")
        gs_joints = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols, subplot_spec=gs_outer[1], hspace=0.55, wspace=0.35
        )
    else:
        gs_joints = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                      hspace=0.55, wspace=0.35)

    vlines = []
    for idx, j in enumerate(joints):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs_joints[row, col])
        _style_ax(ax)

        t_j = data[f"time.observation.{j}"].sort_index().values - t0
        a_vals = data[f"vals.action.{j}"].sort_index().values
        o_vals = data[f"vals.observation.{j}"].sort_index().values
        t_a = data[f"time.action.{j}"].sort_index().values - t0

        ax.plot(t_a, a_vals, color=COLORS["action"], lw=1.2, alpha=0.85, label="action")
        ax.plot(t_j, o_vals, color=COLORS["obs"],    lw=1.2, alpha=0.85, label="obs")
        ax.set_title(j.replace("_", " "), fontsize=8)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.legend(fontsize=6, loc="upper right",
                  facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
                  labelcolor=COLORS["text"])

        vl = ax.axvline(x=0.0, color=COLORS["vline"], lw=1.0, alpha=0.9)
        vlines.append(vl)

    def update(frame_idx):
        t = frame_idx / fps
        for vl in vlines:
            vl.set_xdata([t, t])
        return vlines

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, blit=True, interval=1000 / fps
    )
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "23"],
    )
    ani.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def process_rrd(rrd_path: str, fps: int, out_dir: Path) -> None:
    print(f"\nLoading {rrd_path} ...")
    data, joints = load_run(rrd_path)
    camera = extract_camera_frame(rrd_path)

    stem = Path(rrd_path).stem
    n_samples = len(data[f"vals.observation.{joints[0]}"])
    print(f"  samples  : {n_samples}")
    print(f"  joints   : {joints}")
    print(f"  camera   : {'yes ' + str(camera.shape) if camera is not None else 'none'}")

    out_path = out_dir / f"{stem}.mp4"
    print(f"  rendering ...")
    render_run(data, joints, camera, str(out_path), fps=fps, stem=stem)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct execution videos from RRD files")
    parser.add_argument("rrd_path", nargs="?", help="Path to .rrd file")
    parser.add_argument("--all", action="store_true",
                        help="Process all data*.rrd files in the runs/ directory")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS (default 30)")
    parser.add_argument("--output-dir", default=".", help="Directory for output MP4 files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        runs_dir = Path(__file__).parent
        rrds = sorted(runs_dir.glob("data*.rrd"))
        if not rrds:
            sys.exit("No data*.rrd files found in " + str(runs_dir))
        for rrd in rrds:
            process_rrd(str(rrd), args.fps, out_dir)
    elif args.rrd_path:
        process_rrd(args.rrd_path, args.fps, out_dir)
    else:
        parser.print_help()
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
