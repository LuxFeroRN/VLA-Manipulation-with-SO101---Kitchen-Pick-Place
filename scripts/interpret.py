"""
Robot performance metrics from a Rerun .rrd recording.

Metrics computed:
  Tracking   — MAE and RMSE between action (commanded) and observation (actual)
  Smoothness — std of consecutive position differences (action series)
  Efficiency — episode duration (s) and total angular distance per joint
  Stability  — direction-change count, mean velocity, max velocity
  Consistency — cross-episode std of trajectories (joints × time)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import rerun.experimental as rr_exp
import pandas as pd
import numpy as np


RRD_PATH = sys.argv[1] if len(sys.argv) > 1 else "data1.rrd"

# Gap multiplier: a step is an episode boundary when it is this many times
# larger than the median step. Robust to different recording frequencies.
_GAP_MULTIPLIER = 5


# ─── Loading ─────────────────────────────────────────────────────────────────

def _load_series(path: str) -> tuple[dict, dict, list[str]]:
    """Return (values, timestamps, joints) — joints auto-detected from data."""
    store = rr_exp.RrdReader(path).stream().collect()
    vals, times = {}, {}
    for chunk in store.stream():
        rb = chunk.to_record_batch()
        if "Scalars:scalars" not in rb.schema.names:
            continue
        d = rb.to_pydict()
        name = chunk.entity_path.lstrip("/")
        vals[name]  = pd.Series(
            [v[0] for v in d["Scalars:scalars"]], index=d["log_tick"]
        )
        times[name] = pd.Series(
            [t.timestamp() for t in d["log_time"]], index=d["log_tick"]
        )

    obs_keys = sorted(k for k in vals if k.startswith("observation."))
    joints = []
    for key in obs_keys:
        stem = key[len("observation."):]   # e.g. "elbow_flex.pos"
        action_key = f"action.{stem}"
        if action_key in vals:
            joints.append(stem)

    if not joints:
        raise ValueError(
            "No matching action/observation pairs found. "
            "Expected entity paths like /action.<joint> and /observation.<joint>."
        )
    return vals, times, joints


def _episode_boundaries(time_series: pd.Series) -> list[int]:
    """Return list of positional indices where new episodes start.

    A gap is detected when a time step exceeds _GAP_MULTIPLIER × median step,
    which adapts automatically to any recording frequency.
    """
    s = time_series.sort_index()
    dt = s.diff()
    median_step = dt.median()
    threshold = _GAP_MULTIPLIER * median_step
    gap_locs = [s.index.get_loc(i) for i in dt[dt > threshold].index]
    starts = [0] + gap_locs
    return starts


def _split_episodes(series: pd.Series, starts: list[int]) -> list[pd.Series]:
    s = series.sort_index().reset_index(drop=True)
    ends = starts[1:] + [len(s)]
    return [s.iloc[b:e].reset_index(drop=True) for b, e in zip(starts, ends)]


def load_episodes(path: str) -> tuple[list[dict], list[str]]:
    """
    Return (episodes, joints).

    Each episode dict contains:
      action.<joint>  : pd.Series of commanded positions (index = sample rank)
      obs.<joint>     : pd.Series of actual positions
      time.<joint>    : pd.Series of timestamps for the observation
    """
    vals, times, joints = _load_series(path)

    # Use the joint with the most samples as the boundary reference
    ref_key = max(
        (f"observation.{j}" for j in joints),
        key=lambda k: len(times[k]),
    )
    starts = _episode_boundaries(times[ref_key])

    episodes = [{} for _ in starts]
    for j in joints:
        a_key = f"action.{j}"
        o_key = f"observation.{j}"

        a_episodes = _split_episodes(vals[a_key],  starts)
        o_episodes = _split_episodes(vals[o_key],  starts)
        t_episodes = _split_episodes(times[o_key], starts)

        for i, (a, o, t) in enumerate(zip(a_episodes, o_episodes, t_episodes)):
            episodes[i][f"action.{j}"]  = a
            episodes[i][f"obs.{j}"]     = o
            episodes[i][f"time.{j}"]    = t

    return episodes, joints


# ─── Per-episode metrics ──────────────────────────────────────────────────────

def tracking_metrics(ep: dict, joints: list[str]) -> pd.DataFrame:
    rows = []
    for j in joints:
        err = (ep[f"action.{j}"] - ep[f"obs.{j}"]).dropna()
        rows.append({
            "joint": j,
            "mae":  np.abs(err).mean(),
            "rmse": np.sqrt((err**2).mean()),
            "max_err": np.abs(err).max(),
        })
    return pd.DataFrame(rows).set_index("joint")


def smoothness_metrics(ep: dict, joints: list[str]) -> pd.DataFrame:
    rows = []
    for j in joints:
        diff = ep[f"action.{j}"].diff().dropna()
        rows.append({"joint": j, "smoothness_std": diff.std()})
    return pd.DataFrame(rows).set_index("joint")


def efficiency_metrics(ep: dict, joints: list[str]) -> pd.DataFrame:
    t0 = min(ep[f"time.{j}"].min() for j in joints)
    t1 = max(ep[f"time.{j}"].max() for j in joints)
    duration = t1 - t0

    rows = []
    for j in joints:
        dist = ep[f"obs.{j}"].diff().abs().sum()
        rows.append({"joint": j, "distance_deg": dist})
    df = pd.DataFrame(rows).set_index("joint")
    df["duration_s"] = duration
    return df


def stability_metrics(ep: dict, joints: list[str]) -> pd.DataFrame:
    rows = []
    for j in joints:
        pos  = ep[f"obs.{j}"]
        t    = ep[f"time.{j}"]
        diff = pos.diff().dropna()
        sign_changes = ((diff.shift(1) * diff) < 0).sum()

        vel = pos.diff() / t.diff()
        rows.append({
            "joint": j,
            "oscillations":  int(sign_changes),
            "vel_mean_deg_s": vel.abs().mean(),
            "vel_max_deg_s":  vel.abs().max(),
        })
    return pd.DataFrame(rows).set_index("joint")


# ─── Cross-episode consistency ────────────────────────────────────────────────

def consistency_metrics(episodes: list[dict], joints: list[str]) -> pd.DataFrame | None:
    """Mean cross-episode std of the observation trajectory per joint.

    Returns None when there is only one episode (std is undefined).
    Only episodes whose length matches the majority are used.
    """
    if len(episodes) < 2:
        return None

    ref_j = joints[0]
    lengths = [len(ep[f"obs.{ref_j}"]) for ep in episodes]
    common_len = pd.Series(lengths).mode()[0]
    full_eps = [ep for ep in episodes if len(ep[f"obs.{ref_j}"]) == common_len]

    if len(full_eps) < 2:
        return None

    rows = []
    for j in joints:
        mat = np.stack([ep[f"obs.{j}"].values for ep in full_eps])  # (n_ep, T)
        per_step_std = mat.std(axis=0)
        rows.append({"joint": j, "consistency_std": per_step_std.mean()})
    return pd.DataFrame(rows).set_index("joint")


# ─── Aggregated report ────────────────────────────────────────────────────────

def aggregate(episodes: list[dict], joints: list[str], metric_fn) -> pd.DataFrame:
    dfs = [metric_fn(ep, joints) for ep in episodes]
    combined = pd.concat(dfs, keys=range(len(dfs)), names=["episode"])
    return combined.groupby("joint").mean().round(4)


def print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    print(df.to_string())


def main():
    print(f"Loading {RRD_PATH} ...")
    episodes, joints = load_episodes(RRD_PATH)
    print(f"  joints    : {joints}")
    print(f"  episodes  : {len(episodes)}  "
          f"(lengths: {[len(ep[f'obs.{joints[0]}']) for ep in episodes]})")

    print_section(
        "TRACKING  —  action vs observation error  (degrees)",
        aggregate(episodes, joints, tracking_metrics),
    )
    print_section(
        "SMOOTHNESS  —  std of consecutive action steps  (degrees)",
        aggregate(episodes, joints, smoothness_metrics),
    )
    print_section(
        "EFFICIENCY  —  duration (s) and total angular distance (degrees)",
        aggregate(episodes, joints, efficiency_metrics),
    )
    print_section(
        "STABILITY  —  direction changes, mean/max velocity  (deg/s)",
        aggregate(episodes, joints, stability_metrics),
    )

    cons = consistency_metrics(episodes, joints)
    if cons is not None:
        print_section(
            "CONSISTENCY  —  cross-episode std of observation trajectory",
            cons.round(4),
        )
    else:
        print("\n  CONSISTENCY — skipped (need ≥ 2 episodes of equal length)")


if __name__ == "__main__":
    main()
