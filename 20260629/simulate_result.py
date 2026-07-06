"""
simulate_result_resimulate.py

Always run a fresh simulation for the selected booth layout and save MP4 animations.

This version intentionally does NOT just replay selected_simulation_log.npz by default.
It loads the optimized booth placement from selected_solution.npy, builds SimSettings
from command-line parameters, runs simulate_layout(...), saves a new log, and then
uses that newly generated log to create videos.

Inputs, by default in the same directory as this script:
    - selected_solution.npy

Outputs, by default in the same directory as this script:
    - selected_simulation_log.npz
    - simulation_run_metadata.json
    - agent_animation.mp4
    - congestion_animation.mp4

Example:
    python simulate_result_resimulate.py --T 200 --spawn-mean 10 --max-spawn-per-step 20 --agent-type-probs 1/3,1/3,1/3

Save only the congestion animation:
    python simulate_result_resimulate.py --video congestion

Use an existing log only when explicitly requested:
    python simulate_result_resimulate.py --use-existing-log

Important for MP4 on WSL:
    sudo apt update
    sudo apt install ffmpeg
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import ast
import json
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers
from matplotlib.patches import Rectangle

from event_booth_problem import (
    H,
    W,
    BOOTH_IDS,
    ENTRY_CELLS,
    EXIT_CELLS,
    SELECTED_SOLUTION_NPY,
    SELECTED_SIM_LOG_NPZ,
    AGENT_ANIMATION_MP4,
    CONGESTION_ANIMATION_MP4,
    SimSettings,
    build_layout,
    load_simulation_log,
    save_simulation_log,
    simulate_layout,
)

try:
    from visualize_results import draw_layout_background
except Exception:
    draw_layout_background = None


METADATA_JSON = "simulation_run_metadata.json"

# Specification defaults from UPoC_仕様書_ブラッシュアップ版_20260630.txt.
# These intentionally override the current SimSettings() defaults if they differ.
SPEC_DEFAULTS = {
    "T": 400,
    "sim_seed": 0,
    "spawn_mean": 20.0,
    "spawn_std": 2.0,
    "max_spawn_per_step": 20,
    "agent_type_probs": (1, 0, 0),
    "queue_capacity_per_cell": 5,
    "happiness_crowd_threshold": 3,
    "happiness_decay": 0.95,
    "block_threshold": 1.0e6,
    "max_move_delay": 5,
    "experience_time": 5,
    "penalty_value": 1.0e6,
    "abandon_to_tail": True,
}


def _parse_probability_value(value: str) -> float:
    """Parse values such as '0.3333' or '1/3' without using eval()."""
    value = value.strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_agent_type_probs(text: str) -> Tuple[float, float, float]:
    """Parse a comma-separated probability tuple and normalize it."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--agent-type-probs must have exactly 3 comma-separated values, e.g. 1/3,1/3,1/3")
    probs = np.asarray([_parse_probability_value(p) for p in parts], dtype=float)
    if np.any(probs < 0):
        raise argparse.ArgumentTypeError("agent type probabilities must be non-negative")
    total = float(np.sum(probs))
    if total <= 0:
        raise argparse.ArgumentTypeError("at least one agent type probability must be positive")
    probs = probs / total
    return tuple(float(p) for p in probs)  # type: ignore[return-value]


def make_sim_settings(args: argparse.Namespace) -> SimSettings:
    """Build simulation settings from command-line arguments."""
    return SimSettings(
        T=args.T,
        spawn_mean=args.spawn_mean,
        spawn_std=args.spawn_std,
        max_spawn_per_step=args.max_spawn_per_step,
        agent_type_probs=args.agent_type_probs,
        queue_capacity_per_cell=args.queue_capacity_per_cell,
        happiness_crowd_threshold=args.happiness_crowd_threshold,
        happiness_decay=args.happiness_decay,
        block_threshold=args.block_threshold,
        max_move_delay=args.max_move_delay,
        experience_time=args.experience_time,
        sim_seed=args.sim_seed,
        abandon_to_tail=args.abandon_to_tail,
        penalty_value=args.penalty_value,
        record_positions=True,
        record_cell_counts=True,
    )


def run_and_save_simulation(
    out_dir: Path,
    selected_x: np.ndarray,
    settings: SimSettings,
    step_seconds: float,
    log_name: str = SELECTED_SIM_LOG_NPZ,
) -> dict:
    """Run a fresh simulation for selected_x and save the resulting log."""
    result = simulate_layout(
        selected_x,
        settings=settings,
        return_log=True,
        force_record_positions=True,
        force_record_cell_counts=True,
    )

    log_path = out_dir / log_name
    save_simulation_log(log_path, result, selected_x, settings)

    metadata = {
        "source": "fresh_simulation",
        "selected_x": np.asarray(selected_x, dtype=int).tolist(),
        "step_seconds": float(step_seconds),
        "settings": asdict(settings),
        "F": np.asarray(result.get("F"), dtype=float).tolist(),
        "CV": float(result.get("CV", np.nan)),
        "summary": result.get("summary", {}),
    }
    with open(out_dir / METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return load_simulation_log(log_path)


def ensure_mp4_writer(fps: int) -> FFMpegWriter:
    if not writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required to save MP4 videos. On WSL, install it with: sudo apt update && sudo apt install ffmpeg"
        )
    return FFMpegWriter(fps=fps, bitrate=1800)


def _empty_offsets() -> np.ndarray:
    return np.empty((0, 2), dtype=float)


def make_agent_animation(base_dir: Path, selected_x: np.ndarray, sim_log: dict, fps: int, step_seconds: float) -> None:
    agent_positions = sim_log.get("agent_positions")
    if agent_positions is None or np.asarray(agent_positions).shape == ():
        raise ValueError("No agent position log found. Run a fresh simulation with record_positions=True.")

    layout = build_layout(selected_x)
    fig, ax = plt.subplots(figsize=(10, 8))
    if draw_layout_background is not None:
        draw_layout_background(ax, layout, show_text=False)
    else:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.grid(True, color="0.85")

    scatters = {
        1: ax.scatter([], [], s=20, label="Agent type 1", alpha=0.85),
        2: ax.scatter([], [], s=20, label="Agent type 2", alpha=0.85),
        3: ax.scatter([], [], s=20, label="Agent type 3", alpha=0.85),
    }
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    title = ax.set_title("Agent movement")

    frames = len(agent_positions)

    def update(frame: int):
        rows = list(agent_positions[frame])
        for agent_type, sc in scatters.items():
            coords = [(c, r) for (_aid, atype, r, c, _state) in rows if int(atype) == agent_type]
            sc.set_offsets(np.asarray(coords, dtype=float) if coords else _empty_offsets())
        sim_time = frame * step_seconds
        title.set_text(f"Agent movement | step {frame + 1}/{frames} | t={sim_time:g}s")
        return list(scatters.values()) + [title]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / max(fps, 1), blit=False)
    writer = ensure_mp4_writer(fps)
    out_path = base_dir / AGENT_ANIMATION_MP4
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(out_path)


def overlay_layout_edges(ax, layout) -> None:
    booth_edge_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
        5: "tab:purple",
        6: "tab:brown",
    }
    for booth_id in BOOTH_IDS:
        color = booth_edge_colors[booth_id]
        for r, c in layout.body_by_id[booth_id]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor=color, linewidth=1.0))
        for r, c in layout.queue_by_id[booth_id]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor=color, linewidth=0.6, linestyle="--"))
    for r, c in ENTRY_CELLS:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor="limegreen", linewidth=1.5))
    for r, c in EXIT_CELLS:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none", edgecolor="black", linewidth=1.5))


def make_congestion_animation(
    base_dir: Path,
    selected_x: np.ndarray,
    sim_log: dict,
    fps: int,
    mode: str,
    step_seconds: float,
) -> None:
    cell_counts = sim_log.get("cell_counts")
    if cell_counts is None or np.asarray(cell_counts).shape == ():
        raise ValueError("No cell count log found. Run a fresh simulation with record_cell_counts=True.")

    counts = np.asarray(cell_counts, dtype=float)
    if mode == "excess":
        values = np.maximum(counts - 3 + 1, 0)
        cbar_label = "Excess congestion"
        title_prefix = "Congestion intensity"
    else:
        values = counts
        cbar_label = "Occupancy count"
        title_prefix = "Occupancy intensity"

    layout = build_layout(selected_x)
    vmax = max(1.0, float(np.max(values)))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(values[0], origin="upper", vmin=0, vmax=vmax)
    overlay_layout_edges(ax, layout)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.grid(True, color="white", linewidth=0.4, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    title = ax.set_title(title_prefix)

    frames = values.shape[0]

    def update(frame: int):
        im.set_data(values[frame])
        sim_time = frame * step_seconds
        title.set_text(f"{title_prefix} | step {frame + 1}/{frames} | t={sim_time:g}s")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / max(fps, 1), blit=False)
    writer = ensure_mp4_writer(fps)
    out_path = base_dir / CONGESTION_ANIMATION_MP4
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(out_path)


def parse_args() -> argparse.Namespace:
    defaults = SPEC_DEFAULTS
    parser = argparse.ArgumentParser(description="Run or replay the selected booth-layout simulation and save MP4 animations.")

    parser.add_argument("--video", choices=["agents", "congestion", "both"], default="both")
    parser.add_argument("--congestion-mode", choices=["occupancy", "excess"], default="occupancy")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--step-seconds", type=float, default=1.0, help="Real-world duration represented by one simulation step; used for video labels only.")

    parser.add_argument("--selected-solution", type=Path, default=None, help="Path to selected_solution.npy. Default: same directory as this script.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for logs and videos. Default: same directory as this script.")
    parser.add_argument("--log-name", type=str, default=SELECTED_SIM_LOG_NPZ)
    parser.add_argument("--use-existing-log", action="store_true", help="Do not run a fresh simulation; load the existing log instead.")

    parser.add_argument("--T", type=int, default=defaults["T"], help="Number of simulation time steps.")
    parser.add_argument("--sim-seed", type=int, default=defaults["sim_seed"])
    parser.add_argument("--spawn-mean", type=float, default=defaults["spawn_mean"])
    parser.add_argument("--spawn-std", type=float, default=defaults["spawn_std"])
    parser.add_argument("--max-spawn-per-step", type=int, default=defaults["max_spawn_per_step"])
    parser.add_argument("--agent-type-probs", type=parse_agent_type_probs, default=defaults["agent_type_probs"], help="Comma-separated probabilities, e.g. 1/3,1/3,1/3 or 1,0,0.")
    parser.add_argument("--queue-capacity-per-cell", type=int, default=defaults["queue_capacity_per_cell"])
    parser.add_argument("--happiness-crowd-threshold", type=int, default=defaults["happiness_crowd_threshold"])
    parser.add_argument("--happiness-decay", type=float, default=defaults["happiness_decay"])
    parser.add_argument("--block-threshold", type=float, default=defaults["block_threshold"])
    parser.add_argument("--max-move-delay", type=int, default=defaults["max_move_delay"])
    parser.add_argument("--experience-time", type=int, default=defaults["experience_time"])
    parser.add_argument("--penalty-value", type=float, default=defaults["penalty_value"])
    parser.add_argument("--abandon-to-tail", dest="abandon_to_tail", action="store_true", default=defaults["abandon_to_tail"])
    parser.add_argument("--no-abandon-to-tail", dest="abandon_to_tail", action="store_false")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir = args.output_dir.resolve() if args.output_dir is not None else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_path = args.selected_solution.resolve() if args.selected_solution is not None else script_dir / SELECTED_SOLUTION_NPY
    if not selected_path.exists():
        raise FileNotFoundError(f"{selected_path} not found. Run optimize_event_booths.py first, or pass --selected-solution.")

    selected_x = np.load(selected_path)
    settings = make_sim_settings(args)
    log_path = out_dir / args.log_name

    if args.use_existing_log:
        if not log_path.exists():
            raise FileNotFoundError(f"{log_path} not found. Remove --use-existing-log or run a fresh simulation first.")
        print(f"Loading existing simulation log: {log_path}")
        sim_log = load_simulation_log(log_path)
    else:
        print("Running fresh simulation for selected booth layout...")
        print("selected_x:", np.asarray(selected_x, dtype=int))
        print("settings:", asdict(settings))
        sim_log = run_and_save_simulation(out_dir, selected_x, settings, args.step_seconds, log_name=args.log_name)

    print("Saving MP4 animation(s):")
    if args.video in ("agents", "both"):
        make_agent_animation(out_dir, selected_x, sim_log, fps=args.fps, step_seconds=args.step_seconds)
    if args.video in ("congestion", "both"):
        make_congestion_animation(out_dir, selected_x, sim_log, fps=args.fps, mode=args.congestion_mode, step_seconds=args.step_seconds)


if __name__ == "__main__":
    main()
