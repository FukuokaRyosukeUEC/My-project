"""
visualize_results.py

Visualize saved optimization and simulation results.

Inputs in the same directory:
    - optimization_result.npz
    - selected_solution.npy
    - selected_simulation_log.npz

Outputs in the same directory:
    - pareto_front.png
    - layout_selected.png
    - hv_history.png
    - timeseries_summary.png
    - queue_lengths.png
    - congestion_heatmap.png

Run:
    python visualize_results.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from event_booth_problem import (
    H,
    W,
    BOOTH_IDS,
    ENTRY_CELLS,
    EXIT_CELLS,
    RESULT_NPZ,
    SELECTED_SOLUTION_NPY,
    SELECTED_SIM_LOG_NPZ,
    PARETO_PNG,
    LAYOUT_PNG,
    HV_HISTORY_PNG,
    TIME_SERIES_PNG,
    QUEUE_LENGTHS_PNG,
    CONGESTION_HEATMAP_PNG,
    SimSettings,
    build_layout,
    load_simulation_log,
    save_simulation_log,
    simulate_layout,
)


def _ensure_sim_log(base_dir: Path, x: np.ndarray):
    log_path = base_dir / SELECTED_SIM_LOG_NPZ
    if log_path.exists():
        return load_simulation_log(log_path)
    settings = SimSettings(record_positions=True, record_cell_counts=True)
    result = simulate_layout(x, settings=settings, return_log=True, force_record_positions=True, force_record_cell_counts=True)
    save_simulation_log(log_path, result, x, settings)
    return load_simulation_log(log_path)


def plot_pareto(base_dir: Path, result_data, selected_x: np.ndarray) -> None:
    final_F = np.asarray(result_data["final_F"], dtype=float)
    pareto_F = np.asarray(result_data["pareto_F"], dtype=float)
    selected_f = None
    if "selected_x" in result_data.files:
        selected_x_saved = np.asarray(result_data["selected_x"], dtype=int)
        if np.array_equal(selected_x_saved, selected_x) and len(pareto_F) > 0:
            selected_idx = np.argmin(np.linalg.norm(np.asarray(result_data["pareto_X"], dtype=int) - selected_x, axis=1))
            selected_f = pareto_F[selected_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(final_F[:, 0], -final_F[:, 1], s=28, alpha=0.45, label="Final population")
    if len(pareto_F) > 0:
        order = np.argsort(pareto_F[:, 0])
        ax.plot(pareto_F[order, 0], -pareto_F[order, 1], marker="o", linewidth=1.5, label="Pareto front")
    if selected_f is not None:
        ax.scatter([selected_f[0]], [-selected_f[1]], s=120, marker="*", label="Selected solution")
    ax.set_xlabel("Mean abandon count")
    ax.set_ylabel("Mean happiness")
    ax.set_title("Pareto front")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(base_dir / PARETO_PNG, dpi=200)
    plt.close(fig)


def draw_layout_background(ax, layout, show_text: bool = True) -> None:
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.grid(True, color="0.85", linewidth=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Base cells
    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="white", edgecolor="none", zorder=0))

    booth_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
        5: "tab:purple",
        6: "tab:brown",
    }
    for booth_id in BOOTH_IDS:
        for r, c in layout.body_by_id[booth_id]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=booth_colors[booth_id], alpha=0.70, edgecolor="black", linewidth=0.4, zorder=1))
            if show_text:
                ax.text(c, r, f"B{booth_id}", ha="center", va="center", fontsize=7, color="white", zorder=3)
        for r, c in layout.queue_by_id[booth_id]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=booth_colors[booth_id], alpha=0.23, edgecolor="black", linewidth=0.4, zorder=1))
            if show_text:
                ax.text(c, r, f"Q{booth_id}", ha="center", va="center", fontsize=6, zorder=3)
        er, ec = layout.booth_entry_by_id[booth_id]
        xr, xc = layout.booth_exit_by_id[booth_id]
        ax.scatter([ec], [er], marker="s", s=70, facecolors="none", edgecolors="black", linewidths=1.5, zorder=4)
        ax.scatter([xc], [xr], marker="^", s=70, facecolors="none", edgecolors="black", linewidths=1.5, zorder=4)

    for r, c in ENTRY_CELLS:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="limegreen", alpha=0.8, edgecolor="black", linewidth=0.5, zorder=2))
        if show_text:
            ax.text(c, r, "E", ha="center", va="center", fontsize=8, color="white", zorder=4)
    for r, c in EXIT_CELLS:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="black", alpha=0.75, edgecolor="black", linewidth=0.5, zorder=2))
        if show_text:
            ax.text(c, r, "X", ha="center", va="center", fontsize=8, color="white", zorder=4)


def plot_layout(base_dir: Path, selected_x: np.ndarray) -> None:
    layout = build_layout(selected_x)
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_layout_background(ax, layout, show_text=True)
    ax.set_title("Selected booth layout")
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Walkable cell"),
        Patch(facecolor="tab:blue", alpha=0.70, label="Booth body"),
        Patch(facecolor="tab:blue", alpha=0.23, label="Queue cell"),
        Patch(facecolor="limegreen", alpha=0.8, label="Venue entrance"),
        Patch(facecolor="black", alpha=0.75, label="Venue exit"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(base_dir / LAYOUT_PNG, dpi=200)
    plt.close(fig)


def plot_hv(base_dir: Path, result_data) -> None:
    hv = np.asarray(result_data["hv_history"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(hv) + 1), hv, marker="o", linewidth=1.5, label="Hypervolume")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Hypervolume history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(base_dir / HV_HISTORY_PNG, dpi=200)
    plt.close(fig)


def plot_time_series(base_dir: Path, sim_log) -> None:
    mean_happiness = np.asarray(sim_log["mean_happiness"], dtype=float)
    cumulative_abandon = np.asarray(sim_log["cumulative_abandon"], dtype=float)
    active_count = np.asarray(sim_log["active_count"], dtype=float)
    finished_count = np.asarray(sim_log["finished_count"], dtype=float)
    crowded_cells = np.asarray(sim_log["crowded_cells"], dtype=float)
    t = np.arange(len(mean_happiness))

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    series = [
        (mean_happiness, "Mean happiness"),
        (cumulative_abandon, "Cumulative abandon count"),
        (active_count, "Active agents"),
        (finished_count, "Finished agents"),
        (crowded_cells, "Crowded cells"),
    ]
    for ax, (y, label) in zip(axes, series):
        ax.plot(t, y, label=label)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Simulation time series", y=0.995)
    fig.tight_layout()
    fig.savefig(base_dir / TIME_SERIES_PNG, dpi=200)
    plt.close(fig)


def plot_queue_lengths(base_dir: Path, sim_log) -> None:
    q = np.asarray(sim_log["booth_queue_lengths"], dtype=float)
    t = np.arange(q.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, booth_id in enumerate(BOOTH_IDS):
        ax.plot(t, q[:, j], label=f"Booth {booth_id}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Queue length")
    ax.set_title("Queue length by booth")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(base_dir / QUEUE_LENGTHS_PNG, dpi=200)
    plt.close(fig)


def plot_congestion_heatmap(base_dir: Path, selected_x: np.ndarray, sim_log, mode: str = "occupancy") -> None:
    cell_counts = sim_log.get("cell_counts")
    if cell_counts is None or np.asarray(cell_counts).shape == ():
        print("No cell_counts log found; skip congestion heatmap.")
        return
    counts = np.asarray(cell_counts, dtype=float)
    if mode == "excess":
        # Excess over the happiness-decay threshold. Default threshold is 3.
        values = np.maximum(counts - 3 + 1, 0).sum(axis=0)
        title = "Cumulative excess congestion heatmap"
    else:
        values = counts.sum(axis=0)
        title = "Cumulative occupancy heatmap"

    layout = build_layout(selected_x)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(values, origin="upper")
    draw_layout_background(ax, layout, show_text=False)
    ax.imshow(values, origin="upper", alpha=0.75)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative count")
    fig.tight_layout()
    fig.savefig(base_dir / CONGESTION_HEATMAP_PNG, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap-mode", choices=["occupancy", "excess"], default="occupancy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    result_path = base_dir / RESULT_NPZ
    selected_path = base_dir / SELECTED_SOLUTION_NPY

    if not result_path.exists():
        raise FileNotFoundError(f"{result_path} not found. Run optimize_event_booths.py first.")
    if not selected_path.exists():
        raise FileNotFoundError(f"{selected_path} not found. Run optimize_event_booths.py first.")

    result_data = np.load(result_path, allow_pickle=True)
    selected_x = np.load(selected_path)
    sim_log = _ensure_sim_log(base_dir, selected_x)

    plot_pareto(base_dir, result_data, selected_x)
    plot_layout(base_dir, selected_x)
    plot_hv(base_dir, result_data)
    plot_time_series(base_dir, sim_log)
    plot_queue_lengths(base_dir, sim_log)
    plot_congestion_heatmap(base_dir, selected_x, sim_log, mode=args.heatmap_mode)

    print("Saved visualization files:")
    for name in [PARETO_PNG, LAYOUT_PNG, HV_HISTORY_PNG, TIME_SERIES_PNG, QUEUE_LENGTHS_PNG, CONGESTION_HEATMAP_PNG]:
        print(base_dir / name)


if __name__ == "__main__":
    main()
