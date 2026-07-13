"""metrics.py — 目的関数(§12)と補助 KPI(§12.3)・幸福度(§13)。

f_move は自由歩行換算時間(§12.1【v2】)。ブロック待機時間・待ち行列時間・
入場遅延時間は f_move に含めない(§25)。
"""
from __future__ import annotations

import numpy as np

from .models import FINISHED, NOT_ARRIVED, Config


def free_walk_seconds(v, cfg: Config) -> float:
    """来場者 1 人の自由歩行換算時間(秒)。"""
    return v.moved_cells * cfg.cell_size_meters / v.speed_mps


def happiness(v, cfg: Config) -> float:
    """§13 の幸福度(補助 KPI)。"""
    u_sum = 0.0
    for b in v.visited_booths:
        spec = cfg.booths[b]
        interest = float(v.interests[cfg.category_index(spec.category)])
        u_sum += cfg.w_interest * interest + cfg.w_attract * spec.attractiveness
    walk_min = free_walk_seconds(v, cfg) / 60.0
    queue_min = v.queue_seconds / 60.0
    return (
        u_sum
        - cfg.hap_alpha * walk_min
        - cfg.hap_beta * queue_min
        - cfg.hap_gamma * v.give_up_count
    )


def compute_metrics(sim) -> tuple[float, float, dict]:
    """シミュレーション終了時に目的関数 2 つと補助 KPI 群を計算する。"""
    cfg: Config = sim.cfg
    spawned = [v for v in sim.visitors if v.state != NOT_ARRIVED]
    n_spawned = len(spawned)

    # --- 目的関数 1: 移動負担(自由歩行換算時間, §12.1)
    if n_spawned:
        walk_secs = np.array([free_walk_seconds(v, cfg) for v in spawned])
        f_move = float(walk_secs.mean())
    else:
        walk_secs = np.array([])
        f_move = 0.0

    # --- 目的関数 2: 混雑度(§12.2)。T は総ステップ数(早期終了後の残余は 0 寄与)
    f_congestion = sim.congestion_sum / (cfg.n_steps * sim.n_walkable)

    # --- 補助 KPI(§12.3)
    q = np.array([v.queue_seconds for v in spawned]) if n_spawned else np.array([0.0])
    blocked = np.array([v.blocked_seconds for v in spawned]) if n_spawned else np.array([0.0])
    delays = np.array([v.entry_delay_seconds for v in spawned]) if n_spawned else np.array([0.0])
    n_exp = np.array([len(v.visited_booths) for v in spawned]) if n_spawned else np.array([0])
    giveups = int(sum(v.give_up_count for v in spawned))
    total_exp = int(n_exp.sum())
    hap = np.array([happiness(v, cfg) for v in spawned]) if n_spawned else np.array([0.0])
    finished = sum(1 for v in spawned if v.state == FINISHED)

    kpis = {
        "n_spawned": n_spawned,
        "mean_queue_seconds": float(q.mean()),
        "max_queue_seconds": float(q.max()),
        "mean_free_walk_seconds": f_move,
        "total_moved_cells": int(sum(v.moved_cells for v in spawned)),
        "mean_blocked_seconds": float(blocked.mean()),
        "mean_entry_delay_seconds": float(delays.mean()),
        "max_cell_occupancy_ratio": float(sim.max_occ_ratio),
        "congestion_sum": float(sim.congestion_sum),
        "mean_booths_experienced": float(n_exp.mean()),
        "experience_success_rate": (
            total_exp / (total_exp + giveups) if (total_exp + giveups) else 1.0
        ),
        "give_up_total": giveups,
        "exit_completion_rate": finished / n_spawned if n_spawned else 0.0,
        "mean_happiness": float(hap.mean()),
        "happiness_p10": float(np.percentile(hap, 10)) if n_spawned else 0.0,
    }
    return f_move, f_congestion, kpis
