"""周辺混雑マップと混雑回避A*の単体テスト。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.models import WALKING
from src.routing import astar, bfs_distance_field
from src.simulator import Simulation


def _route(crowd_risk: np.ndarray):
    walkable = np.ones((5, 7), dtype=bool)
    occupancy = np.zeros_like(walkable, dtype=np.int16)
    capacity = np.full_like(occupancy, 3)
    goal = (6, 2)
    field = bfs_distance_field(walkable, [goal])
    return astar(
        walkable,
        occupancy,
        capacity,
        (0, 2),
        {goal},
        1.0,
        field,
        crowd_risk=crowd_risk,
        safe_occupancy_ratio=0.67,
        severe_congestion_weight=4.0,
    )


def test_astar_detours_around_severely_crowded_cell():
    """強混雑セルを通る最短路より、空いた迂回路を選ぶ。"""
    risk = np.zeros((5, 7), dtype=np.float64)
    risk[2, 3] = 1.0

    path = _route(risk)

    assert path is not None
    assert (3, 2) not in path
    assert len(path) == 8


def test_astar_keeps_shortest_path_for_light_crowding():
    """軽い混雑だけなら、過剰に遠回りせず最短路を維持する。"""
    risk = np.zeros((5, 7), dtype=np.float64)
    risk[2, 3] = 0.2

    path = _route(risk)

    assert path is not None
    assert (3, 2) in path
    assert len(path) == 6


def test_crowd_risk_includes_demand_and_inflates_to_neighbors():
    """次セルへの流入予測と、半径2セルへの減衰拡張が反映される。"""
    sim = Simulation.__new__(Simulation)
    sim.walkable = np.ones((5, 7), dtype=bool)
    sim.capacity = np.full((5, 7), 3, dtype=np.int16)
    sim.occupancy = np.zeros((5, 7), dtype=np.int16)
    sim.occupancy[2, 3] = 3
    sim.cfg = SimpleNamespace(
        crowd_prediction_weight=0.5,
        crowd_inflation_radius=2,
    )
    mover = SimpleNamespace(
        position=(0, 2),
        path=[(1, 2)],
        path_index=0,
        state=WALKING,
    )

    risk = sim._refresh_crowd_risk_map([mover])

    assert risk[2, 1] >= 0.5 / 3.0  # 次セルへの1人分の流入予測
    assert risk[2, 2] == 0.75       # 強混雑セルの1セル隣
    assert risk[2, 1] >= 0.50       # 強混雑セルの2セル隣


def test_forward_crowd_detection_is_limited_to_five_cells():
    """前方監視は現在設定の5セルまでとし、6セル先では発火しない。"""
    sim = Simulation.__new__(Simulation)
    sim.cfg = SimpleNamespace(
        crowd_lookahead_cells=5,
        safe_occupancy_ratio=0.67,
    )
    sim.crowd_risk = np.zeros((3, 8), dtype=np.float64)
    visitor = SimpleNamespace(
        path=[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        path_index=0,
    )

    sim.crowd_risk[1, 6] = 1.0
    assert not sim._path_has_crowding_ahead(visitor)

    sim.crowd_risk[1, 5] = 0.67
    assert sim._path_has_crowding_ahead(visitor)


def test_crowd_replan_requires_ten_percent_improvement():
    """混雑再探索では小差を棄却し、十分に安い迂回路だけを採用する。"""
    sim = Simulation.__new__(Simulation)
    sim.cfg = SimpleNamespace(
        booths=[SimpleNamespace(is_table=False)],
        safe_occupancy_ratio=0.67,
        severe_congestion_weight=4.0,
        crowd_replan_improvement_ratio=0.10,
    )
    sim.layout = SimpleNamespace(receptions=[(6, 0)])
    sim.booth_fields = [np.zeros((3, 7), dtype=np.int32)]
    sim.crowd_risk = np.zeros((3, 7), dtype=np.float64)
    current = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    visitor = SimpleNamespace(
        state=WALKING,
        target_booth_id=0,
        position=(0, 0),
        path=current.copy(),
        path_index=0,
        congestion_avoidance=1.0,
        steps_since_plan=4,
        blocked_count=0,
    )

    sim._find_path = lambda *_args: current.copy()
    assert not sim._replan(visitor, require_improvement=True)
    assert visitor.path == current and visitor.steps_since_plan == 4

    sim.crowd_risk[0, 3] = 1.0
    detour = [
        (1, 0), (2, 0), (2, 1), (3, 1),
        (4, 1), (4, 0), (5, 0), (6, 0),
    ]
    sim._find_path = lambda *_args: detour.copy()
    assert sim._replan(visitor, require_improvement=True)
    assert visitor.path == detour and visitor.steps_since_plan == 0


def test_blocked_replan_keeps_priority_for_same_path_and_resets_for_detour():
    """同一経路の再採用では待機優先度を保持し、別経路ならリセットする。"""
    sim = Simulation.__new__(Simulation)
    sim.cfg = SimpleNamespace(
        booths=[SimpleNamespace(is_table=False)],
        safe_occupancy_ratio=0.67,
        severe_congestion_weight=4.0,
        crowd_replan_improvement_ratio=0.10,
    )
    sim.layout = SimpleNamespace(receptions=[(4, 0)])
    sim.booth_fields = [np.zeros((2, 5), dtype=np.int32)]
    sim.crowd_risk = np.zeros((2, 5), dtype=np.float64)
    current = [(1, 0), (2, 0), (3, 0), (4, 0)]
    visitor = SimpleNamespace(
        state=WALKING,
        target_booth_id=0,
        position=(0, 0),
        path=current.copy(),
        path_index=0,
        congestion_avoidance=1.0,
        steps_since_plan=7,
        blocked_count=4,
    )

    sim._find_path = lambda *_args: current.copy()
    assert sim._replan(visitor)
    assert visitor.path == current
    assert visitor.blocked_count == 4
    assert visitor.steps_since_plan == 0

    detour = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 0)]
    sim._find_path = lambda *_args: detour.copy()
    assert sim._replan(visitor)
    assert visitor.path == detour
    assert visitor.blocked_count == 0


def test_blocked_replan_respects_cooldown_when_count_is_preserved():
    """待機回数を保持しても、ブロック再探索を毎ステップ実行しない。"""
    sim = Simulation.__new__(Simulation)
    sim.cfg = SimpleNamespace(
        congestion_travel_time_weight=1.0,
        n_substeps=1,
        dt_seconds=1.0,
        cell_size_meters=1.0,
        blocked_replan_count=3,
        replan_interval_steps=100,
        crowd_replan_cooldown_steps=3,
    )
    visitor = SimpleNamespace(
        position=(0, 0),
        speed_mps=0.0,
        move_credit=0.0,
        steps_since_plan=0,
        blocked_count=3,
        path=[(1, 0)],
        path_index=0,
        state=WALKING,
        last_crowd_replan_step=-1_000_000,
        last_blocked_replan_step=-1_000_000,
    )
    sim._moving_mask = object()
    sim._visitors_from_mask = lambda _mask: [visitor]
    sim._refresh_crowd_risk_map = lambda _movers: None
    sim._path_has_crowding_ahead = lambda _visitor: False
    sim._cell_travel_cost = lambda _cell: 1.0
    replan_steps = []

    def record_replan(_visitor, require_improvement=False):
        replan_steps.append(sim.t)
        _visitor.steps_since_plan = 0
        return True

    sim._replan = record_replan
    for t in range(10, 14):
        sim.t = t
        sim._movement_step(t)

    assert replan_steps == [10, 13]
