"""同種ブース近接制約の設定・距離・例外を検証する。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.constraints import (
    build_placement_geometry,
    compute_same_type_proximity_violation,
    compute_violation,
)
from src.direct_optimization import (
    Direct138LayoutProblem,
    DirectMultiScaleMutation,
    _proximity_construction_groups,
    to_integer_layout,
)
from src.models import load_config
from src.venue import build_base_venue


ROOT = Path(__file__).resolve().parents[1]


def _real_config():
    return load_config(ROOT / "config")


def test_real_config_defines_expected_proximity_groups_and_exceptions():
    cfg = _real_config()
    counts = {
        group: sum(booth.placement_group == group for booth in cfg.booths)
        for group in {booth.placement_group for booth in cfg.booths}
        if group
    }

    assert cfg.same_type_max_distance_meters == 5.0
    assert counts == {"normal_kitchen_car": 2}
    assert all(
        not booth.placement_group
        for booth in cfg.booths
        if booth.kind == "booth"
    )
    assert all(not booth.placement_group for booth in cfg.booths if booth.is_table)
    assert all(
        not booth.placement_group
        for booth in cfg.booths
        if booth.is_waste_station
    )


def test_layout_constructor_separates_only_normal_kitchen_cars():
    cfg = _real_config()
    groups = _proximity_construction_groups(cfg)
    kitchen_car_groups = [
        group for group in groups if group.kind == "type:normal_kitchen_car"
    ]

    assert len(kitchen_car_groups) == 1
    assert kitchen_car_groups[0].booth_ids == (16, 17)
    assert all(
        cfg.booths[booth_id].kind == "kitchen_car"
        and cfg.booths[booth_id].theme == "normal"
        for booth_id in kitchen_car_groups[0].booth_ids
    )
    assert all(
        not cfg.booths[booth_id].placement_group
        for booth_id in (18, 19, 28, 29, 30)
    )


def test_macro_donor_fallback_preserves_kitchen_car_constraint():
    cfg = _real_config()
    problem = Direct138LayoutProblem(cfg)
    mutation = DirectMultiScaleMutation(cfg)

    donor = mutation._random_donor(problem, np.random.default_rng(1))

    assert donor is not None
    positions = to_integer_layout(donor, cfg)
    assert problem.physical_violation(donor) == 0.0
    assert compute_same_type_proximity_violation(cfg, positions) == 0.0


def test_five_meter_boundary_is_allowed_and_six_meters_is_rejected():
    cfg = _real_config()
    positions = [booth.fixed_position for booth in cfg.booths]

    # 通常キッチンカー1番を2番の中心からちょうど5mへ置く。
    positions[16] = (39, 56, 3)
    geometry = build_placement_geometry(cfg, tuple(positions))
    assert compute_same_type_proximity_violation(
        cfg, tuple(positions), geometry
    ) == 0.0

    # 6mへ離すと、互いに対象近傍を失う2店舗が違反となる。
    positions[16] = (38, 56, 3)
    geometry = build_placement_geometry(cfg, tuple(positions))
    assert compute_same_type_proximity_violation(
        cfg, tuple(positions), geometry
    ) == 2.0


def test_table_waste_station_and_singleton_group_are_always_exempt():
    cfg = _real_config()
    booths = []
    for booth in cfg.booths:
        if booth.booth_id in (0, 1):
            group = "ignored_normal_booth"
        elif booth.is_table or booth.is_waste_station:
            group = "ignored_equipment"
        elif booth.booth_id == 12:
            group = "singleton_exhibit"
        else:
            group = ""
        booths.append(replace(booth, placement_group=group))
    cfg = replace(cfg, booths=tuple(booths))
    positions = tuple(booth.fixed_position for booth in cfg.booths)

    assert compute_same_type_proximity_violation(cfg, positions) == 0.0


def test_fixed_reference_is_not_rejected_by_optimized_layout_only_rule():
    cfg = _real_config()
    venue = build_base_venue(cfg)
    positions = tuple(booth.fixed_position for booth in cfg.booths)
    geometry = build_placement_geometry(cfg, positions)

    assert compute_violation(cfg, venue, positions, geometry=geometry) == 0.0
    assert compute_same_type_proximity_violation(cfg, positions, geometry) == 2.0
    assert compute_violation(
        cfg,
        venue,
        positions,
        geometry=geometry,
        enforce_same_type_proximity=True,
    ) == 2.0
