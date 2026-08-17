"""店舗種類別の初期仮説と待機列密度の設定検査。"""
from __future__ import annotations

from pathlib import Path

from src.models import load_config


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_config(ROOT / "config")


def test_queue_density_is_two_people_per_square_meter():
    """1m角の待機列セルには最大2人だけ入れる。"""
    cfg = _config()

    assert cfg.cap_queue == 2
    assert cfg.queue_capacity_per_cell == 2


def test_booth_types_have_distinct_operating_assumptions():
    """物販・体験・飲食・音楽で魅力度、収容人数、利用時間を分ける。"""
    cfg = _config()
    booths = {booth.booth_id: booth for booth in cfg.booths}

    goods = [booths[index] for index in (*range(0, 4), *range(8, 12), *range(22, 28))]
    workshops = [booths[index] for index in range(4, 8)]
    normal_food_trucks = [booths[16], booths[17]]
    festival_food_trucks = [booths[18], booths[19]]
    international_stalls = [booths[index] for index in range(28, 31)]

    assert all(
        (booth.attractiveness, booth.capacity, booth.mean_service_seconds)
        == (0.60, 2, 45.0)
        for booth in goods
    )
    assert all(
        (booth.attractiveness, booth.capacity, booth.mean_service_seconds)
        == (0.75, 4, 90.0)
        for booth in workshops
    )
    assert all(
        (booth.attractiveness, booth.capacity, booth.mean_service_seconds)
        == (0.90, 2, 75.0)
        for booth in normal_food_trucks
    )
    assert all(
        (booth.attractiveness, booth.capacity, booth.mean_service_seconds)
        == (0.95, 2, 90.0)
        for booth in festival_food_trucks
    )
    assert all(
        (booth.attractiveness, booth.capacity, booth.mean_service_seconds)
        == (0.85, 2, 60.0)
        for booth in international_stalls
    )

    assert (
        booths[12].attractiveness,
        booths[12].capacity,
        booths[12].mean_service_seconds,
    ) == (0.88, 12, 120.0)
    assert (
        booths[20].attractiveness,
        booths[20].capacity,
        booths[20].mean_service_seconds,
    ) == (0.82, 12, 120.0)


def test_waste_station_has_service_settings_not_shop_attractiveness():
    """ゴミステーションを店舗人気の対象から明示的に除外する。"""
    waste = _config().booths[21]

    assert waste.is_waste_station
    assert waste.kind == "waste_station"
    assert waste.category == "none"
    assert waste.attractiveness == 0.0
    assert waste.capacity == 2
    assert waste.mean_service_seconds == 15.0
    assert waste.max_queue_length == 3
