"""venue.py — 会場グリッド・セル種別・容量・入口出口・レイアウト適用(§4, §5)。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import (
    BOOTH,
    EMPTY,
    ENTRANCE,
    EXIT,
    FORBIDDEN,
    QUEUE,
    WALKABLE_TYPES,
    Config,
)


@dataclass(frozen=True)
class Venue:
    """ベース会場(ブース未配置)。"""

    base_grid: np.ndarray       # int8 (H, W) セル種別
    base_capacity: np.ndarray   # int8 (H, W)
    entrance_cells: tuple       # ((x,y), ...)
    exit_cells: tuple


@dataclass(frozen=True)
class Layout:
    """有効配置をグリッドへ適用した結果(§5.1, §5.2)。"""

    grid: np.ndarray              # int8 (H, W) セル種別(BOOTH/QUEUE 反映済み)
    capacity: np.ndarray          # int8 (H, W)
    walkable: np.ndarray          # bool (H, W)
    positions: tuple              # ((booth_x, booth_y), ...) 左上基準(§3.5)
    receptions: tuple             # ブースごとの受付セル (x, y)(§5.1)
    queue_cells: tuple            # ブースごとの待ち行列セル列 ((x,y),...)(§5.2)


def build_base_venue(cfg: Config) -> Venue:
    """§4.1〜§4.5 に従いベース会場を構築する。

    - 全セル EMPTY、既定容量
    - 入口: 左辺中央 3 セル / 出口: 右辺中央 3 セル
    - 入口・出口の周囲 2 セル(チェビシェフ距離)以内はブース配置不可 FORBIDDEN(通行可)
    """
    H, W = cfg.grid_height, cfg.grid_width
    grid = np.full((H, W), EMPTY, dtype=np.int8)
    capacity = np.full((H, W), cfg.cap_default, dtype=np.int8)

    for cells, ctype, cap in (
        (cfg.entrance_cells, ENTRANCE, cfg.cap_entrance),
        (cfg.exit_cells, EXIT, cfg.cap_exit),
    ):
        # 周囲 clearance セルを FORBIDDEN 化(先に塗ってから本体を上書き)
        r = cfg.entrance_clearance
        for (x, y) in cells:
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            zone = grid[y0:y1, x0:x1]
            zone[zone == EMPTY] = FORBIDDEN
        for (x, y) in cells:
            grid[y, x] = ctype
            capacity[y, x] = cap

    return Venue(
        base_grid=grid,
        base_capacity=capacity,
        entrance_cells=cfg.entrance_cells,
        exit_cells=cfg.exit_cells,
    )


def reception_cell(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[int, int]:
    """§5.1: 受付セルはブース下辺中央の 1 セル下(y は下へ増加)。"""
    spec = cfg.booths[booth_id]
    return (bx + spec.width // 2, by + spec.height)


def queue_cells_of(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[tuple[int, int], ...]:
    """§5.2: 待ち行列セルは受付セルからさらに下方向へ直線に max_queue_length セル。"""
    spec = cfg.booths[booth_id]
    ex, ey = reception_cell(cfg, booth_id, bx, by)
    return tuple((ex, ey + 1 + k) for k in range(spec.max_queue_length))


def booth_cells_of(cfg: Config, booth_id: int, bx: int, by: int) -> list[tuple[int, int]]:
    """ブース占有セル(§3.5)。会場外セルも含めてそのまま返す(制約チェック用)。"""
    spec = cfg.booths[booth_id]
    return [(x, y) for y in range(by, by + spec.height) for x in range(bx, bx + spec.width)]


def apply_layout(cfg: Config, venue: Venue, positions: tuple) -> Layout:
    """有効(制約充足済み)配置をグリッドへ適用する。

    前提: constraints.compute_violation() で g == 0 が確認済みであること。
    """
    grid = venue.base_grid.copy()
    capacity = venue.base_capacity.copy()
    receptions = []
    queue_cells = []
    for bid, (bx, by) in enumerate(positions):
        spec = cfg.booths[bid]
        grid[by : by + spec.height, bx : bx + spec.width] = BOOTH
        rx, ry = reception_cell(cfg, bid, bx, by)
        receptions.append((rx, ry))
        qcells = queue_cells_of(cfg, bid, bx, by)
        queue_cells.append(qcells)
        for (qx, qy) in qcells:
            grid[qy, qx] = QUEUE
            capacity[qy, qx] = cfg.cap_queue

    walkable = np.isin(grid, WALKABLE_TYPES)
    return Layout(
        grid=grid,
        capacity=capacity,
        walkable=walkable,
        positions=tuple(tuple(p) for p in positions),
        receptions=tuple(receptions),
        queue_cells=tuple(queue_cells),
    )
