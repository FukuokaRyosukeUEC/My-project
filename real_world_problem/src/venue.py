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
    WALL,
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
    area_masks: dict = None     # 広場名 -> bool (H, W)(所属広場制約用)


@dataclass(frozen=True)
class Layout:
    """有効配置をグリッドへ適用した結果(§5.1, §5.2)。"""

    grid: np.ndarray              # int8 (H, W) セル種別(BOOTH/QUEUE 反映済み)
    capacity: np.ndarray          # int8 (H, W)
    walkable: np.ndarray          # bool (H, W)
    positions: tuple              # ((booth_x, booth_y), ...) 左上基準(§3.5)
    receptions: tuple             # ブースごとの受付セル (x, y)(§5.1)
    booth_exits: tuple             # ブース体験後に再出現するセル (x, y)
    queue_cells: tuple            # ブースごとの待ち行列セル列 ((x,y),...)(§5.2)


def _points_in_polygon(px: np.ndarray, py: np.ndarray, polygon) -> np.ndarray:
    """偶奇則(レイキャスト)による点内包判定。

    セル中心(半整数)と整数頂点の組み合わせを想定し、境界上の点は扱わない。
    """
    inside = np.zeros(px.shape, dtype=bool)
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        x_at = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (px < x_at)
    return inside


def _rasterize_one(cfg: Config, polygon) -> np.ndarray:
    """セル中心 ((x+0.5), (y+0.5)) × cell_size_meters による 1 多角形のマスク。"""
    H, W = cfg.grid_height, cfg.grid_width
    ys, xs = np.mgrid[0:H, 0:W]
    px = (xs + 0.5) * cfg.cell_size_meters
    py = (ys + 0.5) * cfg.cell_size_meters
    return _points_in_polygon(px, py, polygon)


def _rasterize(cfg: Config, named_polygons) -> np.ndarray:
    """(name, polygon) の列を合併マスクへ変換する。"""
    H, W = cfg.grid_height, cfg.grid_width
    mask = np.zeros((H, W), dtype=bool)
    for _name, poly in named_polygons:
        mask |= _rasterize_one(cfg, poly)
    return mask


def venue_masks(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(場内マスク, 道路マスク) を返す。venue_shape 未指定なら全面が配置可能区域。

    - 場内: 配置区域または道路に含まれ、障害物に含まれないセル
    - 道路: 場内のうち歩行のみ可能なセル(配置区域と重なる部分は配置区域を優先)
    """
    H, W = cfg.grid_height, cfg.grid_width
    if not cfg.venue_areas:
        return np.ones((H, W), dtype=bool), np.zeros((H, W), dtype=bool)
    area = _rasterize(cfg, cfg.venue_areas)
    walkway = _rasterize(cfg, cfg.venue_walkways)
    obstacle = _rasterize(cfg, cfg.venue_obstacles)
    inside = (area | walkway) & ~obstacle
    walkway_only = walkway & ~area & ~obstacle
    return inside, walkway_only


def boundary_openings(walkway_only: np.ndarray) -> tuple[tuple[int, int], ...]:
    """グリッド境界に接する道路セル=領域外への開口部を (x, y) で返す。

    来場者は領域外に通じる道路から到来・退出する想定(入口/出口の区別なし)。
    """
    H, W = walkway_only.shape
    boundary = np.zeros((H, W), dtype=bool)
    boundary[0, :] = boundary[H - 1, :] = True
    boundary[:, 0] = boundary[:, W - 1] = True
    return tuple((int(x), int(y)) for y, x in np.argwhere(walkway_only & boundary))


def build_base_venue(cfg: Config) -> Venue:
    """§4.1〜§4.5 に従いベース会場を構築する。

    - venue_shape の場外セルは WALL(通行・配置とも不可)
    - 配置区域(広場)は EMPTY、道路は FORBIDDEN(通行可・配置不可)。
      広場と道路の境界は自由に通り抜けられる。
    - 入退場セル: 明示指定があればそれを使用。venue_shape 使用時は境界開口部
      (領域外に通じる道路セル)を入口・出口共通の集合として自動導出する。
    - 入退場セルの周囲 clearance セル以内はブース配置不可 FORBIDDEN(通行可)
    """
    H, W = cfg.grid_height, cfg.grid_width
    grid = np.full((H, W), EMPTY, dtype=np.int8)
    capacity = np.full((H, W), cfg.cap_default, dtype=np.int8)

    inside, walkway_only = venue_masks(cfg)
    grid[~inside] = WALL
    grid[walkway_only] = FORBIDDEN

    if cfg.entrance_cells_override or cfg.exit_cells_override or not cfg.venue_areas:
        entrance_cells = cfg.entrance_cells
        exit_cells = cfg.exit_cells
    else:
        entrance_cells = exit_cells = boundary_openings(walkway_only)
        if not entrance_cells:
            raise ValueError(
                "venue_shape にグリッド境界へ達する道路(開口部)がありません。"
                "walkways を境界まで延ばすか、entrance_cells/exit_cells を明示してください"
            )
    for (x, y) in (*entrance_cells, *exit_cells):
        if not (0 <= x < W and 0 <= y < H) or not inside[y, x]:
            raise ValueError(f"入口/出口セル {(x, y)} が会場範囲外です")

    for cells, ctype, cap in (
        (entrance_cells, ENTRANCE, cfg.cap_entrance),
        (exit_cells, EXIT, cfg.cap_exit),
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

    # 広場名 -> マスク(BoothSpec.area の所属広場制約用。障害物セルは除く)
    obstacle = _rasterize(cfg, cfg.venue_obstacles)
    area_masks = {
        name: _rasterize_one(cfg, poly) & ~obstacle
        for name, poly in cfg.venue_areas
    }

    return Venue(
        base_grid=grid,
        base_capacity=capacity,
        entrance_cells=entrance_cells,
        exit_cells=exit_cells,
        area_masks=area_masks,
    )


def reception_cell(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[int, int] | None:
    """設定されたブース受付セルを返す。テーブルは受付を持たない(None)。"""
    spec = cfg.booths[booth_id]
    if spec.is_table:
        return None
    if cfg.reception_position == "bottom_center":
        return (bx + spec.width // 2, by + spec.height)
    if cfg.reception_position == "bottom_left":
        return (bx, by + spec.height)
    raise ValueError(f"unsupported reception_position: {cfg.reception_position!r}")


def booth_exit_cell(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[int, int] | None:
    """設定されたブース体験後の退出セルを返す。テーブルは持たない(None)。"""
    if cfg.booths[booth_id].is_table:
        return None
    if cfg.booth_exit_position == "reception":
        return reception_cell(cfg, booth_id, bx, by)
    if cfg.booth_exit_position == "top_left":
        return (bx, by - 1)
    raise ValueError(f"unsupported exit_position: {cfg.booth_exit_position!r}")


def queue_cells_of(cfg: Config, booth_id: int, bx: int, by: int) -> tuple[tuple[int, int], ...]:
    """受付側を先頭として、設定された形状の待ち行列セル列を返す。

    テーブルは待ち行列を持たないため空を返す。
    """
    spec = cfg.booths[booth_id]
    if spec.is_table:
        return ()
    ex, ey = reception_cell(cfg, booth_id, bx, by)
    if cfg.queue_pattern == "straight_down":
        return tuple((ex, ey + 1 + k) for k in range(spec.max_queue_length))
    if cfg.queue_pattern == "bottom_then_right":
        # ブース下辺を左→右、その後に右辺を下→上へ並ぶ L 字。
        # 左下の受付セルは到着・入場処理専用とし、待機列には含めない。
        cells = [(bx + dx, by + spec.height) for dx in range(spec.width)]
        cells.append((bx + spec.width, by + spec.height))
        cells.extend(
            (bx + spec.width, y)
            for y in range(by + spec.height - 1, by - 1, -1)
        )
        dedicated_cells = (cell for cell in dict.fromkeys(cells) if cell != (ex, ey))
        return tuple(dedicated_cells)[: spec.max_queue_length]
    raise ValueError(f"unsupported queue_pattern: {cfg.queue_pattern!r}")


def booth_cells_of(cfg: Config, booth_id: int, bx: int, by: int) -> list[tuple[int, int]]:
    """ブース占有セル(§3.5)。会場外セルも含めてそのまま返す(制約チェック用)。"""
    spec = cfg.booths[booth_id]
    return [(x, y) for y in range(by, by + spec.height) for x in range(bx, bx + spec.width)]


def position_bounds(cfg: Config, booth_id: int) -> tuple[int, int, int, int]:
    """ブース本体・受付・退出・待機列が場内に収まる左上座標の上下限。"""
    cells = booth_cells_of(cfg, booth_id, 0, 0)
    for aux in (reception_cell(cfg, booth_id, 0, 0), booth_exit_cell(cfg, booth_id, 0, 0)):
        if aux is not None:
            cells.append(aux)
    cells += list(queue_cells_of(cfg, booth_id, 0, 0))
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    return (
        -min(xs),
        cfg.grid_width - 1 - max(xs),
        -min(ys),
        cfg.grid_height - 1 - max(ys),
    )


def apply_layout(cfg: Config, venue: Venue, positions: tuple) -> Layout:
    """有効(制約充足済み)配置をグリッドへ適用する。

    前提: constraints.compute_violation() で g == 0 が確認済みであること。
    """
    grid = venue.base_grid.copy()
    capacity = venue.base_capacity.copy()
    receptions = []
    booth_exits = []
    queue_cells = []
    for bid, (bx, by) in enumerate(positions):
        spec = cfg.booths[bid]
        grid[by : by + spec.height, bx : bx + spec.width] = BOOTH
        receptions.append(reception_cell(cfg, bid, bx, by))   # テーブルは None
        booth_exits.append(booth_exit_cell(cfg, bid, bx, by))
        qcells = queue_cells_of(cfg, bid, bx, by)
        queue_cells.append(qcells)
        for (qx, qy) in qcells:
            grid[qy, qx] = QUEUE
            capacity[qy, qx] = cfg.cap_queue

    walkable = np.isin(grid, WALKABLE_TYPES)
    if not cfg.queue_walkable:
        walkable[grid == QUEUE] = False
    return Layout(
        grid=grid,
        capacity=capacity,
        walkable=walkable,
        positions=tuple(tuple(p) for p in positions),
        receptions=tuple(receptions),
        booth_exits=tuple(booth_exits),
        queue_cells=tuple(queue_cells),
    )
