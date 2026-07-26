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

# theta は北から時計回りの方向コードとする。
THETA_NORTH = 0
THETA_EAST = 1
THETA_SOUTH = 2
THETA_WEST = 3

VALID_THETAS = (
    THETA_NORTH,
    THETA_EAST,
    THETA_SOUTH,
    THETA_WEST,
)

THETA_LABELS = {
    THETA_NORTH: "north",
    THETA_EAST: "east",
    THETA_SOUTH: "south",
    THETA_WEST: "west",
}

THETA_ARROWS = {
    THETA_NORTH: "↑",
    THETA_EAST: "→",
    THETA_SOUTH: "↓",
    THETA_WEST: "←",
}


def _validate_theta(theta: int) -> int:
    theta = int(theta)

    if theta not in VALID_THETAS:
        raise ValueError(
            f"theta must be one of {VALID_THETAS}: "
            f"{theta}"
        )

    return theta


def oriented_booth_size(
    cfg: Config,
    booth_id: int,
    theta: int,
) -> tuple[int, int]:
    """向きを反映したブース本体の (幅, 高さ) を返す。"""
    theta = _validate_theta(theta)
    spec = cfg.booths[booth_id]

    if theta in (
        THETA_NORTH,
        THETA_SOUTH,
    ):
        return spec.width, spec.height

    return spec.height, spec.width


def _rotate_relative_cell(
    spec,
    x: int,
    y: int,
    theta: int,
) -> tuple[int, int]:
    """南向きを基準にした相対セルを、指定方向へ回転する。

    入力の相対座標はブース本体の左上を (0, 0) とする。
    受付・退出口・待機列のような本体外セルにも同じ変換を適用できる。
    回転後も、回転したブース本体の左上が (0, 0) になる。
    """
    theta = _validate_theta(theta)

    w = int(spec.width)
    h = int(spec.height)

    if theta == THETA_SOUTH:
        return x, y

    if theta == THETA_EAST:
        return y, w - 1 - x

    if theta == THETA_NORTH:
        return w - 1 - x, h - 1 - y

    return h - 1 - y, x


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
    positions: tuple              # ((booth_x, booth_y, theta), ...) 回転後左上基準
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


def _south_reception_relative(
    cfg: Config,
    booth_id: int,
) -> tuple[int, int] | None:
    """南向きを基準にした受付セルの相対座標を返す。

    受付・待機列・退出は常にブース本体の南辺(theta 回転前の正面)に配置する。
    ブースの正面方向は theta で表現するため、この関数自体はアスペクト比に
    依存しない(theta と正面方向を一致させるため)。
    """
    spec = cfg.booths[booth_id]

    if spec.is_table:
        return None

    if cfg.reception_position == "bottom_center":
        return (
            spec.width // 2,
            spec.height,
        )

    if cfg.reception_position == "bottom_left":
        return (
            0,
            spec.height,
        )

    raise ValueError(
        "unsupported reception_position: "
        f"{cfg.reception_position!r}"
    )


def reception_cell(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> tuple[int, int] | None:
    """角度を反映したブース受付セルを返す。テーブルは None。"""
    relative = _south_reception_relative(
        cfg,
        booth_id,
    )

    if relative is None:
        return None

    rx, ry = _rotate_relative_cell(
        cfg.booths[booth_id],
        relative[0],
        relative[1],
        theta,
    )

    return (
        bx + rx,
        by + ry,
    )


def booth_exit_cell(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> tuple[int, int] | None:
    """角度を反映した体験後退出セルを返す。テーブルは None。"""
    spec = cfg.booths[booth_id]

    if spec.is_table:
        return None

    if cfg.booth_exit_position == "reception":
        return reception_cell(
            cfg,
            booth_id,
            bx,
            by,
            theta,
        )

    if cfg.booth_exit_position == "top_left":
        relative = (0, -1)
    elif cfg.booth_exit_position == "bottom_right":
        # 受付(南辺左端)と同じ辺(南辺=正面)の右端。待ち行列は
        # queue_pattern="left_then_clockwise" と組み合わせて南辺を空けておく前提。
        relative = (spec.width - 1, spec.height)
    else:
        raise ValueError(
            "unsupported exit_position: "
            f"{cfg.booth_exit_position!r}"
        )

    ex, ey = _rotate_relative_cell(
        spec,
        relative[0],
        relative[1],
        theta,
    )

    return (
        bx + ex,
        by + ey,
    )


def queue_cells_of(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> tuple[tuple[int, int], ...]:
    """受付側を先頭として、角度を反映した待ち行列セル列を返す。"""
    spec = cfg.booths[booth_id]

    if spec.is_table:
        return ()

    reception_relative = _south_reception_relative(
        cfg,
        booth_id,
    )

    assert reception_relative is not None

    if cfg.queue_pattern == "straight_down":
        south_cells = [
            (
                reception_relative[0],
                reception_relative[1] + 1 + k,
            )
            for k in range(
                spec.max_queue_length
            )
        ]

    elif cfg.queue_pattern == "bottom_then_right":
        # 南向き時のL字形状を作成後、ブース角度に合わせて回転する。
        cells = [
            (dx, spec.height)
            for dx in range(spec.width)
        ]

        cells.append(
            (
                spec.width,
                spec.height,
            )
        )

        cells.extend(
            (
                spec.width,
                y,
            )
            for y in range(
                spec.height - 1,
                -1,
                -1,
            )
        )

        south_cells = [
            cell
            for cell in dict.fromkeys(cells)
            if cell != reception_relative
        ][:spec.max_queue_length]

    elif cfg.queue_pattern == "left_then_clockwise":
        # 受付セル(南辺=正面)の左隣を起点に、ブース外周を時計回り
        # (左辺→上辺→右辺)へ一周する。南辺は受付・退出セル専用に空けておき、
        # exit_position="bottom_right"(南辺右端)と重複しないようにする。
        cells = [
            (reception_relative[0] - 1, y)
            for y in range(spec.height, -2, -1)
        ]

        cells.extend(
            (x, -1)
            for x in range(reception_relative[0] - 1, spec.width + 1)
        )

        cells.extend(
            (spec.width, y)
            for y in range(-1, spec.height + 1)
        )

        south_cells = [
            cell
            for cell in dict.fromkeys(cells)
            if cell != reception_relative
            and not (cell[1] == spec.height and 0 <= cell[0] < spec.width)
        ][:spec.max_queue_length]

    else:
        raise ValueError(
            "unsupported queue_pattern: "
            f"{cfg.queue_pattern!r}"
        )

    rotated = [
        _rotate_relative_cell(
            spec,
            x,
            y,
            theta,
        )
        for x, y in south_cells
    ]

    return tuple(
        (
            bx + x,
            by + y,
        )
        for x, y in rotated
    )


def booth_cells_of(
    cfg: Config,
    booth_id: int,
    bx: int,
    by: int,
    theta: int,
) -> list[tuple[int, int]]:
    """角度を反映したブース本体の占有セルを返す。"""
    width, height = oriented_booth_size(
        cfg,
        booth_id,
        theta,
    )

    return [
        (x, y)
        for y in range(
            by,
            by + height,
        )
        for x in range(
            bx,
            bx + width,
        )
    ]


def position_bounds(
    cfg: Config,
    booth_id: int,
    theta: int,
) -> tuple[int, int, int, int]:
    """指定角度で本体・受付・退出・待機列が場内に収まる座標範囲。"""
    cells = booth_cells_of(
        cfg,
        booth_id,
        0,
        0,
        theta,
    )

    for aux in (
        reception_cell(
            cfg,
            booth_id,
            0,
            0,
            theta,
        ),
        booth_exit_cell(
            cfg,
            booth_id,
            0,
            0,
            theta,
        ),
    ):
        if aux is not None:
            cells.append(aux)

    cells += list(
        queue_cells_of(
            cfg,
            booth_id,
            0,
            0,
            theta,
        )
    )

    xs = [
        x
        for x, _ in cells
    ]

    ys = [
        y
        for _, y in cells
    ]

    return (
        -min(xs),
        cfg.grid_width - 1 - max(xs),
        -min(ys),
        cfg.grid_height - 1 - max(ys),
    )


def apply_layout(
    cfg: Config,
    venue: Venue,
    positions: tuple,
) -> Layout:
    """有効な ``((x, y, theta), ...)`` 配置をグリッドへ適用する。"""
    grid = venue.base_grid.copy()
    capacity = venue.base_capacity.copy()

    receptions = []
    booth_exits = []
    queue_cells = []

    for bid, (
        bx,
        by,
        theta,
    ) in enumerate(positions):

        for x, y in booth_cells_of(
            cfg,
            bid,
            bx,
            by,
            theta,
        ):
            grid[y, x] = BOOTH

        receptions.append(
            reception_cell(
                cfg,
                bid,
                bx,
                by,
                theta,
            )
        )

        booth_exits.append(
            booth_exit_cell(
                cfg,
                bid,
                bx,
                by,
                theta,
            )
        )

        qcells = queue_cells_of(
            cfg,
            bid,
            bx,
            by,
            theta,
        )

        queue_cells.append(qcells)

        for qx, qy in qcells:
            grid[qy, qx] = QUEUE
            capacity[qy, qx] = cfg.cap_queue

    walkable = np.isin(
        grid,
        WALKABLE_TYPES,
    )

    if not cfg.queue_walkable:
        walkable[grid == QUEUE] = False

    return Layout(
        grid=grid,
        capacity=capacity,
        walkable=walkable,
        positions=tuple(
            tuple(p)
            for p in positions
        ),
        receptions=tuple(receptions),
        booth_exits=tuple(booth_exits),
        queue_cells=tuple(queue_cells),
    )