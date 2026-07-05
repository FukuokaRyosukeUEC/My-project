"""§23.1: 配置制約のテスト。"""
from __future__ import annotations

from src.constraints import compute_violation
from tests.conftest import CENTRAL_LAYOUT, PERIMETER_LAYOUT


def test_valid_layouts(cfg, ctx):
    """手作りの中央密集・周辺分散配置は有効(g == 0)であること。"""
    assert compute_violation(cfg, ctx.venue, CENTRAL_LAYOUT) == 0
    assert compute_violation(cfg, ctx.venue, PERIMETER_LAYOUT) == 0


def test_overlap_detected(cfg, ctx):
    """ブース重複を検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    pos[1] = pos[0]  # B1 を B0 と同位置に
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_out_of_bounds_detected(cfg, ctx):
    """会場外配置を検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    pos[0] = (cfg.grid_width - 1, 0)  # 右へはみ出す
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_forbidden_zone_detected(cfg, ctx):
    """入口周辺の配置不可領域(FORBIDDEN)との重なりを検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    pos[0] = (0, 13)  # 入口周辺 2 セル以内
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_queue_space_detected(cfg, ctx):
    """受付・待ち行列セル列が確保できない配置(下端すれすれ)を検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    spec = cfg.booths[0]
    # 受付は場内だが行列が場外へはみ出す y
    pos[0] = (12, cfg.grid_height - spec.height - 1)
    assert compute_violation(cfg, ctx.venue, tuple(pos)) > 0


def test_narrow_corridor_detected(cfg, ctx):
    """幅 1 セルの通路を通路幅違反として検出できる。"""
    pos = list(CENTRAL_LAYOUT)
    # B0(4x3) と B2(3x3) を 1 セル空きで隣接させる
    pos[0] = (12, 6)
    pos[2] = (17, 6)  # B0 は x12..15、B2 は x17..19 → 隙間 x16 のみ
    g = compute_violation(cfg, ctx.venue, tuple(pos))
    assert g > 0
