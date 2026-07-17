"""preview_venue_shape.py — config の venue_shape から会場外形のみを図で出力する。

会場形状の微調整用。default_config.yaml を編集するたびに実行して確認する。

使用例(リポジトリルートから):
    python real_world_problem/tests/preview_venue_shape.py
    python real_world_problem/tests/preview_venue_shape.py --out /tmp/venue.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.models import EMPTY, FORBIDDEN, WALL, load_config  # noqa: E402
from src.venue import build_base_venue  # noqa: E402

# 日本語ラベル用フォント(存在するもののみ適用)
import matplotlib.font_manager as _fm  # noqa: E402

for _f in ("Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans", "Yu Gothic", "Meiryo"):
    if any(f.name == _f for f in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config"), help="config ディレクトリ")
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "venue_shape.png"),
        help="出力 PNG パス",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    venue = build_base_venue(cfg)
    g = venue.base_grid
    H, W = cfg.grid_height, cfg.grid_width

    # 表示区分: 0=広場(配置可) / 1=場外・障害物 / 2=道路(歩行のみ) / 3=開口部(入退場)
    disp = np.zeros((H, W), dtype=int)
    disp[g == WALL] = 1
    disp[g == FORBIDDEN] = 2
    for (x, y) in {*venue.entrance_cells, *venue.exit_cells}:
        disp[y, x] = 3
    cmap = ListedColormap(["#ffffff", "#4d4d4d", "#cfe8cf", "#f4a9a8"])

    fig, ax = plt.subplots(figsize=(max(4.0, W / 6), max(6.0, H / 6)))
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(0, W, 5))
    ax.set_yticks(range(0, H, 5))
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#eeeeee", linewidth=0.2)
    ax.grid(which="major", color="#bbbbbb", linewidth=0.4)
    ax.set_title(
        f"会場外形 {W}×{H} セル(1 セル = {cfg.cell_size_meters:g} m)\n"
        f"広場 {int(np.sum(g == EMPTY))} / 道路 {int(np.sum(g == FORBIDDEN))} / "
        f"開口部 {len(venue.entrance_cells)} セル"
    )
    ax.legend(
        handles=[
            Patch(facecolor="#ffffff", edgecolor="#999999", label="広場(ブース配置可)"),
            Patch(facecolor="#cfe8cf", label="道路(歩行のみ)"),
            Patch(facecolor="#f4a9a8", label="開口部(入退場)"),
            Patch(facecolor="#4d4d4d", label="場外・障害物"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"会場外形図を出力しました: {out}")


if __name__ == "__main__":
    main()
