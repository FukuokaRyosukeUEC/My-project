"""visualization.py — 最適化終了後の可視化(§20)。最適化中は使用しない(§17.1)。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 日本語ラベル用フォント(存在するもののみ適用)
for _f in ("Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans", "Yu Gothic", "Meiryo"):
    try:
        import matplotlib.font_manager as _fm

        if any(f.name == _f for f in _fm.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    except Exception:  # pragma: no cover
        break
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from .models import (
    BOOTH,  # noqa: F401 (セル種別の網羅性のため保持)
    CHOOSING,
    ENTRANCE,
    EXIT,
    EXITING,
    EXPERIENCING,
    FORBIDDEN,
    QUEUE,
    QUEUING,
    WALKING,
    WALL,
    Config,
)
from .venue import (
    Layout,
    THETA_ARROWS,
    THETA_EAST,
    THETA_NORTH,
    THETA_SOUTH,
    THETA_WEST,
    oriented_booth_size,
)

#: 正面(theta)を向く三角マーカー。退場口はこの向き(正面と反対=ブースから離れる向き)、
#: 入場口はこの逆向き(正面を向く=ブースに近づく向き)を使う。
_FRONT_MARKERS = {
    THETA_NORTH: "^",
    THETA_EAST: ">",
    THETA_SOUTH: "v",
    THETA_WEST: "<",
}
_OPPOSITE_THETA = {
    THETA_NORTH: THETA_SOUTH,
    THETA_EAST: THETA_WEST,
    THETA_SOUTH: THETA_NORTH,
    THETA_WEST: THETA_EAST,
}

#: §20.1 エージェント色分け(赤=移動不可待機は blocked_count で近似表示)
STATE_COLORS = {
    CHOOSING: "#1f77b4",
    WALKING: "#1f77b4",   # 青: ブースへ移動中
    QUEUING: "#ffcc00",   # 黄: 待ち行列
    EXPERIENCING: "#2ca02c",  # 緑: 体験中
    EXITING: "#9467bd",   # 紫: 出口へ移動中
}
BLOCKED_COLOR = "#d62728"  # 赤: セル容量により移動できず待機中

_CELL_FACE = {
    WALL: "#4d4d4d",
    QUEUE: "#f7e8a0",
    ENTRANCE: "#a6d8a8",
    EXIT: "#f4a9a8",
    FORBIDDEN: "#e6e6e6",
}

#: (kind, "normal") の既定塗り色・ラベル。theme が normal 以外のブースは
#: booth_style_map() が booths.yaml の登場順にパレットから都度自動割当する
#: (theme 文字列自体をラベルとして使うため、booths.yaml 側で意味の分かる
#: 文字列にしておく。例: theme: ジャズ / ゴミステーション)。
_BASE_ATTR_STYLE = {
    ("booth", "normal"): ("#8c564b", "通常ブース"),        # 茶
    ("kitchen_car", "normal"): ("#ff7f0e", "キッチンカー"),  # 橙
    ("table", "normal"): ("#9e9e9e", "テーブル・椅子"),      # 灰
}
_EXTRA_PALETTE = (
    "#9467bd", "#e377c2", "#2ca02c", "#17becf", "#bcbd22",
    "#d62728", "#1f77b4", "#8dd3c7", "#fdb462", "#b3de69",
)
_AREA_EDGE_PALETTE = (
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#d62728",
    "#17becf",
)


def booth_style_map(cfg: Config) -> dict:
    """cfg.booths に登場する (kind, theme) ごとの (色, ラベル) を都度割り当てる。

    normal は固定色。それ以外の theme は cfg.booths の登場順にパレットから
    自動割当するため、booths.yaml に theme を追加・変更するだけで色分けに
    反映される(コード側の色定義追加は不要)。
    """
    style = dict(_BASE_ATTR_STYLE)
    for spec in cfg.booths:
        theme = "normal" if spec.is_table else spec.theme
        key = (spec.kind, theme)
        if key in style:
            continue
        color = _EXTRA_PALETTE[(len(style) - len(_BASE_ATTR_STYLE)) % len(_EXTRA_PALETTE)]
        style[key] = (color, theme)
    return style


def _booth_face(spec, style_map: dict) -> str:
    key = (spec.kind, "normal" if spec.is_table else spec.theme)
    return style_map.get(key, style_map[("booth", "normal")])[0]


def draw_venue(ax, cfg: Config, layout: Layout, legend: bool = False) -> None:
    """会場・ブース・受付・行列セルの静的描画。ブースは属性で色分けする。"""
    H, W = cfg.grid_height, cfg.grid_width
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # y は下へ増加(§3.5)
    ax.set_aspect("equal")
    grid = layout.grid
    for ctype, color in _CELL_FACE.items():
        ys, xs = np.where(grid == ctype)
        for x, y in zip(xs, ys):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="none"))
    style_map = booth_style_map(cfg)
    used_attrs = []
    for b, (
        bx,
        by,
        theta,
    ) in enumerate(layout.positions):
        spec = cfg.booths[b]

        width, height = oriented_booth_size(
            cfg,
            b,
            theta,
        )

        face = _booth_face(spec, style_map)

        key = (
            spec.kind,
            "normal"
            if spec.is_table
            else spec.theme,
        )

        if key not in used_attrs:
            used_attrs.append(key)

        ax.add_patch(
            Rectangle(
                (
                    bx - 0.5,
                    by - 0.5,
                ),
                width,
                height,
                facecolor=face,
                edgecolor="black",
                linewidth=1.2,
            )
        )

        if not spec.is_table:
            ax.text(
                bx + width / 2 - 0.5,
                by + height / 2 - 0.5,
                f"B{b + 1} "
                f"{THETA_ARROWS[int(theta)]}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

        if layout.receptions[b] is not None:
            rx, ry = layout.receptions[b]

            ax.plot(
                rx,
                ry,
                marker=_FRONT_MARKERS[_OPPOSITE_THETA[int(theta)]],
                color="black",
                markersize=4,
            )

        if layout.booth_exits[b] is not None:
            ex, ey = layout.booth_exits[b]

            ax.plot(
                ex,
                ey,
                marker=_FRONT_MARKERS[int(theta)],
                color="black",
                markersize=4,
            )
    if legend:
        handles = [
            Patch(facecolor=style_map[k][0], edgecolor="black", label=style_map[k][1])
            for k in used_attrs
        ]
        handles.append(Patch(facecolor=_CELL_FACE[QUEUE], label="待機列"))
        handles.append(Patch(facecolor=_CELL_FACE[FORBIDDEN], label="道路(配置不可)"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=8, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_pareto_front(
    F: np.ndarray,
    reps: dict[str, int],
    path: str | Path,
    initial_f: np.ndarray | None = None,
    fixed_f: np.ndarray | None = None,
) -> None:
    """
    最終パレート解、代表解、比較用の初期解、および固定配置を描画する。

    Parameters
    ----------
    F
        最終パレート解の目的関数値。
        各行は [f_move, f_congestion]。
    reps
        代表解名とF内のインデックスの対応。
    path
        グラフの保存先。
    initial_f
        初期解の目的関数値 [f_move, f_congestion]。
        Noneの場合は初期解を描画しない。
    fixed_f
        booths.yaml の固定配置の目的関数値
        [f_move, f_congestion]。
        Noneの場合は固定配置を描画しない。
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # ---------------------------------------------------------
    # 最終パレート解
    # ---------------------------------------------------------
    ax.scatter(
        F[:, 0],
        F[:, 1],
        c="tab:blue",
        s=30,
        alpha=0.75,
        label="Pareto solutions",
        zorder=2,
    )

    # ---------------------------------------------------------
    # 代表解
    # ---------------------------------------------------------
    marks = {
        "best_movement": (
            "s",
            "tab:green",
            "Movement-oriented solution",
        ),
        "best_congestion": (
            "^",
            "tab:red",
            "Congestion-oriented solution",
        ),
        "knee": (
            "*",
            "tab:orange",
            "Balanced solution (knee)",
        ),
    }

    for name, idx in reps.items():
        if name not in marks:
            continue

        marker, color, label = marks[name]

        ax.scatter(
            F[idx, 0],
            F[idx, 1],
            marker=marker,
            c=color,
            s=150,
            edgecolors="black",
            linewidths=0.8,
            label=label,
            zorder=4,
        )

    # ---------------------------------------------------------
    # 比較に使用したランダム初期解
    # ---------------------------------------------------------
    if initial_f is not None:
        initial_f = np.asarray(
            initial_f,
            dtype=float,
        ).reshape(-1)

        if initial_f.size != 2:
            raise ValueError(
                "initial_f must contain exactly two values: "
                "[f_move, f_congestion]."
            )

        if not np.all(np.isfinite(initial_f)):
            raise ValueError(
                f"initial_f contains non-finite values: {initial_f}"
            )

        ax.scatter(
            initial_f[0],
            initial_f[1],
            marker="D",
            c="black",
            s=110,
            edgecolors="white",
            linewidths=1.0,
            label="Random solution",
            zorder=5,
        )

        ax.annotate(
            "Random",
            xy=(initial_f[0], initial_f[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )

    # ---------------------------------------------------------
    # booths.yaml に記載された配置計画
    # ---------------------------------------------------------
    if fixed_f is not None:
        fixed_f = np.asarray(
            fixed_f,
            dtype=float,
        ).reshape(-1)

        if fixed_f.size != 2:
            raise ValueError(
                "fixed_f must contain exactly two values: "
                "[f_move, f_congestion]."
            )

        if not np.all(np.isfinite(fixed_f)):
            raise ValueError(
                f"fixed_f contains non-finite values: {fixed_f}"
            )

        ax.scatter(
            fixed_f[0],
            fixed_f[1],
            marker="X",
            c="tab:purple",
            s=150,
            edgecolors="black",
            linewidths=0.8,
            label="Layout plan (booths.yaml)",
            zorder=5,
        )

        ax.annotate(
            "booths.yaml",
            xy=(fixed_f[0], fixed_f[1]),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )

    ax.set_xlabel("f_move: mean free-walk time [s]")
    ax.set_ylabel("f_congestion")
    ax.set_title("Pareto front and comparison layouts")

    # どちらの目的関数も最小化であることを明示する。
    ax.text(
        0.02,
        0.98,
        "Lower-left is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
def plot_hypervolume_history(
    hv_history: pd.DataFrame,
    path: str | Path,
) -> None:
    """世代ごとのHypervolume推移を画像として保存する。"""

    required = {
        "generation",
        "hypervolume",
    }

    missing = required.difference(
        hv_history.columns
    )

    if missing:
        raise ValueError(
            "hv_history is missing columns: "
            f"{sorted(missing)}"
        )

    if hv_history.empty:
        raise ValueError(
            "hv_history must not be empty"
        )

    generations = hv_history[
        "generation"
    ].to_numpy(
        dtype=int,
    )

    hypervolume = hv_history[
        "hypervolume"
    ].to_numpy(
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        generations,
        hypervolume,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
    )

    ax.set_xlabel(
        "Generation"
    )
    ax.set_ylabel(
        "Hypervolume (HV)"
    )
    ax.set_title(
        "Hypervolume over generations"
    )
    ax.grid(
        alpha=0.3
    )

    # 値が非常に小さい、または大きい場合は科学表記を使用
    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 4),
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)
    
def plot_time_series_comparison(
    initial_result,
    representative_results: dict,
    congestion_path: str | Path,
    movement_path: str | Path,
) -> None:
    """
    初期解と代表3解について、1パネルのグラフを2ファイル保存する。

    congestion_path:
        タイムステップ－混雑度

    movement_path:
        タイムステップ－移動負担
    """

    series = [
        ("初期解", initial_result),
        ("移動重視解", representative_results["best_movement"]),
        ("バランス解", representative_results["knee"]),
        ("混雑重視解", representative_results["best_congestion"]),
    ]

    # =========================================================
    # 混雑度
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, result in series:
        ax.plot(
            result.time_steps,
            result.congestion_history,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("タイムステップ")
    ax.set_ylabel("混雑度")
    ax.set_title("タイムステップごとの混雑度")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(congestion_path, dpi=150)
    plt.close(fig)

    # =========================================================
    # 移動負担
    # =========================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, result in series:
        ax.plot(
            result.time_steps,
            result.movement_history,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("タイムステップ")
    ax.set_ylabel("移動負担 [s/人]")
    ax.set_title("タイムステップごとの移動負担")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(movement_path, dpi=150)
    plt.close(fig)


def plot_heatmaps(cfg: Config, layout: Layout, heatmap, over_map, path: str | Path, title="") -> None:
    """§20.2: ヒートマップへ属性色と広場区画境界を重ねて描画する。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, name in (
        (axes[0], heatmap, "cumulative presence [person-steps]"),
        (axes[1], over_map, "time over safe occupancy [s]"),
    ):
        masked = np.where(layout.walkable, data, np.nan)
        im = ax.imshow(masked, cmap="hot_r", origin="upper")
        draw_venue_outline(ax, cfg, layout)
        ax.set_title(f"{title} {name}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04)
    style_map = booth_style_map(cfg)
    used_attrs = []
    for spec in cfg.booths:
        key = (spec.kind, "normal" if spec.is_table else spec.theme)
        if key not in used_attrs:
            used_attrs.append(key)
    handles = [
        Patch(facecolor=style_map[key][0], edgecolor="black", label=style_map[key][1])
        for key in used_attrs
    ]
    for i, (area_name, _polygon) in enumerate(cfg.venue_areas):
        handles.append(
            Line2D(
                [0],
                [0],
                color=_AREA_EDGE_PALETTE[i % len(_AREA_EDGE_PALETTE)],
                linewidth=2.0,
                label=f"区画: {area_name}",
            )
        )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(5, len(handles)),
            fontsize=8,
            framealpha=0.9,
        )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_venue_outline(
    ax,
    cfg: Config,
    layout: Layout,
) -> None:
    # venue_shape.areasをセル境界座標へ変換し、上・下広場などの区画を明示する。
    for i, (_area_name, polygon) in enumerate(cfg.venue_areas):
        vertices = [
            (
                float(px) / cfg.cell_size_meters - 0.5,
                float(py) / cfg.cell_size_meters - 0.5,
            )
            for px, py in polygon
        ]
        edge = _AREA_EDGE_PALETTE[i % len(_AREA_EDGE_PALETTE)]
        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                fill=False,
                edgecolor=edge,
                linewidth=1.5,
                linestyle="--",
                zorder=3,
            )
        )
    style_map = booth_style_map(cfg)
    for b, (
        bx,
        by,
        theta,
    ) in enumerate(layout.positions):
        spec = cfg.booths[b]

        width, height = oriented_booth_size(
            cfg,
            b,
            theta,
        )

        ax.add_patch(
            Rectangle(
                (
                    bx - 0.5,
                    by - 0.5,
                ),
                width,
                height,
                fill=True,
                facecolor=_booth_face(spec, style_map),
                edgecolor="black",
                linewidth=1.0,
                zorder=4,
            )
        )

        if not spec.is_table:
            ax.text(
                bx + width / 2 - 0.5,
                by + height / 2 - 0.5,
                f"B{b + 1} "
                f"{THETA_ARROWS[int(theta)]}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                zorder=5,
            )

        if layout.receptions[b] is not None:
            rx, ry = layout.receptions[b]
            ax.plot(
                rx,
                ry,
                marker=_FRONT_MARKERS[_OPPOSITE_THETA[int(theta)]],
                color="black",
                markersize=3.5,
                zorder=6,
            )
        if layout.booth_exits[b] is not None:
            ex, ey = layout.booth_exits[b]
            ax.plot(
                ex,
                ey,
                marker=_FRONT_MARKERS[int(theta)],
                color="black",
                markersize=3.5,
                zorder=6,
            )

    ax.set_xlim(-0.5, cfg.grid_width - 0.5)
    ax.set_ylim(cfg.grid_height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def comparison_table(results: dict[str, dict], path: str | Path) -> pd.DataFrame:
    """§20.3 の比較表を CSV へ保存し DataFrame を返す。

    results: {"移動重視": kpis+目的関数 dict, ...}
    """
    rows = {
        "移動負担 f_move [s/人]": "mean_free_walk_seconds",
        "平均待ち時間 [s]": "mean_queue_seconds",
        "平均ブロック待機時間 [s]": "mean_blocked_seconds",
        "混雑度 f_congestion": "f_congestion",
        "会場全体混雑（監査値）": "f_congestion_raw",
        "ブース近傍混雑": "f_congestion_booth_near",
        "最大占有率": "max_cell_occupancy_ratio",
        "体験成功率": "experience_success_rate",
        "諦め回数": "give_up_total",
        "平均幸福度": "mean_happiness",
        "整然さ最大制約比 (≤1)": "orderliness",
        "配置モチーフ非所属率": "organization_nonmembership",
        "配置モチーフ複雑度": "motif_complexity",
        "列・外周基準からの誤差": "alignment_error",
        "モチーフ内間隔変動": "gap_variation",
        "向き不一致率": "orientation",
        "孤立率": "isolation",
        "ブース正面通路閉塞率": "front_clearance",
    }
    table = pd.DataFrame(
        {name: {label: vals.get(key, np.nan) for label, key in rows.items()}
         for name, vals in results.items()}
    )
    table.to_csv(path, encoding="utf-8-sig")
    return table


def render_animation(cfg: Config, layout: Layout, frames, path: str | Path, fps: int = 8) -> None:
    """§20.1: エージェントアニメーション(GIF)。frames は Simulation.record_frames の出力。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.72)
    draw_venue(ax, cfg, layout)
    scat = ax.scatter([], [], s=18)
    label = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=9)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[WALKING],
               label="移動中・目的地選択", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[QUEUING],
               label="待機列", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[EXPERIENCING],
               label="体験中（人数欄）", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=STATE_COLORS[EXITING],
               label="退出中", markersize=6),
        Line2D([0], [0], marker="o", linestyle="none", color=BLOCKED_COLOR,
               label="移動ブロック中", markersize=6),
    ]
    ax.legend(
        handles=legend_handles,
        title="エージェントの色",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=8,
        title_fontsize=9,
    )
    booth_count_text = ax.text(
        1.02,
        0.60,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        linespacing=1.5,
    )

    def update(i):
        t, snap, booth_stats = frames[i]
        xs = [s[1] for s in snap]
        ys = [s[2] for s in snap]
        colors = [
            BLOCKED_COLOR if s[5] > 0 else STATE_COLORS.get(s[3], "#333333")
            for s in snap
        ]
        scat.set_offsets(np.c_[xs, ys] if xs else np.empty((0, 2)))
        scat.set_color(colors)
        label.set_text(f"t = {t * cfg.dt_seconds:.0f} s  agents = {len(snap)}")
        count_lines = ["ブース別人数", "（体験中 / 待機中）"]
        count_lines.extend(
            f"B{booth_id + 1}: {active:>2} / {queued:>2}"
            for booth_id, active, queued in booth_stats
            if not cfg.booths[booth_id].is_table
        )
        booth_count_text.set_text("\n".join(count_lines))
        return scat, label, booth_count_text

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
