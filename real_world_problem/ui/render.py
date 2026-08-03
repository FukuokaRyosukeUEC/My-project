"""render.py — 選択した解を可視化するラッパ。

描画そのものは既存の `src/visualization.py` に任せ、
ここでは「設計ベクトル → シミュレーション → 画像ファイル」の
段取りとキャッシュ管理だけを行う。src/ には手を入れない。
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.constraints import to_integer_layout  # noqa: E402
from src.models import Config  # noqa: E402
from src.parallel import StaticContext, build_context, evaluate_layout  # noqa: E402
from src.venue import apply_layout  # noqa: E402
from src.visualization import (  # noqa: E402
    draw_venue,
    plot_heatmaps,
    render_animation,
)

#: 可視化用シミュレーションのタイムアウト(秒)。
#: 最適化中は §17.6 の打ち切りが効くが、可視化では途中で切れると
#: 図が意味をなさないため十分に長くとる。
VIZ_TIMEOUT_SECONDS = 3600.0


def viz_context(cfg: Config) -> StaticContext:
    """可視化専用の静的コンテキスト(タイムアウトのみ緩める)。"""
    return build_context(replace(cfg, timeout_seconds=VIZ_TIMEOUT_SECONDS))


def design_key(x) -> str:
    """設計ベクトルからキャッシュ用の短いハッシュを作る。"""
    s = ",".join(str(int(v)) for v in x)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- レイアウト図


def render_layout(cfg: Config, ctx: StaticContext, x, path: Path) -> Path:
    """会場レイアウト図(ブース・受付・退出・待機列)を PNG で保存する。"""
    positions = to_integer_layout(x)
    layout = apply_layout(cfg, ctx.venue, positions)

    fig, ax = plt.subplots(figsize=(9, 9 * cfg.grid_height / max(cfg.grid_width, 1)))
    draw_venue(ax, cfg, layout, legend=True)
    ax.set_title("会場レイアウト")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- ヒートマップ


def render_heatmap(cfg: Config, ctx: StaticContext, x, path: Path) -> tuple[Path, dict]:
    """混雑ヒートマップ(累積滞在 + 閾値超過)を PNG で保存する。

    戻り値は (画像パス, KPI dict)。
    """
    positions = to_integer_layout(x)
    result = evaluate_layout(ctx, positions, collect_heatmap=True)
    layout = apply_layout(cfg, ctx.venue, positions)

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(
        cfg,
        layout,
        result.heatmap,
        result.over_threshold_map,
        path,
        title="選択解",
    )

    kpis = dict(result.kpis)
    kpis["f_move"] = float(result.f_move)
    kpis["f_congestion"] = float(result.f_congestion)
    kpis["timed_out"] = bool(result.timed_out)
    return path, kpis


# ---------------------------------------------------------------- 時系列

def simulate_series(cfg: Config, ctx: StaticContext, x) -> dict:
    """1 解をシミュレーションし、混雑度・移動負担の時系列を返す。

    戻り値:
        {"time_seconds": [...], "congestion": [...], "movement": [...],
         "f_move": float, "f_congestion": float, "timed_out": bool}

    time_seconds はステップ番号に cfg.dt_seconds を掛けた実時間(秒)。
    """
    positions = to_integer_layout(x)
    result = evaluate_layout(ctx, positions)

    steps = list(result.time_steps)
    cong = list(result.congestion_history)
    move = list(result.movement_history)

    # 3 本の長さは揃っているはずだが、途中打ち切り時のずれに備えて短い方に合わせる。
    n = min(len(steps), len(cong), len(move))
    dt = float(cfg.dt_seconds)

    cong = [float(cong[i]) for i in range(n)]

    # congestion_history は「歩行可能セルあたりの重み付き平均」で、
    # f_congestion(= 重み付き総和 / n_steps)とは正規化が違う。
    # 曲線の時間平均が f_congestion に一致するようスケールを揃えておくと、
    # パレート図の縦軸と直接見比べられる。
    total = sum(cong)
    if n and total > 0.0:
        scale = float(result.f_congestion) * n / total
    else:
        scale = 1.0

    return {
        "time_seconds": [float(steps[i]) * dt for i in range(n)],
        "congestion": [c * scale for c in cong],
        "congestion_raw": cong,
        "movement": [float(move[i]) for i in range(n)],
        "f_move": float(result.f_move),
        "f_congestion": float(result.f_congestion),
        "timed_out": bool(result.timed_out),
    }


# ---------------------------------------------------------------- アニメーション


def render_agent_animation(
    cfg: Config,
    ctx: StaticContext,
    x,
    path: Path,
    frame_interval: int = 5,
    fps: int = 8,
) -> Path:
    """エージェントの時間発展アニメーションを GIF で保存する。

    frame_interval を大きくするほどフレーム数が減り、生成が速くなる。
    """
    positions = to_integer_layout(x)
    result = evaluate_layout(
        ctx,
        positions,
        record_frames=True,
        frame_interval=max(1, int(frame_interval)),
    )
    layout = apply_layout(cfg, ctx.venue, positions)

    path.parent.mkdir(parents=True, exist_ok=True)
    render_animation(cfg, layout, result.frames, path, fps=fps)
    return path


# ---------------------------------------------------------------- 会場プレビュー


def render_empty_venue(cfg: Config, path: Path) -> Path:
    """ブースを置く前の会場だけを描く(入力パラメータの確認用)。

    最適化を回さずに「この会場サイズで合っているか」を確かめられるようにする。
    """
    from src.venue import build_base_venue

    venue = build_base_venue(cfg)
    # ブース 0 個の Layout は作れないため、grid をそのまま描く。
    fig, ax = plt.subplots(figsize=(7, 7 * cfg.grid_height / max(cfg.grid_width, 1)))
    ax.imshow(venue.base_grid, cmap="Pastel1", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"会場 {cfg.grid_width} × {cfg.grid_height} セル")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
