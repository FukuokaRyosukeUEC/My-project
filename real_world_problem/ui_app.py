"""イベント会場ブース配置最適化の Streamlit UI。

実行:
    streamlit run ui_app.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pymoo.core.callback import Callback

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constraints import round_layout
from src.models import BoothSpec, Config, load_config
from src.optimization import run_nsga2
from src.venue import apply_layout, build_base_venue
from src.visualization import draw_venue


def build_ui_config(
    base: Config,
    *,
    venue_shape: str,
    width: int,
    height: int,
    booth_count: int,
    booth_width: int,
    booth_height: int,
    visitors: int,
    duration: int,
    generations: int,
    population: int,
) -> Config:
    """画面の入力値から、既存シミュレータ用の Config を作る。"""
    if venue_shape == "矩形":
        areas: tuple = ()
        walkways: tuple = ()
        obstacles: tuple = ()
        grid_width, grid_height = width, height
    else:
        areas = base.venue_areas
        walkways = base.venue_walkways
        obstacles = base.venue_obstacles
        grid_width, grid_height = base.grid_width, base.grid_height

    booths = tuple(
        BoothSpec(
            booth_id=i,
            name=f"ブース{i + 1}",
            width=booth_width,
            height=booth_height,
            category="general",
            attractiveness=0.7,
            capacity=2,
            mean_service_seconds=30.0,
            max_queue_length=min(5, max(1, booth_width + booth_height)),
        )
        for i in range(booth_count)
    )
    return replace(
        base,
        grid_width=grid_width,
        grid_height=grid_height,
        venue_areas=areas,
        venue_walkways=walkways,
        venue_obstacles=obstacles,
        entrance_cells_override=(),
        exit_cells_override=(),
        booths=booths,
        categories=("general",),
        n_visitors=visitors,
        duration_seconds=float(duration),
        arrival_profile=((0.0, float(duration), visitors / float(duration)),),
        population_size=population,
        n_generations=generations,
    )


class GenerationHistory(Callback):
    """UI表示に必要な配列だけを世代ごとに保存する。"""

    def __init__(self, on_generation=None):
        super().__init__()
        self.generations: list[dict] = []
        self.on_generation = on_generation

    def notify(self, algorithm):
        pop = algorithm.pop
        x = np.atleast_2d(pop.get("X")).copy()
        f = np.atleast_2d(pop.get("F")).copy()
        g = np.atleast_2d(pop.get("G")).reshape(len(x), -1)
        feasible = np.all(g <= 0, axis=1)
        self.generations.append(
            {"number": int(algorithm.n_gen), "X": x[feasible], "F": f[feasible]}
        )
        if self.on_generation is not None:
            self.on_generation(int(algorithm.n_gen))


def render_layout(cfg: Config, design_vector: np.ndarray):
    venue = build_base_venue(cfg)
    layout = apply_layout(cfg, venue, round_layout(design_vector))
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_venue(ax, cfg, layout, legend=True)
    ax.set_title("選択した配置案")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="イベント会場 配置最適化", layout="wide")
st.title("イベント会場 ブース配置最適化")
st.caption("会場・ブース・来場者・探索条件を設定し、任意の世代の候補解を確認できます。")

base_config = load_config(ROOT / "config")

with st.sidebar:
    st.header("設定")
    shape = st.selectbox("会場の形", ["矩形", "調布駅前広場（既存形状）"])
    c1, c2 = st.columns(2)
    width = c1.number_input("幅 [m]", 10, 200, 40, disabled=shape != "矩形")
    height = c2.number_input("奥行 [m]", 10, 200, 30, disabled=shape != "矩形")

    st.subheader("ブース")
    booth_count = st.number_input("ブース数", 1, 50, 6)
    booth_shape = st.selectbox("ブースの形", ["長方形", "正方形"])
    c1, c2 = st.columns(2)
    booth_width = c1.number_input("幅 [m]", 1, 20, 2)
    booth_height_input = c2.number_input(
        "奥行 [m]", 1, 20, 3, disabled=booth_shape == "正方形"
    )
    booth_height = booth_width if booth_shape == "正方形" else booth_height_input

    st.subheader("シミュレーション")
    visitors = st.number_input("エージェント数", 1, 10000, 200, step=10)
    duration = st.number_input("シミュレート時間 [秒]", 10, 86400, 400, step=10)

    st.subheader("最適化")
    generations = st.number_input("世代数", 1, 1000, 10)
    population = st.number_input("探索個体数", 4, 1000, 16)
    seed = st.number_input("乱数シード", 0, 999999, 1)
    optimize = st.button("最適化を開始", type="primary", width="stretch")

input_signature = (
    shape,
    int(width),
    int(height),
    booth_shape,
    int(booth_count),
    int(booth_width),
    int(booth_height),
    int(visitors),
    int(duration),
    int(generations),
    int(population),
    int(seed),
)

if optimize:
    st.session_state.pop("optimization", None)
    cfg = build_ui_config(
        base_config,
        venue_shape=shape,
        width=int(width),
        height=int(height),
        booth_count=int(booth_count),
        booth_width=int(booth_width),
        booth_height=int(booth_height),
        visitors=int(visitors),
        duration=int(duration),
        generations=int(generations),
        population=int(population),
    )
    try:
        with st.spinner("最適化を実行しています…"):
            progress = st.progress(0, text="初期集団を生成しています…")
            history = GenerationHistory(
                lambda current: progress.progress(
                    min(current / int(generations), 1.0),
                    text=f"最適化中: 第 {current} / {int(generations)} 世代",
                )
            )
            run_nsga2(
                cfg,
                pop_size=int(population),
                n_gen=int(generations),
                seed=int(seed),
                verbose=False,
                callback=history,
            )
            progress.empty()
        st.session_state["optimization"] = {
            "cfg": cfg,
            "generations": history.generations,
            "input_signature": input_signature,
        }
    except Exception as exc:
        st.error(f"最適化を実行できませんでした: {exc}")

data = st.session_state.get("optimization")
if data and data["input_signature"] != input_signature:
    st.warning("入力条件が変更されています。新しい条件で最適化を実行してください。")
    data = None
if data:
    cfg = data["cfg"]
    histories = data["generations"]
    available = [g for g in histories if len(g["X"])]
    if not available:
        st.warning("表示できる有効解がありません。会場を広げるか、ブース数を減らしてください。")
    else:
        left, right = st.columns([1, 2])
        with left:
            generation_no = st.select_slider(
                "表示する世代",
                options=[g["number"] for g in available],
                value=available[-1]["number"],
            )
            selected = next(g for g in available if g["number"] == generation_no)
            f_min = selected["F"].min(axis=0)
            f_span = np.ptp(selected["F"], axis=0)
            f_span = np.where(f_span > 0, f_span, 1.0)
            balance_score = np.linalg.norm((selected["F"] - f_min) / f_span, axis=1)
            order = np.argsort(balance_score)
            solution_rank = st.selectbox(
                "表示する解",
                range(len(order)),
                format_func=lambda i: (
                    f"候補 {i + 1}（バランス順）— 移動 {selected['F'][order[i], 0]:.3f}, "
                    f"混雑 {selected['F'][order[i], 1]:.3f}"
                ),
            )
            idx = order[solution_rank]
            st.metric("移動目的値", f"{selected['F'][idx, 0]:.3f}")
            st.metric("混雑目的値", f"{selected['F'][idx, 1]:.3f}")
            st.caption(f"第 {generation_no} 世代 / 有効解 {len(order)} 件")
        with right:
            figure = render_layout(cfg, selected["X"][idx])
            st.pyplot(figure, width="stretch")
            plt.close(figure)
else:
    st.info("左側で条件を設定し、「最適化を開始」を押してください。")
