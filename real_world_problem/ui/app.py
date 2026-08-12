"""app.py — 最適化 UI 本体(Streamlit)。

起動:
    cd real_world_problem
    streamlit run ui/app.py

画面は 3 段構成。
  サイドバー : 入力(会場・ブース・エージェント・時間・世代数・個体数)
  上段       : 進捗(2 秒ごとに自動更新)
  下段       : 世代スライダ + パレート図(クリックで解を選択)
               → 選択解のレイアウト図 / ヒートマップ / アニメーション

最適化そのものは ui/optimize_worker.py が別プロセスで回す。
この画面はファイル(runs/<run_id>/)を読むだけなので、
ブラウザを閉じても最適化は止まらない。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import load_config  # noqa: E402
from ui import render, run_store  # noqa: E402

st.set_page_config(
    page_title="ブース配置最適化",
    layout="wide",
)

POLL_SECONDS = 2.0

#: 比較基準(booths.yaml の fixed_position = 実会場で実際に使われた配置)を
#: スナップショット内の解と区別するための擬似インデックス。
BASELINE_INDEX = -1
BASELINE_COLOR = "#2ca02c"


# ================================================================ キャッシュ


@st.cache_resource(show_spinner=False)
def get_config(config_dir: str):
    return load_config(config_dir)


@st.cache_resource(show_spinner=False)
def get_viz_context(config_dir: str):
    """可視化用の静的コンテキスト(会場・来場者テーブル)を実行ごとに 1 回だけ構築。"""
    return render.viz_context(get_config(config_dir))


@st.cache_data(show_spinner=False)
def cached_layout_png(config_dir: str, x: tuple, out_dir: str) -> str:
    cfg = get_config(config_dir)
    ctx = get_viz_context(config_dir)
    path = Path(out_dir) / f"layout_{render.design_key(x)}_v{render.RENDER_VERSION}.png"
    if not path.exists():
        render.render_layout(cfg, ctx, x, path)
    return str(path)


@st.cache_data(show_spinner=False)
def cached_heatmap_png(config_dir: str, x: tuple, out_dir: str) -> tuple[str, dict]:
    cfg = get_config(config_dir)
    ctx = get_viz_context(config_dir)
    path = Path(out_dir) / f"heatmap_{render.design_key(x)}_v{render.RENDER_VERSION}.png"
    _, kpis = render.render_heatmap(cfg, ctx, x, path)
    return str(path), kpis


@st.cache_data(show_spinner=False)
def cached_series(config_dir: str, x: tuple) -> dict:
    """1 解の混雑度・移動負担の時系列(比較グラフ用)。"""
    cfg = get_config(config_dir)
    ctx = get_viz_context(config_dir)
    return render.simulate_series(cfg, ctx, x)


@st.cache_data(show_spinner=False)
def cached_fixed_layout(config_dir: str, run_dir: str) -> dict:
    """booths.yaml の固定配置(実会場で実際に使われた配置)を評価する。

    通常はワーカーが最適化の直前に評価して `fixed_layout.json` へ
    書き出しているので、ここが呼ばれるのはこの機能より前に作られた
    実行を開いたときだけ。
    """
    from ui.optimize_worker import evaluate_fixed_layout

    return evaluate_fixed_layout(get_config(config_dir), Path(run_dir))


def baseline_solution(fixed: dict) -> dict | None:
    """比較基準を、パレート図の点と同じ形の dict にする。

    `_index = -1` を目印にして、スナップショット内の解と区別する。
    """
    if not fixed.get("available"):
        return None
    return {
        "_index": BASELINE_INDEX,
        "x": [int(v) for v in fixed["x"]],
        "f_move": float(fixed["f_move"]),
        "f_congestion": float(fixed["f_congestion"]),
        "rank": BASELINE_INDEX,
        "label": fixed.get("label", "実会場配置(booths.yaml)"),
        "feasible": bool(fixed.get("feasible", True)),
    }


@st.cache_data(show_spinner=False)
def cached_animation_gif(
    config_dir: str,
    x: tuple,
    out_dir: str,
    frame_interval: int,
    fps: int,
) -> str:
    cfg = get_config(config_dir)
    ctx = get_viz_context(config_dir)
    path = (
        Path(out_dir)
        / f"anim_{render.design_key(x)}_i{frame_interval}_f{fps}_v{render.RENDER_VERSION}.gif"
    )
    if not path.exists():
        render.render_agent_animation(
            cfg, ctx, x, path, frame_interval=frame_interval, fps=fps
        )
    return str(path)


# ================================================================ サイドバー(入力)


def sidebar_inputs() -> run_store.RunParams:
    st.sidebar.header("入力パラメータ")

    venue_mode = st.sidebar.radio(
        "会場モード",
        options=["rect", "real"],
        format_func=lambda v: {
            "rect": "矩形会場(サイズを自由に指定)",
            "real": "実会場(config/ のポリゴン形状)",
        }[v],
        help=(
            "実会場モードでは会場形状が多角形で定義されているため、"
            "会場サイズは編集できません(config/default_config.yaml の "
            "venue_shape をそのまま使います)。"
        ),
    )

    preset_main, preset_booths, _ = run_store.load_preset_yaml(venue_mode)

    # --- 会場
    with st.sidebar.expander("会場", expanded=True):
        if venue_mode == "rect":
            grid_width = st.number_input(
                "会場の幅(セル)", min_value=5, max_value=200,
                value=int(preset_main["grid"]["width"]), step=1,
            )
            grid_height = st.number_input(
                "会場の高さ(セル)", min_value=5, max_value=200,
                value=int(preset_main["grid"]["height"]), step=1,
            )
            entrance_side = st.selectbox(
                "入口の辺", ["bottom", "top", "left", "right"], index=0
            )
            exit_side = st.selectbox(
                "出口の辺", ["top", "bottom", "left", "right"], index=0
            )
            span = st.number_input(
                "入退場口の幅(セル)",
                min_value=4,
                max_value=20,
                value=max(4, int(preset_main["entrance_exit"].get("span_cells", 4))),
                step=1,
                help="エージェントは4セル以上の幅がある場所だけを通行します。",
            )
        else:
            grid_width = int(preset_main["grid"]["width"])
            grid_height = int(preset_main["grid"]["height"])
            entrance_side, exit_side, span = "bottom", "top", 4
            st.caption(
                f"会場サイズ: {grid_width} × {grid_height} セル(実会場ポリゴン、編集不可)"
            )

        cell_size = st.number_input(
            "1 セルの大きさ(m)", min_value=0.25, max_value=5.0,
            value=float(preset_main["grid"]["cell_size_meters"]), step=0.25,
        )

    # --- ブース
    st.sidebar.subheader("ブース")
    st.sidebar.caption("行の追加・削除でブース数を変えられます。")

    booth_df = pd.DataFrame(
        [
            {
                "name": b.get("name", f"ブース{i + 1}"),
                "width": int(b["width"]),
                "height": int(b["height"]),
                "category": b.get("category", "none"),
                "attractiveness": float(b.get("attractiveness", 0.5)),
                "capacity": int(b.get("capacity", 1)),
                "mean_service_seconds": float(b.get("mean_service_seconds", 30)),
                "max_queue_length": int(b.get("max_queue_length", 5)),
                "kind": b.get("kind", "booth"),
                "theme": b.get("theme", "normal"),
                "area": b.get("area", ""),
                # 既存配置は編集対象にせず、非表示内部列として保持する。
                run_store.EDITOR_FIXED_POSITION_FIELD: (
                    ",".join(str(int(v)) for v in b["fixed_position"])
                    if b.get("fixed_position")
                    else ""
                ),
            }
            for i, b in enumerate(preset_booths)
        ]
    )

    edited = st.sidebar.data_editor(
        booth_df,
        num_rows="dynamic",
        width="stretch",
        height=260,
        key=f"booths_{venue_mode}",
        column_order=[
            column
            for column in booth_df.columns
            if column != run_store.EDITOR_FIXED_POSITION_FIELD
        ],
        column_config={
            "width": st.column_config.NumberColumn("幅", min_value=1, step=1),
            "height": st.column_config.NumberColumn("奥行", min_value=1, step=1),
            "attractiveness": st.column_config.NumberColumn(
                "魅力度", min_value=0.0, max_value=1.0, step=0.05
            ),
            "kind": st.column_config.SelectboxColumn(
                "種別",
                options=[
                    "booth",
                    "kitchen_car",
                    "table",
                    "waste_station",
                    "garbage_station",
                    "trash_station",
                    "recycling_station",
                ],
                help="ゴミステーションは waste_station を選択します。",
            ),
        },
    )
    st.sidebar.caption(f"ブース数: {len(edited)}")

    # --- 来場者・時間
    with st.sidebar.expander("来場者・シミュレーション", expanded=True):
        n_visitors = st.number_input(
            "エージェント数", min_value=1, max_value=100000,
            value=int(preset_main["simulation"]["n_visitors"]), step=10,
        )
        duration = st.number_input(
            "シミュレート時間(秒)", min_value=10, max_value=100000,
            value=int(preset_main["time"]["duration_seconds"]), step=10,
        )
        dt = st.number_input(
            "時間刻み dt(秒)", min_value=0.1, max_value=10.0,
            value=float(preset_main["time"]["dt_seconds"]), step=0.1,
        )
        seed = st.number_input("評価シード", min_value=0, value=0, step=1)
        timeout = st.number_input(
            "1 評価のタイムアウト(秒)", min_value=1.0, max_value=3600.0,
            value=30.0, step=5.0,
            help="1 配置のシミュレーションがこの時間を超えると打ち切り、ペナルティ値になります。",
        )
        waste_cfg = preset_main.get("waste_disposal") or {}
        general_waste_probability = st.number_input(
            "通常時のゴミステーション利用確率",
            min_value=0.0,
            max_value=0.99,
            value=float(waste_cfg.get("general_probability", 0.05)),
            step=0.01,
            format="%.2f",
        )
        food_waste_probability = st.number_input(
            "飲食後のゴミステーション利用確率",
            min_value=0.0,
            max_value=0.99,
            value=float(waste_cfg.get("after_food_probability", 0.85)),
            step=0.01,
            format="%.2f",
            help="100%には設定できません。テーブル利用後もこの確率で立ち寄ります。",
        )

    # --- 最適化
    with st.sidebar.expander("最適化(直接配置NSGA-II)", expanded=True):
        st.caption(
            "通常ブース=(x, y, 方向)、テーブル=(x, y)の直接表現で探索します。"
            "初期集団は固定配置や整列テンプレートを使わず、完全ランダムに生成します。"
        )
        pop = st.number_input(
            "探索個体数", min_value=4, max_value=1000,
            value=16, step=4,
        )
        gen = st.number_input(
            "世代数", min_value=1, max_value=2000,
            value=20, step=5,
        )
        interval = st.number_input(
            "スナップショット間隔(世代)", min_value=1, max_value=100,
            value=10, step=1,
            help="この世代数ごとに集団を保存し、GUI で表示できるようにします。",
        )
        workers = st.number_input(
            "並列プロセス数", min_value=1, max_value=64, value=4, step=1
        )
        optimization_seed = st.number_input(
            "最適化シード", min_value=0, value=1, step=1,
            help="交叉・突然変異と完全ランダム初期集団の再現に使います。",
        )
        initial_max_trials = st.number_input(
            "初期個体の最大試行回数", min_value=1, max_value=100000,
            value=5000, step=500,
            help="物理制約を満たすランダム配置が作れない場合は増やしてください。",
        )
        save_artifacts = st.checkbox(
            "終了後に成果物(CSV・代表解・方式情報)を書き出す", value=True
        )

    return run_store.RunParams(
        venue_mode=venue_mode,
        grid_width=int(grid_width),
        grid_height=int(grid_height),
        cell_size_meters=float(cell_size),
        entrance_side=entrance_side,
        exit_side=exit_side,
        entrance_exit_span=int(span),
        booths=run_store.restore_editor_fixed_positions(
            edited.to_dict(orient="records"), venue_mode
        ),
        n_visitors=int(n_visitors),
        duration_seconds=float(duration),
        dt_seconds=float(dt),
        evaluation_seed=int(seed),
        timeout_seconds=float(timeout),
        waste_station_general_probability=float(general_waste_probability),
        waste_station_after_food_probability=float(food_waste_probability),
        population_size=int(pop),
        n_generations=int(gen),
        workers=int(workers),
        snapshot_interval=int(interval),
        optimization_seed=int(optimization_seed),
        initial_max_trials=int(initial_max_trials),
        save_artifacts=bool(save_artifacts),
    )


# ================================================================ 起動


def start_run(params: run_store.RunParams) -> str:
    """config を書き出してワーカーを起動し、run_id を返す。"""
    params.validate()
    run_id = run_store.new_run_id()
    rd = run_store.run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)

    run_store._write_json_atomic(rd / "params.json", params.to_dict())
    run_store.build_config_dir(params, rd / "config")
    run_store.write_status(rd, state="starting", message="ワーカーを起動中", generation=0)

    log = open(rd / "worker.log", "w", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, "-m", "ui.optimize_worker", str(rd)],
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return run_id


# ================================================================ 進捗


@st.fragment(run_every=POLL_SECONDS)
def progress_panel(run_id: str) -> None:
    rd = run_store.run_dir(run_id)
    status = run_store.read_status(rd)
    state = status.get("state", "unknown")
    gen = int(status.get("generation", 0))
    total = max(1, int(status.get("n_generations", 1)))

    label = {
        "starting": "起動中",
        "baseline": "実会場配置を評価中",
        "initializing": "完全ランダム初期集団を生成中",
        "running": "最適化中",
        "finalizing": "成果物を出力中",
        "done": "完了",
        "error": "エラー",
    }.get(state, state)

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.progress(min(1.0, gen / total), text=f"{label}: {gen} / {total} 世代")
    c2.metric("経過時間", f"{status.get('elapsed_seconds', 0.0):.0f} 秒")
    c3.metric("保存済み世代", len(run_store.list_snapshot_generations(rd)))

    if state == "error":
        st.error(status.get("message", "エラー"))
        with st.expander("トレースバック"):
            st.code(status.get("traceback", ""))
    elif state == "done":
        st.success(
            f"完了(パレート解 {status.get('n_pareto', '?')} 件)。"
            "下のスライダで世代を選び、パレート図の点をクリックしてください。"
        )


# ================================================================ パレート図


def pareto_figure(
    snapshot: dict,
    slots: list[dict] | None = None,
    baseline: dict | None = None,
) -> go.Figure:
    rows = snapshot["solutions"]
    df = pd.DataFrame(rows)

    fig = go.Figure()
    if df.empty and baseline is None:
        fig.add_annotation(text="実行可能解がありません", showarrow=False)
        return fig

    # A/B は「点の上に別トレースを重ねる」のではなく、
    # 既存トレースの点ごとの枠線・ラベルとして表現する。
    # 重ねたトレースはクリックを横取りして選択を壊すため。
    tag_of = {
        int(s["_index"]): tag for s, tag in zip(slots or [], ("A", "B"))
    }

    def decorate(index) -> tuple[list, list, list]:
        """点ごとの (枠線色, 枠線幅, ラベル文字) を返す。"""
        colors, widths, texts = [], [], []
        for i in index:
            tag = tag_of.get(int(i))
            if tag is None:
                colors.append("rgba(0,0,0,0)")
                widths.append(0)
                texts.append("")
            else:
                colors.append(SERIES_COLORS[0] if tag == "A" else SERIES_COLORS[1])
                widths.append(4)
                texts.append(f"<b>{tag}</b>")
        return colors, widths, texts

    # --- 比較基準(実会場配置)を補助線 + 星マーカーで重ねる。
    # 補助線を先に描いて最背面に置き、クリック判定は星マーカーだけが持つ。
    if baseline is not None:
        bx, by = baseline["f_move"], baseline["f_congestion"]

        fig.add_hline(
            y=by, line=dict(color=BASELINE_COLOR, width=1, dash="dash"),
            opacity=0.45,
        )
        fig.add_vline(
            x=bx, line=dict(color=BASELINE_COLOR, width=1, dash="dash"),
            opacity=0.45,
        )

        tag = "" if baseline["_index"] not in tag_of else f"<b>{tag_of[baseline['_index']]}</b>"
        fig.add_trace(
            go.Scatter(
                x=[bx], y=[by],
                mode="markers+text",
                name=baseline["label"],
                marker=dict(
                    size=20, color=BASELINE_COLOR, symbol="star",
                    line=dict(
                        color=(
                            SERIES_COLORS[0]
                            if tag_of.get(baseline["_index"]) == "A"
                            else SERIES_COLORS[1]
                        ) if baseline["_index"] in tag_of else "#ffffff",
                        width=4 if baseline["_index"] in tag_of else 1,
                    ),
                ),
                text=[tag],
                textposition="bottom center",
                textfont=dict(size=14),
                customdata=[BASELINE_INDEX],
                hovertemplate=(
                    f"{baseline['label']}<br>"
                    "f_move=%{x:.3f}<br>f_congestion=%{y:.4g}<extra></extra>"
                ),
            )
        )

    if df.empty:
        fig.update_layout(
            xaxis_title="f_move(移動負担: 自由歩行換算時間 [秒])",
            yaxis_title="f_congestion(混雑度)",
            height=460,
            margin=dict(l=10, r=10, t=40, b=10),
            title="この世代には実行可能解がありません(比較基準のみ表示)",
        )
        return fig

    front = df[df["rank"] == 0]
    rest = df[df["rank"] != 0]

    if not rest.empty:
        c, w, t = decorate(rest.index)
        fig.add_trace(
            go.Scatter(
                x=rest["f_move"], y=rest["f_congestion"],
                mode="markers+text", name="被支配解",
                marker=dict(size=9, color="#c8c8c8", line=dict(color=c, width=w)),
                text=t, textposition="top center", textfont=dict(size=14),
                customdata=rest.index,
            )
        )

    c, w, t = decorate(front.index)
    fig.add_trace(
        go.Scatter(
            x=front["f_move"], y=front["f_congestion"],
            mode="markers+lines+text", name="非支配解(パレートフロント)",
            marker=dict(size=12, color="#1f77b4", line=dict(color=c, width=w)),
            line=dict(color="#1f77b4", width=1, dash="dot"),
            text=t, textposition="top center", textfont=dict(size=14),
            customdata=front.index,
        )
    )

    fig.update_layout(
        xaxis_title="f_move(移動負担: 自由歩行換算時間 [秒])",
        yaxis_title="f_congestion(混雑度)",
        height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"第 {snapshot['generation']} 世代の集団({len(df)} 解)",
        clickmode="event+select",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        # dragmode は既定の "zoom" のまま。"select" にすると単純クリックが
        # 空の矩形選択として扱われ、点のクリック選択が効かなくなる。
    )
    return fig


def _selection_points(event) -> list:
    """選択状態から points を取り出す(dict でも属性アクセス型でも動くように)。"""
    if event is None:
        return []
    sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if sel is None:
        return []
    points = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    return list(points or [])


def resolve_selection(
    snapshot: dict,
    event,
    baseline: dict | None = None,
) -> dict | None:
    """Plotly のクリックイベントから解 1 件を取り出す。

    customdata が BASELINE_INDEX(-1)なら比較基準(実会場配置)を返す。
    """
    points = _selection_points(event)
    if not points:
        return None

    p = points[-1]
    idx = p.get("customdata")
    if isinstance(idx, (list, tuple)):
        idx = idx[0] if idx else None
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return None

    if i == BASELINE_INDEX:
        return dict(baseline) if baseline is not None else None

    try:
        return dict(snapshot["solutions"][i], _index=i)
    except IndexError:
        return None


# ================================================================ A/B スロット
#
# Streamlit の plotly 選択はクリックのたびに選択が「置き換わる」ので、
# 修飾キーによる多点選択はできない。
# そこでクリックを 1 件ずつ受け取り、A → B の順に 2 スロットへ溜める。


def is_baseline(sol: dict) -> bool:
    return int(sol.get("_index", 0)) == BASELINE_INDEX


def solution_name(sol: dict) -> str:
    """解の呼び名。比較基準だけ特別扱いする。"""
    return sol.get("label", "実会場配置") if is_baseline(sol) else f"解 #{sol['_index']}"


def slot_label(sol: dict) -> str:
    tail = (
        "比較基準" if is_baseline(sol) else f"rank {sol['rank']}"
    )
    return (
        f"{solution_name(sol)}  f_move={sol['f_move']:.1f} / "
        f"f_cong={sol['f_congestion']:.4g} / {tail}"
    )


def update_slots(
    gen: int,
    event,
    snapshot: dict,
    baseline: dict | None = None,
) -> list[dict]:
    """クリックを A/B スロットに反映して、現在のスロットを返す。"""
    # 世代を切り替えたら解の並びが変わるのでスロットを捨てる。
    # ただし比較基準は世代に依存しないので残す。
    if st.session_state.get("slot_gen") != gen:
        st.session_state["slot_gen"] = gen
        st.session_state["slots"] = [
            s for s in st.session_state.get("slots", []) if is_baseline(s)
        ]
        st.session_state["slot_sig"] = None

    picked = resolve_selection(snapshot, event, baseline)
    sig = None if picked is None else picked["_index"]

    # 他ウィジェット操作による再実行では選択内容が変わらないので、
    # 同じクリックを二重に処理しないよう最後に処理した点を覚えておく。
    if picked is not None and st.session_state.get("slot_sig") != sig:
        st.session_state["slot_sig"] = sig
        slots = list(st.session_state.get("slots", []))

        if any(s["_index"] == picked["_index"] for s in slots):
            pass  # すでに A か B に入っている点 → 何もしない
        elif len(slots) < 2:
            slots.append(picked)
        else:
            slots = [slots[1], picked]  # 古い方(A)を押し出す

        st.session_state["slots"] = slots

    return list(st.session_state.get("slots", []))


def clear_slots() -> None:
    """スロットを空にし、グラフ側の選択状態もリセットする。

    plotly ウィジェットの選択は key ごとに保持されるので、
    key を変えて作り直さないと、直前のクリックが再び拾われてしまう。
    """
    st.session_state["slots"] = []
    st.session_state["slot_sig"] = None
    st.session_state["pareto_nonce"] = st.session_state.get("pareto_nonce", 0) + 1


def slot_controls(slots: list[dict]) -> None:
    """A/B スロットの状態表示とクリア。"""
    tag_a = slot_label(slots[0]) if len(slots) >= 1 else "未選択(点をクリック)"
    tag_b = slot_label(slots[1]) if len(slots) >= 2 else "未選択(もう 1 点クリック)"

    c1, c2 = st.columns([3, 1])
    c1.markdown(
        f"<span style='color:{SERIES_COLORS[0]}'><b>■ A</b></span> {tag_a}<br>"
        f"<span style='color:{SERIES_COLORS[1]}'><b>■ B</b></span> {tag_b}",
        unsafe_allow_html=True,
    )
    c2.button("選択をクリア", width="stretch", on_click=clear_slots)


# ================================================================ 2 解の比較


SERIES_COLORS = ("#1f77b4", "#d62728")


def _series_figure(
    labels: list[str],
    series: list[dict],
    key: str,
    y_title: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for label, s, color in zip(labels, series, SERIES_COLORS):
        fig.add_trace(
            go.Scatter(
                x=s["time_seconds"],
                y=s[key],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        xaxis_title="経過時間 [秒]",
        yaxis_title=y_title,
        title=title,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def comparison_panel(run_id: str, snapshot: dict, picked: list[dict]) -> None:
    """選択した 2 解の混雑度・移動負担の推移とレイアウトを並べて比較する。"""
    rd = run_store.run_dir(run_id)
    config_dir = str(rd / "config")
    cache_dir = str(rd / "ui_cache")

    a, b = picked[0], picked[1]
    labels = [
        f"A: {solution_name(s)}"
        f"(f_move={s['f_move']:.1f} / f_cong={s['f_congestion']:.4g})"
        if i == 0
        else f"B: {solution_name(s)}"
        f"(f_move={s['f_move']:.1f} / f_cong={s['f_congestion']:.4g})"
        for i, s in enumerate((a, b))
    ]
    xs = [tuple(int(v) for v in s["x"]) for s in (a, b)]

    def rate(new: float, old: float) -> str:
        """old を基準にした改善率(小さいほど良い目的関数)。"""
        if old == 0:
            return "—"
        return f"{(old - new) / old * 100:+.1f} %"

    # --- 指標の横並び
    df_kpi = pd.DataFrame(
        {
            f"A({solution_name(a)})": [
                f"{a['f_move']:.3f}",
                f"{a['f_congestion']:.4g}",
                "比較基準" if is_baseline(a) else f"rank {a['rank']}",
            ],
            f"B({solution_name(b)})": [
                f"{b['f_move']:.3f}",
                f"{b['f_congestion']:.4g}",
                "比較基準" if is_baseline(b) else f"rank {b['rank']}",
            ],
            "差(B - A)": [
                f"{b['f_move'] - a['f_move']:+.3f}",
                f"{b['f_congestion'] - a['f_congestion']:+.4g}",
                "",
            ],
            "A 比の改善(B)": [
                rate(b["f_move"], a["f_move"]),
                rate(b["f_congestion"], a["f_congestion"]),
                "",
            ],
        },
        index=["f_move(移動負担)", "f_congestion(混雑度)", "順位"],
    )
    st.dataframe(df_kpi, width="stretch")

    if is_baseline(a) or is_baseline(b):
        st.caption(
            "★ が実会場で実際に使われた配置(booths.yaml の fixed_position)です。"
            "「A 比の改善」が + なら B のほうが良い(目的関数は小さいほど良い)。"
        )

    st.caption(
        "推移グラフは 2 解をそれぞれ 1 回シミュレーションして描画します"
        "(初回のみ時間がかかり、以後はキャッシュされます)。"
    )
    if not st.button(
        "推移を比較する",
        type="primary",
        key=f"cmp_{render.design_key(xs[0])}_{render.design_key(xs[1])}",
    ):
        return

    with st.spinner("2 解をシミュレーション中..."):
        series = [cached_series(config_dir, x) for x in xs]

    if any(s["timed_out"] for s in series):
        st.warning("いずれかの解でシミュレーションが打ち切られました(以降は 0 埋め)。")

    st.plotly_chart(
        _series_figure(
            labels, series, "congestion", "瞬間混雑度",
            "混雑度の推移(曲線の時間平均が f_congestion に一致)",
        ),
        width="stretch",
        key=f"cmp_cong_{run_id}_{snapshot['generation']}",
    )
    st.plotly_chart(
        _series_figure(
            labels, series, "movement", "移動負担(累積の自由歩行換算 [秒])",
            "移動負担の推移(終端値が f_move に一致)",
        ),
        width="stretch",
        key=f"cmp_move_{run_id}_{snapshot['generation']}",
    )

    st.markdown("**会場レイアウトの比較**")
    cols = st.columns(2)
    for col, label, x in zip(cols, labels, xs):
        with col:
            st.caption(label)
            st.image(cached_layout_png(config_dir, x, cache_dir))


# ================================================================ 比較基準


def _dominance_summary(snapshot: dict, baseline: dict) -> tuple[int, int, dict | None]:
    """比較基準に対する優劣を数える。

    戻り値は (両目的とも上回る解の数, 実行可能解の総数, 最も改善幅の
    大きい非支配解)。「上回る」は 2 目的とも厳密に小さいこと(支配)。
    """
    sols = snapshot["solutions"]
    bm, bc = baseline["f_move"], baseline["f_congestion"]

    dominating = [
        s for s in sols
        if s["f_move"] < bm and s["f_congestion"] < bc
    ]

    best = None
    if dominating:
        # 2 目的の改善率の和が最大のものを代表として出す。
        best = max(
            dominating,
            key=lambda s: (
                (bm - s["f_move"]) / bm if bm else 0.0
            ) + (
                (bc - s["f_congestion"]) / bc if bc else 0.0
            ),
        )
        best = dict(best, _index=sols.index(best))

    return len(dominating), len(sols), best


def baseline_panel(run_id: str, snapshot: dict) -> dict | None:
    """比較基準の読み込み・表示を行い、パレート図に渡す dict を返す。"""
    rd = run_store.run_dir(run_id)
    fixed = run_store.read_fixed_layout(rd)

    if not fixed.get("available"):
        params = run_store._read_json(rd / "params.json") or {}
        if params.get("venue_mode") != "real":
            # 矩形会場には fixed_position が無い(座標が実会場前提のため
            # build_config_dir で落としている)。比較基準は作れない。
            st.caption(
                "矩形会場モードのため、実会場配置(booths.yaml の "
                "fixed_position)との比較はありません。"
            )
            return None

        if not (rd / "fixed_layout.json").exists():
            # この機能より前に作られた実行。ボタンで後から評価できるようにする。
            c1, c2 = st.columns([3, 1])
            c1.caption(
                "この実行には実会場配置との比較基準がありません"
                "(この機能より前に開始された実行です)。"
            )
            if c2.button("実会場配置を評価", width="stretch", key=f"evalfix_{run_id}"):
                with st.spinner("実会場配置をシミュレーション中..."):
                    cached_fixed_layout(str(rd / "config"), str(rd))
                st.rerun()
            return None

        st.caption(f"実会場配置との比較は無効です — {fixed.get('reason', '不明')}")
        return None

    baseline = baseline_solution(fixed)
    n_dom, n_total, best = _dominance_summary(snapshot, baseline)

    c1, c2, c3 = st.columns([2, 1, 2])
    c1.markdown(
        f"<span style='color:{BASELINE_COLOR}'><b>★ {baseline['label']}</b></span>  "
        f"f_move={baseline['f_move']:.3f} / "
        f"f_congestion={baseline['f_congestion']:.4g}",
        unsafe_allow_html=True,
    )
    c2.metric(
        "実会場を両目的で上回る解",
        f"{n_dom} / {n_total}",
        help="f_move と f_congestion の両方が実会場配置より小さい解の数。",
    )
    if best is not None:
        dm = (baseline["f_move"] - best["f_move"]) / baseline["f_move"] * 100
        dc = (
            (baseline["f_congestion"] - best["f_congestion"])
            / baseline["f_congestion"] * 100
        )
        c3.markdown(
            f"最大改善: **f_move −{dm:.1f}%** / **f_congestion −{dc:.1f}%**  \n"
            f"<small>解 #{best['_index']}(rank {best['rank']})</small>",
            unsafe_allow_html=True,
        )
    else:
        c3.markdown(
            "<small>この世代には、実会場配置を両目的で上回る解はまだありません。</small>",
            unsafe_allow_html=True,
        )

    if not baseline["feasible"]:
        st.caption(
            "※ 実会場配置は現在の設定では配置制約を満たしていません(参考値として表示)。"
        )

    return baseline


# ================================================================ 解の可視化


def solution_panel(
    run_id: str,
    snapshot: dict,
    solution: dict,
    baseline: dict | None = None,
) -> None:
    rd = run_store.run_dir(run_id)
    config_dir = str(rd / "config")
    cache_dir = str(rd / "ui_cache")
    x = tuple(int(v) for v in solution["x"])

    # 比較基準があれば、実会場配置に対する増減を metric の delta に出す。
    # 目的関数は小さいほど良いので delta_color は "inverse"。
    show_delta = baseline is not None and not is_baseline(solution)

    def delta(key: str) -> str | None:
        if not show_delta or baseline[key] == 0:
            return None
        diff = solution[key] - baseline[key]
        return f"{diff:+.3g}({diff / baseline[key] * 100:+.1f}% 実会場比)"

    m1, m2, m3 = st.columns(3)
    m1.metric("f_move", f"{solution['f_move']:.3f}",
              delta=delta("f_move"), delta_color="inverse")
    m2.metric("f_congestion", f"{solution['f_congestion']:.4g}",
              delta=delta("f_congestion"), delta_color="inverse")
    if is_baseline(solution):
        m3.metric("種別", "実会場配置")
    else:
        m3.metric("世代 / 順位", f"{snapshot['generation']} / rank {solution['rank']}")

    if is_baseline(solution) and not solution.get("feasible", True):
        st.warning(
            "この配置は現在の設定では配置制約を満たしていません(参考値)。"
        )

    tab_layout, tab_heat, tab_anim = st.tabs(
        ["会場レイアウト図", "混雑ヒートマップ", "エージェント動画"]
    )

    with tab_layout:
        with st.spinner("レイアウトを描画中..."):
            st.image(cached_layout_png(config_dir, x, cache_dir))

    with tab_heat:
        st.caption(
            "この解を 1 回シミュレーションして、累積滞在人数と混雑閾値の"
            "超過時間を描画します。"
        )
        if st.button("ヒートマップを生成", key=f"heat_{render.design_key(x)}"):
            with st.spinner("シミュレーション中..."):
                png, kpis = cached_heatmap_png(config_dir, x, cache_dir)
            st.image(png)
            st.dataframe(
                pd.DataFrame([kpis]).T.rename(columns={0: "値"}),
                width="stretch",
            )

    with tab_anim:
        c1, c2 = st.columns(2)
        frame_interval = c1.slider(
            "フレーム間隔(ステップ)", 1, 20, 5,
            help="大きいほどフレーム数が減り、生成が速くなります。",
        )
        fps = c2.slider("再生 fps", 2, 20, 8)
        st.caption("シミュレーション + GIF 生成のため、数十秒かかることがあります。")
        if st.button("アニメーションを生成", key=f"anim_{render.design_key(x)}"):
            with st.spinner("シミュレーションと GIF 生成中..."):
                gif = cached_animation_gif(
                    config_dir, x, cache_dir, int(frame_interval), int(fps)
                )
            st.image(gif)


# ================================================================ 本体


def main() -> None:
    st.title("イベント会場ブース配置最適化")

    params = sidebar_inputs()

    st.sidebar.divider()
    if st.sidebar.button("最適化を開始", type="primary", width="stretch"):
        try:
            st.session_state["run_id"] = start_run(params)
            st.session_state.pop("selected", None)
            st.session_state.pop("slots", None)
            st.session_state.pop("slot_sig", None)
        except Exception as exc:  # 設定の不備はここで拾う
            st.sidebar.error(f"起動に失敗しました: {exc}")

    # --- 実行の選択(過去の実行も開ける)
    runs = run_store.list_runs()
    if not runs:
        st.info("左のサイドバーでパラメータを設定し、「最適化を開始」を押してください。")
        return

    default_idx = 0
    if st.session_state.get("run_id") in runs:
        default_idx = runs.index(st.session_state["run_id"])
    run_id = st.selectbox("実行", runs, index=default_idx)
    rd = run_store.run_dir(run_id)

    # 実行を切り替えたら選択解を捨てる。
    # 実行ごとにブース数(=設計変数の次元)が違いうるため。
    if st.session_state.get("viewing_run") != run_id:
        st.session_state["viewing_run"] = run_id
        st.session_state.pop("selected", None)
        st.session_state.pop("slots", None)
        st.session_state.pop("slot_sig", None)
        st.session_state.pop("slot_gen", None)

    progress_panel(run_id)
    st.divider()

    gens = run_store.list_snapshot_generations(rd)
    if not gens:
        st.info("最初のスナップショットを待っています…(進捗は上のバーで更新されます)")
        return

    # --- 世代選択
    c1, c2 = st.columns([4, 1])
    with c1:
        gen = st.select_slider("表示する世代", options=gens, value=gens[-1])
    with c2:
        st.write("")
        if st.button("最新世代へ", width="stretch"):
            st.rerun()

    snapshot = run_store.read_snapshot(rd, gen)
    if snapshot is None:
        st.warning("スナップショットを読み込めませんでした。")
        return

    # --- 比較基準(実会場で実際に使われた配置)
    baseline = baseline_panel(run_id, snapshot)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("パレート図(点をクリックして解を選択)")
        st.caption(
            "点を続けて 2 つクリックすると A・B に入り、下で推移を比較できます。"
            "3 つ目をクリックすると古い方(A)が押し出されます。"
            "★(実会場配置)もクリックして比較対象にできます。"
            "やり直すときは「選択をクリア」。"
        )
        nonce = st.session_state.get("pareto_nonce", 0)
        chart_key = f"pareto_{run_id}_{gen}_{nonce}"

        # 図を描く前に選択状態を反映しておく。描画後のイベントだけを見ると
        # A/B の枠線が 1 クリック遅れて表示されるため。
        slots = update_slots(
            gen, st.session_state.get(chart_key), snapshot, baseline
        )

        event = st.plotly_chart(
            pareto_figure(snapshot, slots, baseline),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=chart_key,
        )

        # session_state 経由で拾えなかった場合の保険。
        after = update_slots(gen, event, snapshot, baseline)
        if [s["_index"] for s in after] != [s["_index"] for s in slots]:
            st.rerun()
        slots = after

        slot_controls(slots)

    with right:
        st.subheader("選択した解")
        selected = slots[-1] if slots else None
        if selected is None:
            solutions = snapshot["solutions"]
            if solutions:
                st.caption(
                    "未選択のため、この世代の f_move 最小解を表示しています。"
                )
                selected = dict(solutions[0], _index=0)
        if selected is None:
            st.info("この世代には実行可能解がありません。")
        else:
            solution_panel(run_id, snapshot, selected, baseline)

    # --- 2 解の比較
    st.divider()
    st.subheader("2 解の比較")
    if len(slots) >= 2:
        comparison_panel(run_id, snapshot, slots)
    else:
        st.info(
            "パレート図の点を 2 つクリックする(または矩形選択で 2 点選ぶ)と、"
            "混雑度と移動負担の推移をここで重ねて比較できます。"
        )


# streamlit run は __name__ == "__main__" でこのスクリプトを実行する。
main()