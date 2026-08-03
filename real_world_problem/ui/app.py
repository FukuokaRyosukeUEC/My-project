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
    path = Path(out_dir) / f"layout_{render.design_key(x)}.png"
    if not path.exists():
        render.render_layout(cfg, ctx, x, path)
    return str(path)


@st.cache_data(show_spinner=False)
def cached_heatmap_png(config_dir: str, x: tuple, out_dir: str) -> tuple[str, dict]:
    cfg = get_config(config_dir)
    ctx = get_viz_context(config_dir)
    path = Path(out_dir) / f"heatmap_{render.design_key(x)}.png"
    _, kpis = render.render_heatmap(cfg, ctx, x, path)
    return str(path), kpis


@st.cache_data(show_spinner=False)
def cached_series(config_dir: str, x: tuple) -> dict:
    """1 解の混雑度・移動負担の時系列(比較グラフ用)。"""
    cfg = get_config(config_dir)
    ctx = get_viz_context(config_dir)
    return render.simulate_series(cfg, ctx, x)


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
        / f"anim_{render.design_key(x)}_i{frame_interval}_f{fps}.gif"
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
                "入退場口の幅(セル)", min_value=1, max_value=20, value=2, step=1
            )
        else:
            grid_width = int(preset_main["grid"]["width"])
            grid_height = int(preset_main["grid"]["height"])
            entrance_side, exit_side, span = "bottom", "top", 2
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
        column_config={
            "width": st.column_config.NumberColumn("幅", min_value=1, step=1),
            "height": st.column_config.NumberColumn("奥行", min_value=1, step=1),
            "attractiveness": st.column_config.NumberColumn(
                "魅力度", min_value=0.0, max_value=1.0, step=0.05
            ),
            "kind": st.column_config.SelectboxColumn(
                "種別", options=["booth", "kitchen_car", "table"]
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

    # --- 最適化
    with st.sidebar.expander("最適化(NSGA-II)", expanded=True):
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
        save_artifacts = st.checkbox(
            "終了後に成果物(CSV・代表解 JSON)を書き出す", value=True
        )

    return run_store.RunParams(
        venue_mode=venue_mode,
        grid_width=int(grid_width),
        grid_height=int(grid_height),
        cell_size_meters=float(cell_size),
        entrance_side=entrance_side,
        exit_side=exit_side,
        entrance_exit_span=int(span),
        booths=edited.to_dict(orient="records"),
        n_visitors=int(n_visitors),
        duration_seconds=float(duration),
        dt_seconds=float(dt),
        evaluation_seed=int(seed),
        timeout_seconds=float(timeout),
        population_size=int(pop),
        n_generations=int(gen),
        workers=int(workers),
        snapshot_interval=int(interval),
        save_artifacts=bool(save_artifacts),
    )


# ================================================================ 起動


def start_run(params: run_store.RunParams) -> str:
    """config を書き出してワーカーを起動し、run_id を返す。"""
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


def pareto_figure(snapshot: dict, slots: list[dict] | None = None) -> go.Figure:
    rows = snapshot["solutions"]
    df = pd.DataFrame(rows)

    fig = go.Figure()
    if df.empty:
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


def resolve_selection(snapshot: dict, event) -> dict | None:
    """Plotly のクリックイベントから解 1 件を取り出す。"""
    points = _selection_points(event)
    if not points:
        return None

    p = points[-1]
    idx = p.get("customdata")
    if isinstance(idx, (list, tuple)):
        idx = idx[0] if idx else None
    try:
        i = int(idx)
        return dict(snapshot["solutions"][i], _index=i)
    except (TypeError, ValueError, IndexError):
        return None


# ================================================================ A/B スロット
#
# Streamlit の plotly 選択はクリックのたびに選択が「置き換わる」ので、
# 修飾キーによる多点選択はできない。
# そこでクリックを 1 件ずつ受け取り、A → B の順に 2 スロットへ溜める。


def slot_label(sol: dict) -> str:
    return (
        f"#{sol['_index']}  f_move={sol['f_move']:.1f} / "
        f"f_cong={sol['f_congestion']:.4g} / rank {sol['rank']}"
    )


def update_slots(gen: int, event, snapshot: dict) -> list[dict]:
    """クリックを A/B スロットに反映して、現在のスロットを返す。"""
    # 世代を切り替えたら解の並びが変わるのでスロットを捨てる。
    if st.session_state.get("slot_gen") != gen:
        st.session_state["slot_gen"] = gen
        st.session_state["slots"] = []
        st.session_state["slot_sig"] = None

    picked = resolve_selection(snapshot, event)
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
        f"解 A(f_move={a['f_move']:.1f} / f_cong={a['f_congestion']:.4g})",
        f"解 B(f_move={b['f_move']:.1f} / f_cong={b['f_congestion']:.4g})",
    ]
    xs = [tuple(int(v) for v in s["x"]) for s in (a, b)]

    # --- 指標の横並び
    df_kpi = pd.DataFrame(
        {
            "解 A": [a["f_move"], a["f_congestion"], a["rank"]],
            "解 B": [b["f_move"], b["f_congestion"], b["rank"]],
            "差(B - A)": [
                b["f_move"] - a["f_move"],
                b["f_congestion"] - a["f_congestion"],
                b["rank"] - a["rank"],
            ],
        },
        index=["f_move", "f_congestion", "rank"],
    )
    st.dataframe(df_kpi, width="stretch")

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


# ================================================================ 解の可視化


def solution_panel(run_id: str, snapshot: dict, solution: dict) -> None:
    rd = run_store.run_dir(run_id)
    config_dir = str(rd / "config")
    cache_dir = str(rd / "ui_cache")
    x = tuple(int(v) for v in solution["x"])

    m1, m2, m3 = st.columns(3)
    m1.metric("f_move", f"{solution['f_move']:.3f}")
    m2.metric("f_congestion", f"{solution['f_congestion']:.4g}")
    m3.metric("世代 / 順位", f"{snapshot['generation']} / rank {solution['rank']}")

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

    left, right = st.columns([1, 1])

    with left:
        st.subheader("パレート図(点をクリックして解を選択)")
        st.caption(
            "点を続けて 2 つクリックすると A・B に入り、下で推移を比較できます。"
            "3 つ目をクリックすると古い方(A)が押し出されます。"
            "やり直すときは「選択をクリア」。"
        )
        nonce = st.session_state.get("pareto_nonce", 0)
        chart_key = f"pareto_{run_id}_{gen}_{nonce}"

        # 図を描く前に選択状態を反映しておく。描画後のイベントだけを見ると
        # A/B の枠線が 1 クリック遅れて表示されるため。
        slots = update_slots(gen, st.session_state.get(chart_key), snapshot)

        event = st.plotly_chart(
            pareto_figure(snapshot, slots),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=chart_key,
        )

        # session_state 経由で拾えなかった場合の保険。
        after = update_slots(gen, event, snapshot)
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
                selected = solutions[0]
        if selected is None:
            st.info("この世代には実行可能解がありません。")
        else:
            solution_panel(run_id, snapshot, selected)

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
