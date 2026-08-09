"""optimize_worker.py — UI から起動される NSGA-II 実行プロセス。

    python -m ui.optimize_worker <run_dir>

Streamlit のプロセス内で最適化を回すと、
  - multiprocessing.Pool(Windows は spawn)がアプリスクリプトを再 import する
  - 長時間ブロックして UI が固まる
という 2 つの問題が出るため、独立プロセスとして分離する。
UI との通信は run_dir 配下のファイルのみ(run_store.py 参照)。

既存の `scripts/run_optimization.py` と同じ関数(run_nsga2 / save_results)を
呼ぶだけで、最適化ロジックには手を入れない。世代ごとのスナップショット出力は
`run_nsga2(on_generation=...)` の任意フックとして差し込む。
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymoo.core.problem import StarmapParallelization  # noqa: E402

from src.constraints import compute_violation  # noqa: E402
from src.models import load_config  # noqa: E402
from src.optimization import run_nsga2, save_results  # noqa: E402
from src.parallel import build_context, evaluate_layout, init_worker  # noqa: E402
from ui import run_store  # noqa: E402

#: 実会場配置の評価に使うタイムアウト(秒)。
#: 最適化中の打ち切り(§17.6)とは別に、比較基準は必ず完走させたいので長くとる。
BASELINE_TIMEOUT_SECONDS = 3600.0


def _population_snapshot(algorithm) -> list[dict]:
    """環境選択後の集団を、UI が扱いやすい素の list[dict] にする。

    実行可能(g <= 0)かつ目的関数値が有限な個体のみを残す。
    同一の設計ベクトルは 1 件にまとめる。
    """
    pop = algorithm.pop
    if pop is None or len(pop) == 0:
        return []

    X = np.asarray(pop.get("X"), dtype=np.int64)
    F = np.atleast_2d(np.asarray(pop.get("F"), dtype=float))

    G_raw = pop.get("G")
    if G_raw is None or np.size(G_raw) == 0:
        G = np.zeros((F.shape[0], 1), dtype=float)
    else:
        G = np.asarray(G_raw, dtype=float)
        if G.ndim == 1:
            G = G.reshape(-1, 1)

    rank_raw = pop.get("rank")
    rank = (
        np.full(F.shape[0], -1, dtype=float)
        if rank_raw is None
        else np.asarray(rank_raw, dtype=float).reshape(-1)
    )

    rows: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    for i in range(F.shape[0]):
        if not np.all(np.isfinite(F[i])):
            continue
        if not np.all(G[i] <= 0.0):
            continue

        key = tuple(int(v) for v in X[i])
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "x": list(key),
                "f_move": float(F[i, 0]),
                "f_congestion": float(F[i, 1]),
                "rank": (
                    int(rank[i])
                    if np.isfinite(rank[i])
                    else -1
                ),
            }
        )

    # 第1目的の昇順にしておくと UI 側でそのまま線を引ける。
    rows.sort(key=lambda r: (r["f_move"], r["f_congestion"]))
    return rows


def evaluate_fixed_layout(cfg, rd: Path) -> dict:
    """booths.yaml の fixed_position を評価し、比較基準として保存する。

    実際に使われた会場配置(調布市マルシェの実配置)を最適化解と
    同じ条件でシミュレーションし、パレート図に重ねられるようにする。
    `scripts/run_optimization.py` の `fixed_f` と同じ位置づけだが、
    こちらは制約違反や評価失敗でも例外を投げず、
    `available: false` を記録して最適化本体は続行する。
    """
    payload: dict = {"available": False}

    missing = [
        spec.booth_id
        for spec in cfg.booths
        if spec.fixed_position is None
    ]
    if missing or not cfg.booths:
        payload["reason"] = (
            "booths.yaml に fixed_position が未設定のブースがあります"
            f"(booth_id: {missing[:5]}{'…' if len(missing) > 5 else ''})"
            if missing
            else "ブースが定義されていません"
        )
        run_store.write_fixed_layout(rd, payload)
        return payload

    positions = tuple(
        tuple(int(v) for v in spec.fixed_position)
        for spec in cfg.booths
    )

    try:
        # 比較基準は必ず完走させたいのでタイムアウトだけ緩める。
        base_cfg = replace(cfg, timeout_seconds=BASELINE_TIMEOUT_SECONDS)
        ctx = build_context(base_cfg)

        violation = float(compute_violation(base_cfg, ctx.venue, positions))
        payload["violation"] = violation
        payload["feasible"] = violation <= 0.0

        if violation > 0.0:
            # 制約違反でもシミュレーション自体は行い、参考値として残す。
            payload["reason"] = (
                f"固定配置が配置制約を満たしていません(g = {violation:g})。"
                "参考値として表示します。"
            )

        result = evaluate_layout(ctx, positions)

        payload.update(
            available=True,
            label="実会場配置(booths.yaml)",
            positions=[list(p) for p in positions],
            x=[int(v) for p in positions for v in p],
            f_move=float(result.f_move),
            f_congestion=float(result.f_congestion),
            timed_out=bool(result.timed_out),
        )

    except Exception as exc:
        payload["available"] = False
        payload["reason"] = f"固定配置の評価に失敗しました: {exc}"

    run_store.write_fixed_layout(rd, payload)
    return payload


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m ui.optimize_worker <run_dir>")

    rd = Path(sys.argv[1]).resolve()
    params = run_store.RunParams.from_dict(
        run_store._read_json(rd / "params.json") or {}
    )

    cfg = load_config(rd / "config")
    t0 = time.perf_counter()

    run_store.write_status(
        rd,
        state="baseline",
        generation=0,
        n_generations=int(params.n_generations),
        started_at=t0,
        elapsed_seconds=0.0,
        message="実会場配置(booths.yaml)を評価中",
    )

    # 最適化より先に比較基準を確定させる。
    # 第 1 世代のスナップショットが出た時点でパレート図に重ねられる。
    fixed = evaluate_fixed_layout(cfg, rd)
    if fixed.get("available"):
        print(
            "実会場配置(booths.yaml): "
            f"f_move = {fixed['f_move']:.6f}, "
            f"f_congestion = {fixed['f_congestion']:.6e}"
        )
    else:
        print(f"実会場配置の比較は無効: {fixed.get('reason', '不明')}")

    run_store.write_status(
        rd,
        state="running",
        elapsed_seconds=time.perf_counter() - t0,
        message="最適化を開始しました",
    )

    interval = max(1, int(params.snapshot_interval))
    last_gen = {"n": 0}

    def on_generation(algorithm) -> None:
        """毎世代: status を更新。interval 世代ごと: スナップショットを保存。"""
        gen = int(algorithm.n_gen)
        last_gen["n"] = gen
        elapsed = time.perf_counter() - t0

        # 第 1 世代と最終世代は必ず残す(初期集団と最終結果を見るため)。
        is_edge = gen == 1 or gen >= int(params.n_generations)
        if gen % interval == 0 or is_edge:
            rows = _population_snapshot(algorithm)
            run_store.write_snapshot(
                rd,
                {
                    "generation": gen,
                    "n_generations": int(params.n_generations),
                    "elapsed_seconds": elapsed,
                    "n_solutions": len(rows),
                    "solutions": rows,
                },
            )

        run_store.write_status(
            rd,
            state="running",
            generation=gen,
            n_generations=int(params.n_generations),
            elapsed_seconds=elapsed,
            message=f"第 {gen} 世代を評価しました",
        )

    try:
        workers = max(1, int(params.workers))
        if workers > 1:
            with Pool(workers, initializer=init_worker, initargs=(cfg,)) as pool:
                res = run_nsga2(
                    cfg,
                    pop_size=params.population_size,
                    n_gen=params.n_generations,
                    elementwise_runner=StarmapParallelization(pool.starmap),
                    on_generation=on_generation,
                )
        else:
            res = run_nsga2(
                cfg,
                pop_size=params.population_size,
                n_gen=params.n_generations,
                on_generation=on_generation,
            )

        if params.save_artifacts:
            run_store.write_status(
                rd,
                state="finalizing",
                message="成果物(パレート解 CSV・代表解 JSON)を書き出し中",
            )
            save_results(cfg, res, rd / "results")

        run_store.write_status(
            rd,
            state="done",
            generation=last_gen["n"],
            elapsed_seconds=time.perf_counter() - t0,
            n_pareto=int(np.atleast_2d(res.F).shape[0]),
            message="完了",
        )

    except Exception:
        run_store.write_status(
            rd,
            state="error",
            elapsed_seconds=time.perf_counter() - t0,
            message="エラーで終了しました",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":  # §16.3 Windows 対応ガード
    main()
