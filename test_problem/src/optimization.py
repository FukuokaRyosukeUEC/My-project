"""optimization.py — pymoo Problem・NSGA-II・代表解抽出(§15, §19)。

- 離散変数 + 評価時丸め方式(§15.2)
- 制約は単一の違反量 g(n_ieq_constr = 1)
- 初期集団はリジェクションサンプリングで有効配置のみ生成(§15.3)
- 交叉 SBX(eta=15)、突然変異 PM(eta=20)は pymoo 既定
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from .constraints import compute_violation, to_integer_layout
from .models import Config
from .parallel import evaluate_vector
from .rng import sampling_rng
from .venue import build_base_venue, position_bounds


def variable_bounds(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """ブース座標の整数上下限を返す。

    ブース本体・受付・退出セル・待機列を含む形状全体が
    会場内に収まる範囲を使用する。
    """
    xl = []
    xu = []

    for booth_id, _spec in enumerate(cfg.booths):
        xmin, xmax, ymin, ymax = position_bounds(
            cfg,
            booth_id,
        )

        xl += [xmin, ymin]
        xu += [xmax, ymax]

    return (
        np.asarray(xl, dtype=np.int64),
        np.asarray(xu, dtype=np.int64),
    )


class LayoutProblem(ElementwiseProblem):
    """12 次元・2 目的・制約 1 本の配置最適化問題(§15.1)。"""

    def __init__(self, cfg: Config, **kwargs) -> None:
        self.cfg = cfg
        xl, xu = variable_bounds(cfg)

        super().__init__(
            n_var=len(cfg.booths) * 2,
            n_obj=2,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            vtype=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f_move, f_cong, g, kpis = evaluate_vector(x, self.cfg)
        out["F"] = [f_move, f_cong]
        out["G"] = [g]
        out["kpis"] = kpis


class ValidLayoutSampling(Sampling):
    """§15.3: 有効配置のみを生成するリジェクションサンプリング。"""

    def __init__(self, cfg: Config, max_trials: int = 1000) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_trials = max_trials

    def _do(self, problem, n_samples, **kwargs):
        cfg = self.cfg
        venue = build_base_venue(cfg)
        rng = sampling_rng(cfg.evaluation_seed)

        lower = problem.xl
        upper = problem.xu

        out = np.empty(
            (n_samples, problem.n_var),
            dtype=np.int64,
        )

        for i in range(n_samples):
            out[i] = sample_valid_layout(
                rng,
                lower,
                upper,
                cfg,
                venue,
                self.max_trials,
            )

        return out


def sample_valid_layout(
    rng,
    lower_bounds,
    upper_bounds,
    cfg,
    venue,
    max_trials=1000,
):
    """整数の有効配置をリジェクションサンプリングする。"""
    lower = np.asarray(
        lower_bounds,
        dtype=np.int64,
    )
    upper = np.asarray(
        upper_bounds,
        dtype=np.int64,
    )

    for _ in range(max_trials):
        # rng.integers の上限は排他的であるため、
        # upper + 1 として上下限を両方含める。
        layout = rng.integers(
            low=lower,
            high=upper + 1,
            dtype=np.int64,
        )

        positions = to_integer_layout(layout)

        if compute_violation(
            cfg,
            venue,
            positions,
        ) == 0:
            return layout

    raise RuntimeError(
        "有効配置を生成できません。"
        "会場が狭すぎる可能性があります。"
    )


def run_nsga2(
    cfg: Config,
    pop_size: int | None = None,
    n_gen: int | None = None,
    elementwise_runner=None,
    seed: int = 1,
    verbose: bool = True,
):
    """NSGA-II を実行し pymoo の Result を返す。"""
    kwargs = {}
    if elementwise_runner is not None:
        kwargs["elementwise_runner"] = elementwise_runner
    problem = LayoutProblem(cfg, **kwargs)
    algorithm = NSGA2(
        pop_size=pop_size or cfg.population_size,
        sampling=ValidLayoutSampling(cfg),

        # SBX自体の計算は実数空間で行い、
        # 生成直後に整数へ修復する。
        crossover=SBX(
            eta=15,
            prob=0.9,
            vtype=float,
            repair=RoundingRepair(),
        ),

        # Polynomial Mutationも生成直後に整数へ修復する。
        mutation=PM(
            eta=20,
            vtype=float,
            repair=RoundingRepair(),
        ),

        # 整数化後に同じ設計ベクトルとなった個体を除去する。
        eliminate_duplicates=True,
    )
    return minimize(
        problem,
        algorithm,
        ("n_gen", n_gen or cfg.n_generations),
        seed=seed,
        verbose=verbose,
        save_history=False,
    )


# ---------------------------------------------------------------- 代表解と出力(§19)
def select_representatives(F: np.ndarray) -> dict[str, int]:
    """§19.2: 移動重視・混雑重視・バランス(理想点距離最小)のインデックス。"""
    i_move = int(np.argmin(F[:, 0]))
    i_cong = int(np.argmin(F[:, 1]))
    fmin, fmax = F.min(axis=0), F.max(axis=0)
    span = np.where(fmax - fmin > 0, fmax - fmin, 1.0)
    Fn = (F - fmin) / span
    i_knee = int(np.argmin(np.sqrt((Fn**2).sum(axis=1))))
    return {"best_movement": i_move, "best_congestion": i_cong, "knee": i_knee}


def save_results(cfg: Config, res, out_dir: str | Path) -> dict:
    """§19.1 の成果物一式を書き出す。戻り値は代表解 {名前: 12 次元 list}。"""
    from .parallel import build_context, evaluate_layout  # 遅延 import

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(
        np.atleast_2d(res.X),
        dtype=np.int64,
    )
    F = np.atleast_2d(res.F)

    # パレート解ごとに KPI を再取得(有効解のみが res に残る)
    ctx = build_context(cfg)
    rows = []
    for x, f in zip(X, F):
        positions = to_integer_layout(x)
        r = evaluate_layout(ctx, positions)
        row = {
            f"x{j}": int(v)
            for j, v in enumerate(x)
        }
        row.update(
            f_move=f[0],
            f_congestion=f[1],
            mean_queue_seconds=r.kpis["mean_queue_seconds"],
            mean_blocked_seconds=r.kpis["mean_blocked_seconds"],
            experience_success_rate=r.kpis["experience_success_rate"],
            give_up_total=r.kpis["give_up_total"],
            max_cell_occupancy_ratio=r.kpis["max_cell_occupancy_ratio"],
            mean_happiness=r.kpis["mean_happiness"],
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "pareto_solutions.csv", index=False)

    reps = select_representatives(F)
    name_to_file = {
        "best_movement": "best_movement_layout.json",
        "best_congestion": "best_congestion_layout.json",
        "knee": "knee_layout.json",
    }
    rep_layouts = {}
    for name, idx in reps.items():
        payload = {
            "design_vector": [
                int(v)
                for v in X[idx]
            ],
            "positions": [
                list(p)
                for p in to_integer_layout(X[idx])
            ],
            "f_move": float(F[idx, 0]),
            "f_congestion": float(F[idx, 1]),
        }
        (out / name_to_file[name]).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rep_layouts[name] = payload
    return rep_layouts
