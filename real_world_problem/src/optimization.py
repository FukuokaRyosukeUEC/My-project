"""optimization.py — pymoo Problem・NSGA-II・代表解抽出(§15, §19)。

- 離散変数 + 評価時丸め方式(§15.2)
- 制約は単一の違反量 g(n_ieq_constr = 1)
- 初期集団はブース単位の貪欲法サンプリングで有効配置のみ生成(§15.3)
- 交叉 SBX(eta=15)、突然変異 PM(eta=20)は pymoo 既定
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from .constraints import compute_violation, to_integer_layout
from .models import EMPTY, FORBIDDEN, Config
from .parallel import evaluate_vector
from .rng import sampling_rng
from .venue import (
    booth_cells_of,
    booth_exit_cell,
    build_base_venue,
    position_bounds,
    queue_cells_of,
    reception_cell,
)


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
    """§15.3: 有効配置のみを生成する貪欲法サンプリング。"""

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


def _booth_claimed_cells(cfg, booth_id, x, y):
    """このブースが占有するセル集合(本体・受付・退出・行列)。

    テーブル(受付・退出・行列を持たない)は本体セルのみになる。
    """
    claimed = set(booth_cells_of(cfg, booth_id, x, y))

    reception = reception_cell(cfg, booth_id, x, y)
    if reception is not None:
        claimed.add(reception)

    exit_cell = booth_exit_cell(cfg, booth_id, x, y)
    if exit_cell is not None:
        claimed.add(exit_cell)

    claimed.update(queue_cells_of(cfg, booth_id, x, y))
    return claimed


def _booth_fits(cfg, venue, occupied, booth_id, x, y) -> bool:
    """候補座標 (x, y) が、会場のベースグリッド・所属広場・既配置ブースの
    いずれとも衝突しないか判定する。

    compute_violation の項目1〜5(幾何・所属広場制約)相当を、このブース
    単体+既配置分についてだけ検査する軽量版。到達性(項目6〜8)は
    全ブース配置後に compute_violation でまとめて検証する。
    """
    H, W = cfg.grid_height, cfg.grid_width
    base = venue.base_grid

    area = cfg.booths[booth_id].area
    mask = venue.area_masks.get(area) if (venue.area_masks and area) else None

    for (cx, cy) in booth_cells_of(cfg, booth_id, x, y):
        if not (0 <= cx < W and 0 <= cy < H):
            return False
        if base[cy, cx] != EMPTY:
            return False
        if mask is not None and not mask[cy, cx]:
            return False

    reception = reception_cell(cfg, booth_id, x, y)
    if reception is not None:
        rx, ry = reception
        if not (0 <= rx < W and 0 <= ry < H) or base[ry, rx] != EMPTY:
            return False

    exit_cell = booth_exit_cell(cfg, booth_id, x, y)
    if exit_cell is not None:
        ex, ey = exit_cell
        if not (0 <= ex < W and 0 <= ey < H) or base[ey, ex] != EMPTY:
            return False

    for (qx, qy) in queue_cells_of(cfg, booth_id, x, y):
        if not (0 <= qx < W and 0 <= qy < H):
            return False
        if base[qy, qx] not in (EMPTY, FORBIDDEN):
            return False

    # ここまでの各セルは全て場内・許可種別であることを確認済みなので、
    # 既配置ブースとの重複だけを追加でチェックすればよい。
    for (cx, cy) in _booth_claimed_cells(cfg, booth_id, x, y):
        if occupied[cy, cx]:
            return False

    return True


def _greedy_layout_attempt(rng, cfg, venue, lower, upper, trials_per_booth):
    """ブースを本体面積の大きい順(同面積はランダム順)に1個ずつ、
    既配置分・会場と衝突しない座標へ配置する。

    いずれかのブースで ``trials_per_booth`` 回試しても置き場所が
    見つからなければ None を返す(呼び出し側で順序を変えて再試行する)。
    """
    n_booths = len(cfg.booths)
    H, W = cfg.grid_height, cfg.grid_width
    occupied = np.zeros((H, W), dtype=bool)

    order = sorted(
        range(n_booths),
        key=lambda b: (
            -(cfg.booths[b].width * cfg.booths[b].height),
            rng.random(),
        ),
    )

    layout = np.empty(n_booths * 2, dtype=np.int64)

    for booth_id in order:
        xmin, xmax = int(lower[booth_id, 0]), int(upper[booth_id, 0])
        ymin, ymax = int(lower[booth_id, 1]), int(upper[booth_id, 1])

        placed = False
        for _ in range(trials_per_booth):
            x = int(rng.integers(xmin, xmax + 1))
            y = int(rng.integers(ymin, ymax + 1))

            if _booth_fits(cfg, venue, occupied, booth_id, x, y):
                for (cx, cy) in _booth_claimed_cells(cfg, booth_id, x, y):
                    occupied[cy, cx] = True
                layout[2 * booth_id] = x
                layout[2 * booth_id + 1] = y
                placed = True
                break

        if not placed:
            return None

    return layout


def sample_valid_layout(
    rng,
    lower_bounds,
    upper_bounds,
    cfg,
    venue,
    max_trials=1000,
):
    """有効配置を貪欲法でサンプリングする(1個体)。

    ブースを1個ずつ、既配置分・会場と衝突しない座標へ配置していく方式。
    46ブースのような多数のブースを一括で独立ランダム配置してから
    まとめて棄却するリジェクションサンプリングは、ブース数が増えると
    全て同時に成立する確率がほぼ0になり実用的でないため、
    ブース単位で逐次検証する貪欲法に置き換えている。

    全ブースを配置できたら、最後に到達性等を含む全制約を
    compute_violation で検証する。配置順序の巡り合わせで失敗した
    場合は、順序を変えて最初からやり直す。
    """
    n_booths = len(cfg.booths)
    lower = np.asarray(lower_bounds, dtype=np.int64).reshape(n_booths, 2)
    upper = np.asarray(upper_bounds, dtype=np.int64).reshape(n_booths, 2)

    # ブースごとの候補探索回数は、ブース数で割って減らすと(46 ブースなど
    # 多い場合に)1 ブースあたりの試行数が不足するため、ブース数に関係なく
    # 十分な回数を確保する。max_trials はやり直し(配置順序の再抽選)回数
    # の予算として使う。
    trials_per_booth = max(150, max_trials // 10)
    max_restarts = max(10, max_trials // trials_per_booth)

    for _ in range(max_restarts):
        layout = _greedy_layout_attempt(rng, cfg, venue, lower, upper, trials_per_booth)
        if layout is None:
            continue

        positions = to_integer_layout(layout)

        if compute_violation(cfg, venue, positions) == 0:
            return layout

    raise RuntimeError(
        "有効配置を生成できません。"
        "会場が狭すぎる可能性があります。"
    )


class RandomValidBoothMutation(Mutation):
    """個体内の複数ブースを、制約を満たす別位置へ移動する突然変異。

    各個体について確率 ``prob`` で突然変異対象に選ぶ。
    選ばれた個体では、各ブースについて確率 ``booth_prob`` で
    移動を試みる。

    1ブースの移動先候補はランダム順に最大
    ``max_position_trials`` 個だけ確認し、最初に見つかった
    実行可能座標を採用する。

    最大試行回数以内に実行可能な座標が見つからない場合、
    そのブースは変更しない。
    """

    def __init__(
        self,
        cfg: Config,
        prob: float = 0.2,
        booth_prob: float = 0.1,
        max_position_trials: int = 100,
        seed: int = 1,
    ) -> None:
        # pymooのMutation基底クラス側では全個体に_doを適用し、
        # 個体単位・ブース単位の確率判定はこのクラス内で行う。
        super().__init__(prob=1.0)

        if not 0.0 <= prob <= 1.0:
            raise ValueError(
                "individual mutation probability must be in [0, 1]: "
                f"{prob}"
            )

        if not 0.0 <= booth_prob <= 1.0:
            raise ValueError(
                "booth mutation probability must be in [0, 1]: "
                f"{booth_prob}"
            )

        if max_position_trials < 1:
            raise ValueError(
                "max_position_trials must be at least 1: "
                f"{max_position_trials}"
            )

        self.cfg = cfg

        # 個体を突然変異対象にする確率。
        self.individual_prob = float(prob)

        # 突然変異対象個体内で、
        # 各ブースを移動対象にする確率。
        self.booth_prob = float(booth_prob)

        # 各ブースの移動先候補を制約確認する最大回数。
        self.max_position_trials = int(max_position_trials)

        # 制約判定に使用するベース会場。
        self.venue = build_base_venue(cfg)

        # 古いpymooなど、_doへrandom_stateが渡らない場合に
        # 使用する予備の乱数生成器。
        self._fallback_rng = np.random.default_rng(seed)

    def _do(
        self,
        problem,
        X,
        random_state=None,
        **kwargs,
    ):
        rng = (
            random_state
            if random_state is not None
            else self._fallback_rng
        )

        # SBX後は、整数値ではあるがfloat型になっている場合がある。
        # 例：3.0、7.0
        #
        # to_integer_layout()へ渡せるように整数配列へ変換する。
        Y = np.rint(
            np.asarray(X)
        ).astype(
            np.int64,
            copy=True,
        )

        lower = np.asarray(
            problem.xl,
            dtype=np.int64,
        )

        upper = np.asarray(
            problem.xu,
            dtype=np.int64,
        )

        n_booths = len(self.cfg.booths)

        for individual_id in range(len(Y)):
            # -----------------------------------------------------
            # この個体を突然変異対象にするか判定
            # -----------------------------------------------------
            if float(rng.random()) >= self.individual_prob:
                continue

            # 1個体内で複数のブースを順次変更できるように、
            # 成功した変更をcurrentへ保持する。
            current = Y[individual_id].copy()

            # ブース番号の小さいものから常に処理すると偏りが生じるため、
            # ブースの処理順をランダム化する。
            booth_order = rng.permutation(n_booths)

            for booth_id_raw in booth_order:
                booth_id = int(booth_id_raw)

                # -------------------------------------------------
                # このブースを突然変異させるか独立に判定
                # -------------------------------------------------
                if float(rng.random()) >= self.booth_prob:
                    continue

                x_index = 2 * booth_id
                y_index = x_index + 1

                old_x = int(current[x_index])
                old_y = int(current[y_index])

                # -------------------------------------------------
                # このブースが取り得る別座標を列挙
                # -------------------------------------------------
                candidate_positions = [
                    (x_pos, y_pos)
                    for x_pos in range(
                        int(lower[x_index]),
                        int(upper[x_index]) + 1,
                    )
                    for y_pos in range(
                        int(lower[y_index]),
                        int(upper[y_index]) + 1,
                    )
                    if (x_pos, y_pos) != (old_x, old_y)
                ]

                if not candidate_positions:
                    # 別座標が存在しない場合は、このブースを変更しない。
                    continue

                # -------------------------------------------------
                # 候補座標の確認回数を最大100回に制限
                # -------------------------------------------------
                n_trials = min(
                    self.max_position_trials,
                    len(candidate_positions),
                )

                candidate_order = rng.permutation(
                    len(candidate_positions)
                )[:n_trials]

                for candidate_id_raw in candidate_order:
                    candidate_id = int(candidate_id_raw)

                    new_x, new_y = candidate_positions[
                        candidate_id
                    ]

                    # それまでに成功した他ブースの突然変異も含む
                    # currentをコピーして候補個体を生成する。
                    candidate = current.copy()

                    candidate[x_index] = new_x
                    candidate[y_index] = new_y

                    positions = to_integer_layout(
                        candidate
                    )

                    # ---------------------------------------------
                    # 配置全体が全制約を満たす場合だけ採用
                    # ---------------------------------------------
                    if compute_violation(
                        self.cfg,
                        self.venue,
                        positions,
                    ) == 0:
                        current = candidate
                        break

                # 最大100候補を確認しても実行可能座標が
                # 見つからなかった場合、currentは変更されない。
                #
                # その場合も個体全体の突然変異は中断せず、
                # 次のブースの突然変異判定へ進む。

            # 複数ブースの突然変異結果を個体へ反映する。
            Y[individual_id] = current

        return Y

class MemeticNSGA2(NSGA2):
    """5世代ごとに基本近傍探索を行うNSGA-II。"""

    # 基本近傍：
    # 1個のブースを上下左右へ1セルだけ移動する。
    BASIC_MOVES = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    )

    def __init__(
        self,
        cfg: Config,
        local_search_interval: int = 5,
        n_dominated_parents: int = 3,
        n_random_parents: int = 5,
        n_neighbors_per_parent: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.cfg = cfg

        # 5世代に1回、局所探索を行う。
        self.local_search_interval = int(
            local_search_interval
        )

        # 被支配解から選択する親の数。
        self.n_dominated_parents = int(
            n_dominated_parents
        )

        # 集団からランダムに選択する親の数。
        self.n_random_parents = int(
            n_random_parents
        )

        # 1親から生成・評価する近傍個体の最大数。
        self.n_neighbors_per_parent = int(
            n_neighbors_per_parent
        )

        # 近傍配置の制約判定に使うベース会場。
        self.local_search_venue = build_base_venue(
            cfg
        )

    def _post_advance(self) -> None:
        """通常のNSGA-II更新後、指定世代だけ局所探索を行う。"""

        if (
            self.pop is not None
            and self.n_gen is not None
            and self.n_gen % self.local_search_interval == 0
        ):
            self._apply_local_search()

        # 局所探索後の集団を使って、
        # optimum、終了判定、callbackを更新する。
        super()._post_advance()

    def _select_parent_indices(
        self,
    ) -> np.ndarray:
        """局所探索を適用する親個体を選択する。

        選択内容：
        ・被支配解から最大3個体
        ・残りの実行可能解からランダムに最大5個体

        被支配解として選択済みの個体は、
        ランダム選択側では重複選択しない。
        """

        if self.pop is None or len(self.pop) == 0:
            return np.empty(
                0,
                dtype=np.int64,
            )

        # 実行可能性。
        feasible = np.asarray(
            self.pop.get("FEAS"),
            dtype=bool,
        ).reshape(-1)

        # NSGA-IIの非支配ランク。
        rank = np.asarray(
            self.pop.get("rank"),
            dtype=float,
        ).reshape(-1)

        # 目的関数値。
        F = np.asarray(
            self.pop.get("F"),
            dtype=float,
        )

        # 実行可能かつ目的関数値が有限である個体だけを対象にする。
        feasible_indices = np.where(
            feasible
            & np.all(
                np.isfinite(F),
                axis=1,
            )
        )[0]

        if feasible_indices.size == 0:
            return np.empty(
                0,
                dtype=np.int64,
            )

        # NSGA-IIでは、
        # rank == 0：非支配解
        # rank > 0 ：被支配解
        dominated_indices = feasible_indices[
            rank[feasible_indices] > 0
        ]

        # ---------------------------------------------------------
        # 被支配解から最大3個体
        # ---------------------------------------------------------
        n_dominated = min(
            self.n_dominated_parents,
            dominated_indices.size,
        )

        if n_dominated > 0:
            selected_dominated = np.asarray(
                self.random_state.choice(
                    dominated_indices,
                    size=n_dominated,
                    replace=False,
                ),
                dtype=np.int64,
            )
        else:
            selected_dominated = np.empty(
                0,
                dtype=np.int64,
            )

        # ---------------------------------------------------------
        # 残りの実行可能解からランダムに最大5個体
        # ---------------------------------------------------------
        random_pool = np.setdiff1d(
            feasible_indices,
            selected_dominated,
            assume_unique=False,
        )

        n_random = min(
            self.n_random_parents,
            random_pool.size,
        )

        if n_random > 0:
            selected_random = np.asarray(
                self.random_state.choice(
                    random_pool,
                    size=n_random,
                    replace=False,
                ),
                dtype=np.int64,
            )
        else:
            selected_random = np.empty(
                0,
                dtype=np.int64,
            )

        return np.concatenate(
            (
                selected_dominated,
                selected_random,
            )
        )

    def _generate_basic_neighbors(
        self,
        parent_x: np.ndarray,
        known_layouts: set[tuple[int, ...]],
    ) -> list[np.ndarray]:
        """基本近傍から実行可能な個体を最大2個生成する。

        基本近傍：
        1個のブースだけを上下左右へ1セル移動した配置。

        実行可能な基本近傍が1個も存在しない場合は、
        空リストを返して、その親の局所探索を終了する。
        """

        original = np.asarray(
            parent_x,
            dtype=np.int64,
        ).copy()

        lower = np.asarray(
            self.problem.xl,
            dtype=np.int64,
        )

        upper = np.asarray(
            self.problem.xu,
            dtype=np.int64,
        )

        # 全ブースについて、上下左右への1セル移動を列挙する。
        moves = [
            (
                booth_id,
                dx,
                dy,
            )
            for booth_id in range(
                len(self.cfg.booths)
            )
            for dx, dy in self.BASIC_MOVES
        ]

        neighbors: list[np.ndarray] = []

        # 特定ブースや特定方向へ偏らないように、
        # 基本近傍をランダム順に確認する。
        move_order = self.random_state.permutation(
            len(moves)
        )

        for move_id_raw in move_order:
            move_id = int(
                move_id_raw
            )

            booth_id, dx, dy = moves[
                move_id
            ]

            x_index = 2 * booth_id
            y_index = x_index + 1

            new_x = (
                int(original[x_index])
                + dx
            )

            new_y = (
                int(original[y_index])
                + dy
            )

            # 設計変数の上下限を超える移動は棄却する。
            if not (
                int(lower[x_index])
                <= new_x
                <= int(upper[x_index])
                and int(lower[y_index])
                <= new_y
                <= int(upper[y_index])
            ):
                continue

            candidate = original.copy()

            candidate[x_index] = new_x
            candidate[y_index] = new_y

            key = tuple(
                int(value)
                for value in candidate
            )

            # 現集団または今回生成済みの配置と
            # 同一なら評価しない。
            if key in known_layouts:
                continue

            positions = to_integer_layout(
                candidate
            )

            # 全制約を満たさない基本近傍は評価しない。
            if compute_violation(
                self.cfg,
                self.local_search_venue,
                positions,
            ) != 0:
                continue

            neighbors.append(
                candidate
            )

            known_layouts.add(
                key
            )

            # 各親から評価する近傍は最大2個。
            if (
                len(neighbors)
                >= self.n_neighbors_per_parent
            ):
                break

        # ここが空なら、その親については
        # 局所探索を行わない。
        return neighbors

    def _apply_local_search(
        self,
    ) -> None:
        """近傍個体を評価し、NSGA-IIの環境選択を再適用する。"""

        parent_indices = (
            self._select_parent_indices()
        )

        if parent_indices.size == 0:
            return

        current_X = np.asarray(
            self.pop.get("X"),
            dtype=np.int64,
        )

        # 現在集団との重複を防止する。
        known_layouts = {
            tuple(
                int(value)
                for value in row
            )
            for row in current_X
        }

        neighbor_vectors: list[
            np.ndarray
        ] = []

        # 選択された各親から、最大2個の近傍を生成する。
        for parent_index_raw in parent_indices:
            parent_index = int(
                parent_index_raw
            )

            parent_neighbors = (
                self._generate_basic_neighbors(
                    current_X[parent_index],
                    known_layouts,
                )
            )

            # 実行可能近傍が0個なら、
            # extendされず、その親の探索は終了する。
            neighbor_vectors.extend(
                parent_neighbors
            )

        if not neighbor_vectors:
            return

        # 近傍個体をpymooのPopulationへ変換する。
        local_pop = Population.new(
            "X",
            np.asarray(
                neighbor_vectors,
                dtype=np.int64,
            ),
        )

        # LayoutProblemへ設定されている
        # elementwise_runnerを使って一括評価する。
        # workers > 1なら既存の並列評価がそのまま使われる。
        self.evaluator.eval(
            self.problem,
            local_pop,
            algorithm=self,
        )

        # 現在集団と近傍個体を統合する。
        merged = Population.merge(
            self.pop,
            local_pop,
        )

        # NSGA-IIの非支配ランクと混雑距離を用いて、
        # pop_size個へ戻す。
        self.pop = self.survival.do(
            self.problem,
            merged,
            n_survive=self.pop_size,
            algorithm=self,
            random_state=self.random_state,
        )
    
class ObjectiveHistoryCallback(Callback):
    """各世代の環境選択後集団の目的関数値と制約値を保存する。"""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict] = []

    def notify(self, algorithm) -> None:
        # その世代の環境選択後集団を取得
        pop = algorithm.pop

        # 目的関数値
        F = np.asarray(
            pop.get("F"),
            dtype=float,
        ).copy()

        # 制約違反量
        G_raw = pop.get("G")

        if G_raw is None:
            G = None
        else:
            G = np.asarray(
                G_raw,
                dtype=float,
            ).copy()

        self.records.append(
            {
                "generation": int(algorithm.n_gen),
                "F": F,
                "G": G,
            }
        )


def _feasible_objectives(record: dict) -> np.ndarray:
    """履歴1世代分から、有限かつ実行可能な目的関数値だけを返す。"""

    F = np.atleast_2d(
        np.asarray(
            record["F"],
            dtype=float,
        )
    )

    # NaNやinfを含む目的関数値を除外
    finite = np.all(
        np.isfinite(F),
        axis=1,
    )

    G = record.get("G")

    if G is None or np.size(G) == 0:
        feasible = np.ones(
            F.shape[0],
            dtype=bool,
        )
    else:
        G = np.asarray(
            G,
            dtype=float,
        )

        if G.ndim == 1:
            G = G.reshape(-1, 1)

        # pymooではG <= 0が制約を満たす解
        feasible = np.all(
            G <= 0.0,
            axis=1,
        )

    return F[finite & feasible]


def _non_dominated_2d(F: np.ndarray) -> np.ndarray:
    """2目的・最小化問題の非劣解集合を返す。"""

    F = np.atleast_2d(
        np.asarray(
            F,
            dtype=float,
        )
    )

    if F.size == 0:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    if F.shape[1] != 2:
        raise ValueError(
            "HV calculation currently supports exactly "
            f"two objectives: {F.shape[1]}"
        )

    # 同じ目的関数ベクトルを除去
    points = np.unique(
        F,
        axis=0,
    )

    # 第1目的を昇順、第1目的が同値なら第2目的を昇順に並べる
    order = np.lexsort(
        (
            points[:, 1],
            points[:, 0],
        )
    )
    points = points[order]

    # 左から走査し、第2目的の最良値を更新する点だけを残す
    nondominated = []
    best_second = np.inf

    for point in points:
        if point[1] < best_second:
            nondominated.append(point)
            best_second = point[1]

    return np.asarray(
        nondominated,
        dtype=float,
    )


def _hypervolume_2d(
    F: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """2目的・最小化問題のHVを矩形和により計算する。"""

    ref = np.asarray(
        reference_point,
        dtype=float,
    ).reshape(-1)

    if ref.size != 2:
        raise ValueError(
            "reference_point must contain exactly two values"
        )

    F = np.atleast_2d(
        np.asarray(
            F,
            dtype=float,
        )
    )

    if F.size == 0:
        return 0.0

    # 参照点よりも悪い成分を持つ解はHVに寄与しない
    inside = np.all(
        F < ref,
        axis=1,
    )

    nd = _non_dominated_2d(
        F[inside]
    )

    if nd.size == 0:
        return 0.0

    hv = 0.0
    previous_y = float(ref[1])

    for x_value, y_value in nd:
        width = float(
            ref[0] - x_value
        )
        height = float(
            previous_y - y_value
        )

        if width > 0.0 and height > 0.0:
            hv += width * height

        previous_y = min(
            previous_y,
            float(y_value),
        )

    return float(hv)


def calculate_hv_history(
    res,
    reference_point: np.ndarray | None = None,
    reference_margin: float = 0.1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """各世代の実行可能な非劣解集合に対するHVを計算する。

    全世代で同一の参照点を使用する。

    reference_pointがNoneの場合は、全世代の実行可能解から
    各目的の最大値を求め、目的値レンジのreference_margin分だけ
    悪い方向へ余白を加えた点を参照点とする。
    """

    records = getattr(
        res,
        "objective_history",
        None,
    )

    if not records:
        raise ValueError(
            "Objective history is unavailable. "
            "Run the optimizer with ObjectiveHistoryCallback enabled."
        )

    if reference_margin <= 0.0:
        raise ValueError(
            "reference_margin must be positive"
        )

    # 各世代の実行可能解
    feasible_sets = [
        _feasible_objectives(record)
        for record in records
    ]

    nonempty = [
        F
        for F in feasible_sets
        if F.size > 0
    ]

    if not nonempty:
        raise ValueError(
            "No feasible objective vectors were found in the history"
        )

    all_feasible = np.vstack(
        nonempty
    )

    if all_feasible.shape[1] != 2:
        raise ValueError(
            "HV calculation currently supports exactly "
            f"two objectives: {all_feasible.shape[1]}"
        )

    # ---------------------------------------------------------
    # 全世代共通の参照点を決定
    # ---------------------------------------------------------
    if reference_point is None:
        ideal = np.min(
            all_feasible,
            axis=0,
        )
        nadir = np.max(
            all_feasible,
            axis=0,
        )

        span = nadir - ideal

        # 目的値レンジが0の軸にも正の余白を与える
        fallback_scale = np.maximum(
            np.abs(nadir),
            1.0,
        )

        padding = np.where(
            span > 0.0,
            reference_margin * span,
            reference_margin * fallback_scale,
        )

        ref = nadir + padding

    else:
        ref = np.asarray(
            reference_point,
            dtype=float,
        ).reshape(-1)

        if ref.size != 2:
            raise ValueError(
                "reference_point must contain exactly two values"
            )

        if not np.all(
            np.isfinite(ref)
        ):
            raise ValueError(
                "reference_point contains non-finite values: "
                f"{ref}"
            )

        max_objectives = np.max(
            all_feasible,
            axis=0,
        )

        if np.any(
            ref <= max_objectives
        ):
            raise ValueError(
                "reference_point must be strictly worse than every "
                "feasible objective vector. "
                f"max objectives={max_objectives}, "
                f"reference_point={ref}"
            )

    # ---------------------------------------------------------
    # 世代ごとのHVを計算
    # ---------------------------------------------------------
    rows = []

    for record, feasible_F in zip(
        records,
        feasible_sets,
    ):
        nd = _non_dominated_2d(
            feasible_F
        )

        rows.append(
            {
                "generation": int(
                    record["generation"]
                ),
                "hypervolume": _hypervolume_2d(
                    nd,
                    ref,
                ),
                "n_feasible": int(
                    feasible_F.shape[0]
                ),
                "n_nondominated": int(
                    nd.shape[0]
                ),
                "reference_f_move": float(
                    ref[0]
                ),
                "reference_f_congestion": float(
                    ref[1]
                ),
            }
        )

    return (
        pd.DataFrame(rows),
        ref,
    )

def run_nsga2(
    cfg: Config,
    pop_size: int | None = None,
    n_gen: int | None = None,
    elementwise_runner=None,
    seed: int = 1,
    verbose: bool = True,
    mutation_prob: float = 0.2,
    booth_mutation_prob: float = 0.1,
    mutation_max_position_trials: int = 100,
):
    """近傍探索付きNSGA-IIを実行し、pymooのResultを返す。"""

    kwargs = {}

    if elementwise_runner is not None:
        kwargs["elementwise_runner"] = (
            elementwise_runner
        )

    problem = LayoutProblem(
        cfg,
        **kwargs,
    )

    algorithm = MemeticNSGA2(
        cfg=cfg,

        # ---------------------------------------------------------
        # Memetic Algorithm設定
        # ---------------------------------------------------------

        # 5世代に1回、局所探索する。
        local_search_interval=5,

        # 被支配解から3個体選択する。
        n_dominated_parents=3,

        # 残りの実行可能解からランダムに5個体選択する。
        n_random_parents=5,

        # 各親から評価する実行可能近傍は最大2個。
        n_neighbors_per_parent=2,

        # ---------------------------------------------------------
        # 通常のNSGA-II設定
        # ---------------------------------------------------------
        pop_size=(
            pop_size
            or cfg.population_size
        ),

        sampling=ValidLayoutSampling(
            cfg
        ),

        # SBX自体の計算は実数空間で行い、
        # 生成直後に整数へ修復する。
        crossover=SBX(
            eta=15,
            prob=0.9,
            vtype=float,
            repair=RoundingRepair(),
        ),

        # 大域的な探索を担当する突然変異。
        #
        # 確率で選ばれた個体について、
        # 個体内の各ブースを独立確率で突然変異させる。
        #
        # 各ブースの移動先候補に対する制約確認は
        # 最大mutation_max_position_trials回までとする。
        mutation=RandomValidBoothMutation(
            cfg=cfg,
            prob=mutation_prob,
            booth_prob=booth_mutation_prob,
            max_position_trials=mutation_max_position_trials,
            seed=seed,
        ),

        # 同じ整数設計ベクトルとなった個体を除去する。
        eliminate_duplicates=True,
    )

    # 各世代の目的関数値と制約値を記録する。
    history_callback = (
        ObjectiveHistoryCallback()
    )

    res = minimize(
        problem,
        algorithm,
        (
            "n_gen",
            n_gen
            or cfg.n_generations,
        ),
        seed=seed,
        verbose=verbose,
        callback=history_callback,
        save_history=False,
    )

    # HV計算に必要な目的関数値と制約値だけを保存する。
    res.objective_history = (
        history_callback.records
    )

    return res


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
