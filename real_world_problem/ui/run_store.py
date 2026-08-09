"""run_store.py — 実行ディレクトリの管理と、UI・ワーカー間のファイル受け渡し。

UI(Streamlit)とワーカー(サブプロセス)は共有メモリを持たない。
すべての受け渡しは実行ディレクトリ配下のファイルで行う。

    runs/<run_id>/
        params.json              UI が入力した生パラメータ(再現用)
        config/                  ワーカーが load_config() で読む YAML 3 点
            default_config.yaml
            booths.yaml
            visitor_profiles.yaml
        status.json              進捗(毎世代更新。UI のポーリング対象)
        snapshots/gen_0010.json  N 世代ごとの集団スナップショット
        results/                 終了後の成果物(既存 save_results と同じ)
        worker.log               ワーカーの標準出力・標準エラー

書き込みは一時ファイル + os.replace による原子的置換で行う。
UI が書きかけのファイルを読むことがないようにするため。
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------- パス定義

ROOT = Path(__file__).resolve().parents[1]      # real_world_problem/
REPO_ROOT = ROOT.parent                          # リポジトリルート
RUNS_DIR = ROOT / "runs"

#: 会場プリセット。UI の「会場モード」に対応する。
#: real   : config/ の実会場ポリゴンをそのまま使う(会場サイズは編集不可)
#: rect   : venue_shape を外した矩形会場(会場サイズを自由に指定できる)
PRESET_CONFIG_DIRS = {
    "real": ROOT / "config",
    "rect": REPO_ROOT / "test_problem" / "config",
}


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def new_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------- パラメータ


@dataclass
class RunParams:
    """UI サイドバーの入力値。そのまま params.json へ保存する。"""

    # --- 会場
    venue_mode: str = "real"        # "real"(実会場ポリゴン) / "rect"(矩形)
    grid_width: int = 20            # rect のときのみ有効
    grid_height: int = 16
    cell_size_meters: float = 1.0
    entrance_side: str = "bottom"   # rect のときのみ有効
    exit_side: str = "top"
    entrance_exit_span: int = 2

    # --- ブース(booths.yaml の行そのもの)
    booths: list = field(default_factory=list)

    # --- 来場者・時間
    n_visitors: int = 200
    duration_seconds: float = 400.0
    dt_seconds: float = 1.0
    evaluation_seed: int = 0
    timeout_seconds: float = 30.0

    # --- 最適化
    population_size: int = 16
    n_generations: int = 20
    workers: int = 4
    snapshot_interval: int = 10

    # --- 終了後の成果物出力
    save_artifacts: bool = True

    def to_dict(self) -> dict:
        return {
            "venue_mode": self.venue_mode,
            "grid_width": int(self.grid_width),
            "grid_height": int(self.grid_height),
            "cell_size_meters": float(self.cell_size_meters),
            "entrance_side": self.entrance_side,
            "exit_side": self.exit_side,
            "entrance_exit_span": int(self.entrance_exit_span),
            "booths": self.booths,
            "n_visitors": int(self.n_visitors),
            "duration_seconds": float(self.duration_seconds),
            "dt_seconds": float(self.dt_seconds),
            "evaluation_seed": int(self.evaluation_seed),
            "timeout_seconds": float(self.timeout_seconds),
            "population_size": int(self.population_size),
            "n_generations": int(self.n_generations),
            "workers": int(self.workers),
            "snapshot_interval": int(self.snapshot_interval),
            "save_artifacts": bool(self.save_artifacts),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunParams":
        known = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**known)


# ---------------------------------------------------------------- 設定 YAML 生成


def load_preset_yaml(venue_mode: str) -> tuple[dict, list, list]:
    """プリセットの YAML 3 点を dict として読み込む。

    戻り値は (default_config, booths, profiles)。
    UI の初期値表示にも、実行用 config 生成にも使う。
    """
    d = PRESET_CONFIG_DIRS[venue_mode]
    main = yaml.safe_load((d / "default_config.yaml").read_text(encoding="utf-8"))
    booths = yaml.safe_load((d / "booths.yaml").read_text(encoding="utf-8"))["booths"]
    profiles = yaml.safe_load(
        (d / "visitor_profiles.yaml").read_text(encoding="utf-8")
    )["profiles"]
    return main, booths, profiles


def build_config_dir(params: RunParams, dest: Path) -> Path:
    """params を反映した config ディレクトリを dest に作る。

    プリセット YAML を読み、UI で編集された項目だけを上書きして書き出す。
    `src/models.py::load_config` はこのディレクトリをそのまま読む。
    既存のローダには一切手を入れない。
    """
    main, _preset_booths, profiles = load_preset_yaml(params.venue_mode)
    main = copy.deepcopy(main)

    # --- 会場
    if params.venue_mode == "rect":
        # 矩形会場: ポリゴン定義と明示的な入退場セルを取り除き、
        # グリッドサイズと入退場辺だけで会場を決める。
        main.pop("venue_shape", None)
        main["grid"]["width"] = int(params.grid_width)
        main["grid"]["height"] = int(params.grid_height)
        ee = main.setdefault("entrance_exit", {})
        ee.pop("entrance_cells", None)
        ee.pop("exit_cells", None)
        ee["entrance_side"] = params.entrance_side
        ee["exit_side"] = params.exit_side
        ee["span_cells"] = int(params.entrance_exit_span)
    # venue_mode == "real" のときは会場形状・グリッドをプリセットのまま使う。

    main["grid"]["cell_size_meters"] = float(params.cell_size_meters)

    # --- 時間・来場者
    main["time"]["dt_seconds"] = float(params.dt_seconds)
    main["time"]["duration_seconds"] = float(params.duration_seconds)
    main["simulation"]["n_visitors"] = int(params.n_visitors)
    main["simulation"]["evaluation_seed"] = int(params.evaluation_seed)
    main["simulation"]["timeout_seconds"] = float(params.timeout_seconds)

    # 到着プロファイルの区間はシミュレート時間に追随させる
    # (プリセットの形状比率を保ったまま横軸を伸縮する)。
    main["arrival_profile"] = _rescale_arrival_profile(
        main["arrival_profile"],
        float(params.duration_seconds),
    )

    # --- 最適化
    main["optimization"]["population_size"] = int(params.population_size)
    main["optimization"]["n_generations"] = int(params.n_generations)

    # --- ブース
    booths = _normalize_booths(params.booths, params.venue_mode)

    dest.mkdir(parents=True, exist_ok=True)
    _dump_yaml(dest / "default_config.yaml", main)
    _dump_yaml(dest / "booths.yaml", {"booths": booths})
    _dump_yaml(dest / "visitor_profiles.yaml", {"profiles": profiles})
    return dest


def _rescale_arrival_profile(profile, duration: float) -> list:
    """到着プロファイルの時間軸を duration に合わせて線形に伸縮する。"""
    rows = [[float(v) for v in row] for row in profile]
    if not rows:
        return [[0.0, duration, 1.0]]
    span = max(row[1] for row in rows)
    if span <= 0:
        return [[0.0, duration, 1.0]]
    scale = duration / span
    return [[row[0] * scale, row[1] * scale, row[2]] for row in rows]


def _normalize_booths(booths: list, venue_mode: str) -> list:
    """UI のブース表を booths.yaml のスキーマへ整える。

    - booth_id を 0 起点で振り直す(行の追加・削除に追随)
    - 矩形会場では area(広場所属)と fixed_position を落とす
      (広場が存在しないため、そのままでは配置制約が成立しない)
    """
    out = []
    for i, b in enumerate(booths):
        row = {
            "booth_id": i,
            "name": str(b.get("name") or f"ブース{i + 1}"),
            "width": int(b["width"]),
            "height": int(b["height"]),
            "category": str(b.get("category") or "none"),
            "attractiveness": float(b.get("attractiveness") or 0.0),
            "capacity": int(b.get("capacity") or 0),
            "mean_service_seconds": float(b.get("mean_service_seconds") or 0.0),
            "max_queue_length": int(b.get("max_queue_length") or 0),
            "kind": str(b.get("kind") or "booth"),
            "theme": str(b.get("theme") or "normal"),
        }
        if venue_mode == "real":
            row["area"] = str(b.get("area") or "")
            if b.get("fixed_position"):
                row["fixed_position"] = [int(v) for v in b["fixed_position"]]
        out.append(row)
    return out


def _dump_yaml(path: Path, data) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- 原子的書き込み


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- status


def write_status(rd: Path, **fields) -> None:
    """status.json を更新する。既存の内容にマージする。"""
    current = _read_json(rd / "status.json") or {}
    current.update(fields)
    current["updated_at"] = time.time()
    _write_json_atomic(rd / "status.json", current)


def read_status(rd: Path) -> dict:
    return _read_json(rd / "status.json") or {"state": "unknown"}


# ---------------------------------------------------------------- snapshot


def snapshot_path(rd: Path, generation: int) -> Path:
    return rd / "snapshots" / f"gen_{generation:04d}.json"


def write_snapshot(rd: Path, payload: dict) -> None:
    _write_json_atomic(snapshot_path(rd, int(payload["generation"])), payload)


def list_snapshot_generations(rd: Path) -> list[int]:
    """保存済みスナップショットの世代番号を昇順で返す。"""
    snap_dir = rd / "snapshots"
    if not snap_dir.is_dir():
        return []
    gens = []
    for p in snap_dir.glob("gen_*.json"):
        try:
            gens.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(gens)


def read_snapshot(rd: Path, generation: int) -> dict | None:
    return _read_json(snapshot_path(rd, generation))


# ---------------------------------------------------------------- 比較基準


def write_fixed_layout(rd: Path, payload: dict) -> None:
    """booths.yaml の固定配置の評価結果を保存する(比較基準)。"""
    _write_json_atomic(rd / "fixed_layout.json", payload)


def read_fixed_layout(rd: Path) -> dict:
    """比較基準を読む。未評価なら available: False を返す。"""
    return _read_json(rd / "fixed_layout.json") or {
        "available": False,
        "reason": "まだ評価されていません",
    }


def list_runs() -> list[str]:
    """実行 ID を新しい順に返す。"""
    if not RUNS_DIR.is_dir():
        return []
    return sorted(
        (p.name for p in RUNS_DIR.iterdir() if p.is_dir()),
        reverse=True,
    )
