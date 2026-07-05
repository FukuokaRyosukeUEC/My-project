"""テスト共通フィクスチャ。中央密集・周辺分散の手作り配置(§23.6 用)。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import load_config  # noqa: E402
from src.parallel import build_context  # noqa: E402

#: 中央密集配置(ブースを中央付近へ集める)
CENTRAL_LAYOUT = ((12, 6), (24, 6), (18, 6), (12, 17), (24, 17), (19, 17))
#: 周辺分散配置(ブースを外周へ分散)
PERIMETER_LAYOUT = ((4, 0), (32, 0), (18, 0), (4, 16), (32, 16), (18, 19))


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="session")
def ctx(cfg):
    return build_context(cfg)
