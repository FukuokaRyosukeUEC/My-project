import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.direct_optimization import (
    ORDERLINESS_COMPONENT_KEYS,
    OrderlinessLimits,
    save_direct_metadata,
)
from src.models import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_direct_metadata_saves_configured_proximity_distance(tmp_path):
    cfg = load_config(ROOT / "config")
    components = {"total": 0.0}
    components.update({key: 0.0 for key in ORDERLINESS_COMPONENT_KEYS})
    result = SimpleNamespace(
        X=np.zeros((1, 120), dtype=np.int64),
        orderliness_components=[components],
        width_reachability_profiles=[
            {
                "four_cell_achievement_count": 0,
                "exact_counts": {4: 0, 3: 0, 2: 0, 1: 0},
            }
        ],
        memetic_history=[],
        mutation_history=[],
        diversity_history=[],
        orderliness_limits=OrderlinessLimits(),
        mutation_settings={},
        memetic_settings={},
        orderliness_constraint_mode="descriptive_only_not_a_constraint",
        diversity_settings={},
        donor_pool_final_size=0,
    )

    save_direct_metadata(result, tmp_path, cfg=cfg)

    metadata = json.loads((tmp_path / "direct_120_method.json").read_text("utf-8"))
    proximity = metadata["same_type_proximity_constraint"]
    assert proximity["maximum_center_distance_meters"] == 5.0
    assert proximity["group_source"] == "booths.yaml placement_group"
