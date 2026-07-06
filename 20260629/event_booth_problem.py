"""
event_booth_problem.py

Event booth placement optimization problem and grid-based pedestrian simulation.

This module is intentionally self-contained so that the other scripts can import:
    - EventBoothProblem
    - simulate_layout
    - build_layout
    - save_simulation_log / load_simulation_log

The optimization objective is minimization of:
    f1 = mean abandon count
    f2 = - mean happiness
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import os

import numpy as np

try:
    from pymoo.core.problem import Problem
except Exception:  # pymoo is only required for optimization, not for visualization/simulation.
    Problem = object


# ============================================================
# File names used by all scripts
# ============================================================
RESULT_NPZ = "optimization_result.npz"
SELECTED_SOLUTION_NPY = "selected_solution.npy"
SELECTED_SIM_LOG_NPZ = "selected_simulation_log.npz"
SETTINGS_JSON = "settings.json"

PARETO_PNG = "pareto_front.png"
LAYOUT_PNG = "layout_selected.png"
HV_HISTORY_PNG = "hv_history.png"
TIME_SERIES_PNG = "timeseries_summary.png"
QUEUE_LENGTHS_PNG = "queue_lengths.png"
CONGESTION_HEATMAP_PNG = "congestion_heatmap.png"
AGENT_ANIMATION_MP4 = "agent_animation.mp4"
CONGESTION_ANIMATION_MP4 = "congestion_animation.mp4"


# ============================================================
# Main constants
# ============================================================
H = 16
W = 20
BOOTH_IDS = (1, 2, 3, 4, 5, 6)
BOOTH_SPECS: Dict[int, Tuple[int, int]] = {
    1: (2, 2),
    2: (2, 2),
    3: (2, 2),
    4: (2, 3),
    5: (2, 3),
    6: (2, 3),
}
ENTRY_CELLS: Tuple[Tuple[int, int], ...] = ((15, 9), (15, 10))
EXIT_CELLS: Tuple[Tuple[int, int], ...] = ((0, 9), (0, 10))

# Neighbor order is fixed for deterministic tie-breaking.
NEIGHBORS_4: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))


@dataclass(frozen=True)
class SimSettings:
    """Simulation parameters. Change values here or pass a custom object."""

    T: int = 200
    spawn_mean: float = 20.0
    spawn_std: float = 2.0
    max_spawn_per_step: int = 20
    agent_type_probs: Tuple[float, float, float] = (1, 0, 0)
    queue_capacity_per_cell: int = 5
    happiness_crowd_threshold: int = 3
    happiness_decay: float = 0.9
    block_threshold: int = 1.0e6
    max_move_delay: int = 5
    experience_time: int = 20
    sim_seed: int = 0
    abandon_to_tail: bool = True
    penalty_value: float = 1.0e6
    # Set False for faster optimization if animation is not needed during evaluation.
    record_positions: bool = False
    record_cell_counts: bool = False


@dataclass(frozen=True)
class OptSettings:
    """Optimization parameters used by optimize_event_booths.py."""

    pop_size: int = 100
    n_gen: int = 100
    opt_seed: int = 0
    n_processes: int = max(1, min(os.cpu_count() or 1, 8))
    verbose: bool = True


@dataclass
class Layout:
    x: np.ndarray
    body_by_id: Dict[int, List[Tuple[int, int]]]
    queue_by_id: Dict[int, List[Tuple[int, int]]]
    queue_order_by_id: Dict[int, List[Tuple[int, int]]]
    booth_entry_by_id: Dict[int, Tuple[int, int]]
    booth_exit_by_id: Dict[int, Tuple[int, int]]
    body_owner_grid: np.ndarray
    queue_owner_grid: np.ndarray
    walkable_grid: np.ndarray
    cv: int
    cv_parts: Dict[str, int]
    dist_to_booth_entry: Dict[int, np.ndarray] = field(default_factory=dict)
    dist_to_exit: Optional[np.ndarray] = None


@dataclass
class Agent:
    agent_id: int
    agent_type: int
    position: Tuple[int, int]
    prev_position: Tuple[int, int]
    target_booth: Optional[int]
    visited_booths: set
    visit_queue: List[int]
    happiness: float = 0.0
    abandon_count: int = 0
    state: str = "walking"  # walking, queued, experiencing, exiting, finished
    cooldown: int = 0
    experience_remaining: int = 0
    experiencing_booth: Optional[int] = None
    priority_score: Dict[int, int] = field(default_factory=dict)
    temporarily_skipped: set = field(default_factory=set)


# ============================================================
# Basic utilities
# ============================================================
def in_bounds(cell: Tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W


def iter_neighbors(cell: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    r, c = cell
    for dr, dc in NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            yield (nr, nc)


def variable_bounds() -> Tuple[np.ndarray, np.ndarray]:
    xl: List[int] = []
    xu: List[int] = []
    for booth_id in BOOTH_IDS:
        h, w = BOOTH_SPECS[booth_id]
        xl.extend([1, 0])
        xu.extend([H - h - 1, W - w - 1])
    return np.array(xl, dtype=int), np.array(xu, dtype=int)


def repair_x(x: Sequence[float | int]) -> np.ndarray:
    arr = np.rint(np.asarray(x, dtype=float)).astype(int)
    xl, xu = variable_bounds()
    return np.minimum(np.maximum(arr, xl), xu)


def booth_cells_from_top_left(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    return [(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)]


def queue_cells_from_top_left(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    bottom = [(r + h, cc) for cc in range(c, c + w)]
    right = [(rr, c + w) for rr in range(r, r + h + 1)]
    # Preserve uniqueness while keeping deterministic order.
    seen = set()
    cells = []
    for cell in bottom + right:
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)
    return cells


def queue_order_from_top_left(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """FIFO visualization order. The first cell is the booth entry."""
    cells: List[Tuple[int, int]] = []
    cells.extend([(r + h, cc) for cc in range(c, c + w)])
    cells.append((r + h, c + w))
    cells.extend([(rr, c + w) for rr in range(r + h - 1, r - 1, -1)])
    seen = set()
    ordered: List[Tuple[int, int]] = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            ordered.append(cell)
    return ordered


def _overlap_count(cells: Sequence[Tuple[int, int]]) -> int:
    counter = Counter(cells)
    return int(sum(v - 1 for v in counter.values() if v > 1))


# ============================================================
# Layout construction and constraint checking
# ============================================================
def build_layout(x: Sequence[float | int], precompute_distances: bool = True) -> Layout:
    x_arr = repair_x(x)

    body_by_id: Dict[int, List[Tuple[int, int]]] = {}
    queue_by_id: Dict[int, List[Tuple[int, int]]] = {}
    queue_order_by_id: Dict[int, List[Tuple[int, int]]] = {}
    booth_entry_by_id: Dict[int, Tuple[int, int]] = {}
    booth_exit_by_id: Dict[int, Tuple[int, int]] = {}

    all_body_cells: List[Tuple[int, int]] = []
    all_queue_cells: List[Tuple[int, int]] = []
    out_of_bounds_count = 0

    for idx, booth_id in enumerate(BOOTH_IDS):
        r = int(x_arr[2 * idx])
        c = int(x_arr[2 * idx + 1])
        h, w = BOOTH_SPECS[booth_id]

        body = booth_cells_from_top_left(r, c, h, w)
        queue = queue_cells_from_top_left(r, c, h, w)
        queue_order = queue_order_from_top_left(r, c, h, w)
        b_entry = (r + h, c)
        b_exit = (r - 1, c)

        body_by_id[booth_id] = body
        queue_by_id[booth_id] = queue
        queue_order_by_id[booth_id] = queue_order
        booth_entry_by_id[booth_id] = b_entry
        booth_exit_by_id[booth_id] = b_exit

        for cell in body + queue + [b_entry, b_exit]:
            if not in_bounds(cell):
                out_of_bounds_count += 1

        all_body_cells.extend(body)
        all_queue_cells.extend(queue)

    body_overlap = _overlap_count([cell for cell in all_body_cells if in_bounds(cell)])
    queue_overlap = _overlap_count([cell for cell in all_queue_cells if in_bounds(cell)])
    body_set = set(cell for cell in all_body_cells if in_bounds(cell))
    queue_set = set(cell for cell in all_queue_cells if in_bounds(cell))
    body_queue_overlap = len(body_set & queue_set)

    special_cells = set(ENTRY_CELLS) | set(EXIT_CELLS)
    booth_entry_exit_cells = set(booth_entry_by_id.values()) | set(booth_exit_by_id.values())
    entry_exit_occupancy = len(special_cells & (body_set | queue_set | booth_entry_exit_cells))

    invalid_booth_io = 0
    for booth_id in BOOTH_IDS:
        entry = booth_entry_by_id[booth_id]
        exit_cell = booth_exit_by_id[booth_id]
        if not in_bounds(entry):
            invalid_booth_io += 1
        else:
            other_body = set()
            other_queue = set()
            for other_id in BOOTH_IDS:
                if other_id == booth_id:
                    continue
                other_body.update(body_by_id[other_id])
                other_queue.update(queue_by_id[other_id])
            if entry in other_body or entry in other_queue or entry in body_set or entry in special_cells:
                invalid_booth_io += 1
        if not in_bounds(exit_cell):
            invalid_booth_io += 1
        else:
            if exit_cell in body_set or exit_cell in queue_set or exit_cell in special_cells:
                invalid_booth_io += 1

    body_owner_grid = np.zeros((H, W), dtype=np.int16)
    queue_owner_grid = np.zeros((H, W), dtype=np.int16)
    for booth_id, cells in body_by_id.items():
        for r, c in cells:
            if in_bounds((r, c)) and body_owner_grid[r, c] == 0:
                body_owner_grid[r, c] = booth_id
    for booth_id, cells in queue_by_id.items():
        for r, c in cells:
            if in_bounds((r, c)) and queue_owner_grid[r, c] == 0:
                queue_owner_grid[r, c] = booth_id

    walkable_grid = np.ones((H, W), dtype=bool)
    walkable_grid[body_owner_grid > 0] = False
    walkable_grid[queue_owner_grid > 0] = False

    cv_parts = {
        "out_of_bounds": int(out_of_bounds_count),
        "body_overlap": int(body_overlap),
        "queue_overlap": int(queue_overlap),
        "body_queue_overlap": int(body_queue_overlap),
        "entry_exit_occupancy": int(entry_exit_occupancy),
        "invalid_booth_io": int(invalid_booth_io),
        "unreachable_paths": 0,
    }

    layout = Layout(
        x=x_arr,
        body_by_id=body_by_id,
        queue_by_id=queue_by_id,
        queue_order_by_id=queue_order_by_id,
        booth_entry_by_id=booth_entry_by_id,
        booth_exit_by_id=booth_exit_by_id,
        body_owner_grid=body_owner_grid,
        queue_owner_grid=queue_owner_grid,
        walkable_grid=walkable_grid,
        cv=0,
        cv_parts=cv_parts,
    )

    # Only run reachability checks when basic geometric constraints are valid.
    basic_cv = sum(v for k, v in cv_parts.items() if k != "unreachable_paths")
    if basic_cv == 0:
        if precompute_distances:
            precompute_distance_maps(layout)
        cv_parts["unreachable_paths"] = count_unreachable_paths(layout)
    layout.cv = int(sum(cv_parts.values()))
    return layout


def bfs_distance(
    walkable_grid: np.ndarray,
    targets: Sequence[Tuple[int, int]],
    extra_allowed: Optional[Iterable[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Reverse BFS distance to a target set.

    Cells in extra_allowed can be treated as traversable endpoints, which is used
    for booth entry cells that are otherwise queue cells.
    """
    allowed = walkable_grid.copy()
    if extra_allowed is not None:
        for cell in extra_allowed:
            if in_bounds(cell):
                allowed[cell] = True
    dist = np.full((H, W), np.inf, dtype=float)
    q: deque[Tuple[int, int]] = deque()
    for cell in targets:
        if in_bounds(cell):
            allowed[cell] = True
            dist[cell] = 0.0
            q.append(cell)
    while q:
        cell = q.popleft()
        for nb in iter_neighbors(cell):
            if not allowed[nb]:
                continue
            if math.isinf(dist[nb]):
                dist[nb] = dist[cell] + 1.0
                q.append(nb)
    return dist


def precompute_distance_maps(layout: Layout) -> None:
    layout.dist_to_booth_entry = {}
    for booth_id in BOOTH_IDS:
        entry = layout.booth_entry_by_id[booth_id]
        layout.dist_to_booth_entry[booth_id] = bfs_distance(
            layout.walkable_grid,
            targets=[entry],
            extra_allowed=[entry],
        )
    layout.dist_to_exit = bfs_distance(layout.walkable_grid, targets=EXIT_CELLS)


def count_unreachable_paths(layout: Layout) -> int:
    if not layout.dist_to_booth_entry or layout.dist_to_exit is None:
        precompute_distance_maps(layout)

    unreachable = 0
    # Venue entries -> each booth entry
    for booth_id in BOOTH_IDS:
        dist = layout.dist_to_booth_entry[booth_id]
        if all(math.isinf(dist[cell]) for cell in ENTRY_CELLS):
            unreachable += 1

    # Each booth exit -> venue exit
    for booth_id in BOOTH_IDS:
        start = layout.booth_exit_by_id[booth_id]
        if not in_bounds(start) or math.isinf(layout.dist_to_exit[start]):
            unreachable += 1

    # Each booth exit -> every other booth entry
    for from_id in BOOTH_IDS:
        start = layout.booth_exit_by_id[from_id]
        for to_id in BOOTH_IDS:
            if from_id == to_id:
                continue
            dist = layout.dist_to_booth_entry[to_id]
            if not in_bounds(start) or math.isinf(dist[start]):
                unreachable += 1
    return int(unreachable)


def next_step_toward(position: Tuple[int, int], dist: np.ndarray) -> Optional[Tuple[int, int]]:
    if not in_bounds(position) or math.isinf(dist[position]):
        return None
    best = position
    best_d = dist[position]
    for nb in iter_neighbors(position):
        if dist[nb] < best_d:
            best = nb
            best_d = dist[nb]
    if best == position:
        return None
    return best


# ============================================================
# Simulation logic
# ============================================================
def _new_agent(agent_id: int, agent_type: int, position: Tuple[int, int], rng: np.random.Generator) -> Agent:
    booth_ids = list(BOOTH_IDS)
    if agent_type == 1:
        visit_order = list(rng.permutation(booth_ids))
        scores = {booth_id: score for booth_id, score in zip(visit_order, [60, 50, 40, 30, 20, 10])}
    elif agent_type == 2:
        visit_order = booth_ids
        scores = {booth_id: 35 for booth_id in booth_ids}
    else:
        visit_order = booth_ids
        scores = {booth_id: 35 for booth_id in booth_ids}

    agent = Agent(
        agent_id=agent_id,
        agent_type=agent_type,
        position=position,
        prev_position=position,
        target_booth=None,
        visited_booths=set(),
        visit_queue=visit_order,
        priority_score=scores,
    )
    return agent


def _select_target_booth(agent: Agent, layout: Layout) -> Optional[int]:
    remaining = [b for b in agent.visit_queue if b not in agent.visited_booths]
    if not remaining:
        return None

    if agent.agent_type in (1, 2):
        return remaining[0]

    # Agent type 3: nearest booth by current shortest-path distance.
    candidates = [b for b in remaining if b not in agent.temporarily_skipped]
    if not candidates:
        agent.temporarily_skipped.clear()
        candidates = remaining
    best_booth = None
    best_key = (np.inf, np.inf)
    for booth_id in candidates:
        dist = layout.dist_to_booth_entry[booth_id][agent.position]
        key = (dist, booth_id)
        if key < best_key:
            best_key = key
            best_booth = booth_id
    return best_booth


def _handle_abandon(agent: Agent, settings: SimSettings, layout: Layout) -> None:
    abandoned = agent.target_booth
    agent.abandon_count += 1
    if abandoned is None:
        agent.target_booth = _select_target_booth(agent, layout)
        return

    if agent.agent_type in (1, 2):
        if settings.abandon_to_tail and abandoned in agent.visit_queue:
            agent.visit_queue.remove(abandoned)
            agent.visit_queue.append(abandoned)
    else:
        remaining = [b for b in agent.visit_queue if b not in agent.visited_booths]
        if len(remaining) > 1:
            agent.temporarily_skipped.add(abandoned)
    agent.target_booth = _select_target_booth(agent, layout)


def _handle_success(agent: Agent, booth_id: int, layout: Layout) -> None:
    agent.visited_booths.add(booth_id)
    if booth_id in agent.visit_queue:
        agent.visit_queue.remove(booth_id)
    agent.temporarily_skipped.discard(booth_id)
    agent.happiness += float(agent.priority_score.get(booth_id, 35))
    agent.target_booth = _select_target_booth(agent, layout)
    if agent.target_booth is None:
        agent.state = "exiting"
    else:
        agent.state = "walking"


def _make_occupancy_counts(
    agents: Dict[int, Agent],
    include_all_states: bool = False,
) -> np.ndarray:
    counts = np.zeros((H, W), dtype=np.int16)
    for a in agents.values():
        if a.state == "finished":
            continue
        if include_all_states or a.state in ("walking", "exiting"):
            if in_bounds(a.position):
                counts[a.position] += 1
    return counts


def _update_queue_positions(layout: Layout, agents: Dict[int, Agent], booth_queues: Dict[int, deque], settings: SimSettings) -> None:
    per_cell = max(1, settings.queue_capacity_per_cell)
    for booth_id, q in booth_queues.items():
        order = layout.queue_order_by_id[booth_id]
        for index, agent_id in enumerate(q):
            cell_index = min(index // per_cell, len(order) - 1)
            agents[agent_id].position = order[cell_index]


def _serialize_agent_positions(agents: Dict[int, Agent]) -> List[Tuple[int, int, int, int, str]]:
    data: List[Tuple[int, int, int, int, str]] = []
    for a in sorted(agents.values(), key=lambda z: z.agent_id):
        if a.state == "finished":
            continue
        r, c = a.position
        data.append((a.agent_id, a.agent_type, r, c, a.state))
    return data


def simulate_layout(
    x: Sequence[float | int],
    settings: Optional[SimSettings] = None,
    return_log: bool = True,
    force_record_positions: bool = False,
    force_record_cell_counts: bool = False,
) -> Dict[str, object]:
    """Run the grid-based agent simulation for one layout.

    Returns a dict containing F, CV, mean metrics and optional time-series logs.
    """
    settings = settings or SimSettings()
    layout = build_layout(x, precompute_distances=True)
    if layout.cv > 0:
        return {
            "F": np.array([settings.penalty_value, settings.penalty_value], dtype=float),
            "CV": float(layout.cv),
            "cv_parts": layout.cv_parts,
            "layout": layout,
            "summary": {
                "mean_abandon": settings.penalty_value,
                "mean_happiness": -settings.penalty_value,
                "total_spawned": 0,
                "total_finished": 0,
            },
            "log": None,
        }

    rng = np.random.default_rng(settings.sim_seed)
    agents: Dict[int, Agent] = {}
    next_agent_id = 0
    booth_queues: Dict[int, deque] = {b: deque() for b in BOOTH_IDS}
    booth_current: Dict[int, Optional[int]] = {b: None for b in BOOTH_IDS}
    booth_completed = {b: 0 for b in BOOTH_IDS}
    cumulative_abandon = 0

    T = int(settings.T)
    spawned_per_step = np.zeros(T, dtype=np.int16)
    active_count = np.zeros(T, dtype=np.int32)
    finished_count = np.zeros(T, dtype=np.int32)
    cumulative_abandon_log = np.zeros(T, dtype=np.int32)
    mean_happiness_log = np.zeros(T, dtype=float)
    crowded_cells_log = np.zeros(T, dtype=np.int32)
    booth_queue_log = np.zeros((T, len(BOOTH_IDS)), dtype=np.int16)
    booth_experiencing_log = np.zeros((T, len(BOOTH_IDS)), dtype=np.int16)

    record_positions = settings.record_positions or force_record_positions
    record_cell_counts = settings.record_cell_counts or force_record_cell_counts
    agent_position_log: List[List[Tuple[int, int, int, int, str]]] = []
    cell_counts_log = np.zeros((T, H, W), dtype=np.int16) if record_cell_counts else None

    total_spawned = 0

    for t in range(T):
        # 1. Spawn new agents.
        requested_spawn = int(np.rint(rng.normal(settings.spawn_mean, settings.spawn_std)))
        requested_spawn = max(0, min(requested_spawn, settings.max_spawn_per_step))
        walking_counts = _make_occupancy_counts(agents, include_all_states=False)

        for _ in range(requested_spawn):
            available_entries = [cell for cell in ENTRY_CELLS if walking_counts[cell] < settings.block_threshold]
            if not available_entries:
                break
            min_count = min(walking_counts[cell] for cell in available_entries)
            best_entries = [cell for cell in available_entries if walking_counts[cell] == min_count]
            entry_cell = best_entries[int(rng.integers(0, len(best_entries)))]
            agent_type = int(rng.choice([1, 2, 3], p=np.array(settings.agent_type_probs, dtype=float)))
            agent = _new_agent(next_agent_id, agent_type, entry_cell, rng)
            agent.target_booth = _select_target_booth(agent, layout)
            agents[next_agent_id] = agent
            next_agent_id += 1
            total_spawned += 1
            spawned_per_step[t] += 1
            walking_counts[entry_cell] += 1

        # 2. Decrease remaining experience time.
        for a in sorted(agents.values(), key=lambda z: z.agent_id):
            if a.state == "experiencing":
                a.experience_remaining -= 1

        # 3. Complete booth experience.
        for a in sorted(agents.values(), key=lambda z: z.agent_id):
            if a.state == "experiencing" and a.experience_remaining <= 0:
                booth_id = a.experiencing_booth
                if booth_id is None:
                    continue
                booth_current[booth_id] = None
                booth_completed[booth_id] += 1
                a.experiencing_booth = None
                a.position = layout.booth_exit_by_id[booth_id]
                a.prev_position = a.position
                a.cooldown = 0
                _handle_success(a, booth_id, layout)

        # 4. Start booth experience from queues.
        for booth_id in BOOTH_IDS:
            if booth_current[booth_id] is None and booth_queues[booth_id]:
                aid = booth_queues[booth_id].popleft()
                a = agents[aid]
                a.state = "experiencing"
                a.experiencing_booth = booth_id
                a.experience_remaining = settings.experience_time
                a.position = layout.body_by_id[booth_id][0]
                booth_current[booth_id] = aid
        _update_queue_positions(layout, agents, booth_queues, settings)

        # 5 and 6. Cooldown and movement, in ascending agent_id order.
        walking_counts = _make_occupancy_counts(agents, include_all_states=False)
        for aid in sorted(agents.keys()):
            a = agents[aid]
            if a.state not in ("walking", "exiting"):
                continue
            if a.cooldown > 0:
                a.cooldown -= 1
                continue

            if a.state == "walking":
                if a.target_booth is None:
                    a.target_booth = _select_target_booth(a, layout)
                    if a.target_booth is None:
                        a.state = "exiting"
                        dist = layout.dist_to_exit
                    else:
                        dist = layout.dist_to_booth_entry[a.target_booth]
                else:
                    dist = layout.dist_to_booth_entry[a.target_booth]
            else:
                dist = layout.dist_to_exit

            if dist is None:
                continue
            next_cell = next_step_toward(a.position, dist)
            if next_cell is None:
                continue

            if walking_counts[next_cell] >= settings.block_threshold:
                # Stop due to dynamic congestion. No rerouting here.
                continue

            old_pos = a.position
            if in_bounds(old_pos):
                walking_counts[old_pos] = max(0, walking_counts[old_pos] - 1)
            a.prev_position = old_pos
            a.position = next_cell
            walking_counts[next_cell] += 1
            delay = min(max(1, int(walking_counts[next_cell])), settings.max_move_delay)
            a.cooldown = delay - 1

        # 7. Booth entry arrival: join queue or abandon.
        walking_counts = _make_occupancy_counts(agents, include_all_states=False)
        for aid in sorted(agents.keys()):
            a = agents[aid]
            if a.state != "walking" or a.target_booth is None:
                continue
            target = a.target_booth
            if a.position != layout.booth_entry_by_id[target]:
                continue
            capacity = settings.queue_capacity_per_cell * len(layout.queue_order_by_id[target])
            if len(booth_queues[target]) < capacity:
                # Remove the agent from walking occupancy and place it into FIFO queue.
                a.state = "queued"
                a.cooldown = 0
                booth_queues[target].append(aid)
            else:
                # Return to previous walkable cell to avoid getting stuck on a queue-only cell.
                if in_bounds(a.prev_position) and layout.walkable_grid[a.prev_position]:
                    a.position = a.prev_position
                _handle_abandon(a, settings, layout)
                cumulative_abandon += 1
        _update_queue_positions(layout, agents, booth_queues, settings)

        # 8. Finish agents that reached venue exit.
        for aid in sorted(agents.keys()):
            a = agents[aid]
            if a.state == "exiting" and a.position in EXIT_CELLS:
                a.state = "finished"
                a.cooldown = 0

        # 9. Happiness decay due to congestion on walkable cells.
        walking_counts = _make_occupancy_counts(agents, include_all_states=False)
        for a in agents.values():
            if a.state in ("walking", "exiting") and in_bounds(a.position):
                if walking_counts[a.position] >= settings.happiness_crowd_threshold:
                    a.happiness = max(0.0, a.happiness * settings.happiness_decay)

        # 10. Logs.
        total_happiness = sum(a.happiness for a in agents.values())
        total_finished = sum(1 for a in agents.values() if a.state == "finished")
        total_abandon_now = sum(a.abandon_count for a in agents.values())
        cumulative_abandon = total_abandon_now
        active_now = sum(1 for a in agents.values() if a.state != "finished")
        all_counts = _make_occupancy_counts(agents, include_all_states=True)

        active_count[t] = active_now
        finished_count[t] = total_finished
        cumulative_abandon_log[t] = cumulative_abandon
        mean_happiness_log[t] = total_happiness / total_spawned if total_spawned > 0 else 0.0
        crowded_cells_log[t] = int(np.sum(walking_counts >= settings.happiness_crowd_threshold))
        for j, booth_id in enumerate(BOOTH_IDS):
            booth_queue_log[t, j] = len(booth_queues[booth_id])
            booth_experiencing_log[t, j] = 1 if booth_current[booth_id] is not None else 0
        if record_cell_counts and cell_counts_log is not None:
            cell_counts_log[t] = all_counts
        if record_positions:
            agent_position_log.append(_serialize_agent_positions(agents))

    total_abandon = sum(a.abandon_count for a in agents.values())
    total_happiness = sum(a.happiness for a in agents.values())
    mean_abandon = total_abandon / total_spawned if total_spawned > 0 else 0.0
    mean_happiness = total_happiness / total_spawned if total_spawned > 0 else 0.0
    F = np.array([mean_abandon, -mean_happiness], dtype=float)

    log = None
    if return_log:
        log = {
            "spawned_per_step": spawned_per_step,
            "active_count": active_count,
            "finished_count": finished_count,
            "cumulative_abandon": cumulative_abandon_log,
            "mean_happiness": mean_happiness_log,
            "crowded_cells": crowded_cells_log,
            "booth_queue_lengths": booth_queue_log,
            "booth_experiencing": booth_experiencing_log,
            "booth_completed": np.array([booth_completed[b] for b in BOOTH_IDS], dtype=np.int32),
            "cell_counts": cell_counts_log,
            "agent_positions": np.array(agent_position_log, dtype=object) if record_positions else None,
        }

    return {
        "F": F,
        "CV": 0.0,
        "cv_parts": layout.cv_parts,
        "layout": layout,
        "summary": {
            "mean_abandon": float(mean_abandon),
            "mean_happiness": float(mean_happiness),
            "total_spawned": int(total_spawned),
            "total_finished": int(sum(1 for a in agents.values() if a.state == "finished")),
            "total_abandon": int(total_abandon),
        },
        "log": log,
    }


def evaluate_candidate(args: Tuple[Sequence[float | int], SimSettings]) -> Tuple[np.ndarray, float]:
    x, settings = args
    # During optimization, detailed logs are disabled for speed.
    eval_settings = SimSettings(**{**asdict(settings), "record_positions": False, "record_cell_counts": False})
    result = simulate_layout(x, settings=eval_settings, return_log=False)
    return np.asarray(result["F"], dtype=float), float(result["CV"])


# ============================================================
# Saving and loading simulation logs
# ============================================================
def save_simulation_log(path: str | Path, sim_result: Dict[str, object], x: Sequence[float | int], settings: SimSettings) -> None:
    log = sim_result.get("log") or {}
    summary = sim_result.get("summary") or {}
    cv_parts = sim_result.get("cv_parts") or {}
    np.savez_compressed(
        path,
        x=repair_x(x),
        F=np.asarray(sim_result.get("F"), dtype=float),
        CV=float(sim_result.get("CV", np.nan)),
        summary_json=json.dumps(summary, ensure_ascii=False),
        cv_parts_json=json.dumps(cv_parts, ensure_ascii=False),
        settings_json=json.dumps(asdict(settings), ensure_ascii=False),
        spawned_per_step=log.get("spawned_per_step"),
        active_count=log.get("active_count"),
        finished_count=log.get("finished_count"),
        cumulative_abandon=log.get("cumulative_abandon"),
        mean_happiness=log.get("mean_happiness"),
        crowded_cells=log.get("crowded_cells"),
        booth_queue_lengths=log.get("booth_queue_lengths"),
        booth_experiencing=log.get("booth_experiencing"),
        booth_completed=log.get("booth_completed"),
        cell_counts=log.get("cell_counts"),
        agent_positions=log.get("agent_positions"),
    )


def load_simulation_log(path: str | Path) -> Dict[str, object]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    for key in ("summary_json", "cv_parts_json", "settings_json"):
        if key in out:
            raw = out[key]
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            out[key.replace("_json", "")] = json.loads(raw)
    return out


# ============================================================
# Nondominated sorting and 2D hypervolume utilities
# ============================================================
def nondominated_mask(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominates_i):
            mask[i] = False
    return mask


def hypervolume_2d_min(F: np.ndarray, ref_point: Sequence[float]) -> float:
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if F.size == 0:
        return 0.0
    finite = np.all(np.isfinite(F), axis=1)
    F = F[finite]
    if F.size == 0:
        return 0.0
    F = F[nondominated_mask(F)]
    # Keep only points better than the reference point.
    F = F[np.all(F < ref, axis=1)]
    if F.size == 0:
        return 0.0
    order = np.argsort(F[:, 0])
    F = F[order]
    hv = 0.0
    prev_y = ref[1]
    for x, y in F:
        if y < prev_y:
            hv += max(0.0, ref[0] - x) * max(0.0, prev_y - y)
            prev_y = y
    return float(max(0.0, hv))


def select_representative_solution(X: np.ndarray, F: np.ndarray) -> int:
    """Select a balanced representative from a Pareto set."""
    if len(X) == 0:
        return 0
    F = np.asarray(F, dtype=float)
    fmin = np.min(F, axis=0)
    fmax = np.max(F, axis=0)
    span = np.where(np.abs(fmax - fmin) < 1e-12, 1.0, fmax - fmin)
    Fn = (F - fmin) / span
    return int(np.argmin(np.linalg.norm(Fn, axis=1)))


# ============================================================
# pymoo Problem class
# ============================================================
class EventBoothProblem(Problem):
    def __init__(self, sim_settings: Optional[SimSettings] = None, n_processes: int = 1):
        self.sim_settings = sim_settings or SimSettings()
        self.n_processes = max(1, int(n_processes))
        xl, xu = variable_bounds()
        super().__init__(
            n_var=12,
            n_obj=2,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(np.rint(X).astype(int))
        tasks = [(x, self.sim_settings) for x in X]
        if self.n_processes > 1 and len(tasks) > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=self.n_processes) as ex:
                results = list(ex.map(evaluate_candidate, tasks))
        else:
            results = [evaluate_candidate(task) for task in tasks]
        F = np.vstack([r[0] for r in results])
        CV = np.array([r[1] for r in results], dtype=float)
        out["F"] = F
        out["G"] = CV.reshape(-1, 1)


if __name__ == "__main__":
    # Quick smoke test.
    xl, xu = variable_bounds()
    rng = np.random.default_rng(0)
    x = rng.integers(xl, xu + 1)
    layout = build_layout(x)
    print("x:", x)
    print("CV:", layout.cv)
    print("CV parts:", layout.cv_parts)
    result = simulate_layout(x, SimSettings(T=10, record_positions=True, record_cell_counts=True))
    print("F:", result["F"], "summary:", result["summary"])
