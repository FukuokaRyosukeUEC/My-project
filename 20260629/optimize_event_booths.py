"""
optimize_event_booths.py

Run NSGA-II for the event booth placement problem.

Outputs in the same directory:
    - optimization_result.npz
    - selected_solution.npy
    - selected_simulation_log.npz
    - settings.json

Recommended WSL setup:
    python -m venv .venv
    source .venv/bin/activate
    pip install numpy matplotlib pymoo

Run:
    python optimize_event_booths.py

For a quick test:
    python optimize_event_booths.py --pop-size 20 --n-gen 10 --T 100 --n-processes 4
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import numpy as np

from event_booth_problem import (
    RESULT_NPZ,
    SELECTED_SOLUTION_NPY,
    SELECTED_SIM_LOG_NPZ,
    SETTINGS_JSON,
    EventBoothProblem,
    OptSettings,
    SimSettings,
    hypervolume_2d_min,
    nondominated_mask,
    save_simulation_log,
    select_representative_solution,
    simulate_layout,
)


def build_algorithm(pop_size: int):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair

    return NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )


def collect_history(res) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    X_hist = []
    F_hist = []
    CV_hist = []
    for alg in res.history:
        pop = alg.pop
        X_hist.append(np.rint(pop.get("X")).astype(int))
        F_hist.append(np.asarray(pop.get("F"), dtype=float))
        G = np.asarray(pop.get("G"), dtype=float)
        if G.ndim == 2:
            G = G[:, 0]
        CV_hist.append(G)
    return X_hist, F_hist, CV_hist


def make_hv_history(F_hist: list[np.ndarray], CV_hist: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    feasible_F_all = []
    for F, CV in zip(F_hist, CV_hist):
        feasible = CV <= 0
        if np.any(feasible):
            feasible_F_all.append(F[feasible])

    if not feasible_F_all:
        return np.zeros(len(F_hist), dtype=float), np.array([1.0, 1.0], dtype=float)

    all_F = np.vstack(feasible_F_all)
    fmin = np.min(all_F, axis=0)
    fmax = np.max(all_F, axis=0)
    margin = np.maximum(0.1 * np.abs(fmax - fmin), 1.0e-6)
    ref_point = fmax + margin

    hv_history = []
    for F, CV in zip(F_hist, CV_hist):
        feasible = CV <= 0
        if not np.any(feasible):
            hv_history.append(0.0)
            continue
        Fg = F[feasible]
        Fnd = Fg[nondominated_mask(Fg)]
        hv_history.append(hypervolume_2d_min(Fnd, ref_point))
    return np.asarray(hv_history, dtype=float), ref_point


def choose_pareto_from_population(final_X: np.ndarray, final_F: np.ndarray, final_CV: np.ndarray):
    feasible = final_CV <= 0
    if np.any(feasible):
        Xf = final_X[feasible]
        Ff = final_F[feasible]
        nd = nondominated_mask(Ff)
        return Xf[nd], Ff[nd]

    # If no feasible solution is found, save the least-infeasible solutions for inspection.
    best = np.argsort(final_CV)[: min(10, len(final_CV))]
    return final_X[best], final_F[best]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=OptSettings().pop_size)
    parser.add_argument("--n-gen", type=int, default=OptSettings().n_gen)
    parser.add_argument("--opt-seed", type=int, default=OptSettings().opt_seed)
    parser.add_argument("--sim-seed", type=int, default=SimSettings().sim_seed)
    parser.add_argument("--T", type=int, default=SimSettings().T)
    parser.add_argument("--n-processes", type=int, default=OptSettings().n_processes)
    parser.add_argument("--no-verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(__file__).resolve().parent

    opt_settings = OptSettings(
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        opt_seed=args.opt_seed,
        n_processes=args.n_processes,
        verbose=not args.no_verbose,
    )
    sim_settings = SimSettings(T=args.T, sim_seed=args.sim_seed)

    print("=== Event Booth Placement Optimization ===")
    print("Population size:", opt_settings.pop_size)
    print("Generations:", opt_settings.n_gen)
    print("Simulation T:", sim_settings.T)
    print("Parallel processes:", opt_settings.n_processes)

    problem = EventBoothProblem(sim_settings=sim_settings, n_processes=opt_settings.n_processes)
    algorithm = build_algorithm(opt_settings.pop_size)

    from pymoo.optimize import minimize

    res = minimize(
        problem,
        algorithm,
        ("n_gen", opt_settings.n_gen),
        seed=opt_settings.opt_seed,
        save_history=True,
        verbose=opt_settings.verbose,
    )

    final_X = np.rint(res.pop.get("X")).astype(int)
    final_F = np.asarray(res.pop.get("F"), dtype=float)
    final_G = np.asarray(res.pop.get("G"), dtype=float)
    final_CV = final_G[:, 0] if final_G.ndim == 2 else final_G

    pareto_X, pareto_F = choose_pareto_from_population(final_X, final_F, final_CV)
    X_hist, F_hist, CV_hist = collect_history(res)
    hv_history, ref_point = make_hv_history(F_hist, CV_hist)

    if len(pareto_X) > 0:
        selected_idx = select_representative_solution(pareto_X, pareto_F)
        selected_x = pareto_X[selected_idx]
    else:
        selected_idx = 0
        selected_x = final_X[int(np.argmin(final_CV))]

    # Re-run selected solution with detailed logs for visualization and simulation scripts.
    log_settings = SimSettings(
        **{
            **asdict(sim_settings),
            "record_positions": True,
            "record_cell_counts": True,
        }
    )
    selected_result = simulate_layout(
        selected_x,
        settings=log_settings,
        return_log=True,
        force_record_positions=True,
        force_record_cell_counts=True,
    )

    np.save(out_dir / SELECTED_SOLUTION_NPY, selected_x)
    save_simulation_log(out_dir / SELECTED_SIM_LOG_NPZ, selected_result, selected_x, log_settings)

    settings_dump = {
        "opt_settings": asdict(opt_settings),
        "sim_settings": asdict(sim_settings),
        "selected_index_in_pareto": int(selected_idx),
        "objective_definition": ["mean_abandon", "negative_mean_happiness"],
        "display_happiness_is_minus_f2": True,
    }
    with open(out_dir / SETTINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(settings_dump, f, ensure_ascii=False, indent=2)

    # Save arrays. Object arrays are used for history because generation sizes can vary.
    np.savez_compressed(
        out_dir / RESULT_NPZ,
        final_X=final_X,
        final_F=final_F,
        final_CV=final_CV,
        pareto_X=pareto_X,
        pareto_F=pareto_F,
        hv_history=hv_history,
        hv_ref_point=ref_point,
        selected_x=selected_x,
        X_history=np.array(X_hist, dtype=object),
        F_history=np.array(F_hist, dtype=object),
        CV_history=np.array(CV_hist, dtype=object),
        settings_json=json.dumps(settings_dump, ensure_ascii=False),
    )

    print("\n=== Saved files ===")
    print(out_dir / RESULT_NPZ)
    print(out_dir / SELECTED_SOLUTION_NPY)
    print(out_dir / SELECTED_SIM_LOG_NPZ)
    print(out_dir / SETTINGS_JSON)
    print("\nSelected solution:", selected_x)
    print("Selected F [mean abandon, -mean happiness]:", selected_result["F"])
    print("Selected summary:", selected_result["summary"])
    print("HV reference point:", ref_point)


if __name__ == "__main__":
    main()
