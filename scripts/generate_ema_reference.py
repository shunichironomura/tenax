#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "ema-workbench==3.0.0",
# ]
# ///
"""Regenerate deterministic PRIM fixtures with EMA Workbench.

The fixture is intentionally produced without importing or invoking Tenax.  It
therefore remains an independent, executable reference for the Rust tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ema_workbench
import numpy as np
import pandas as pd
from ema_workbench.analysis import prim

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "ema_workbench_3_0_0.json"


def continuous_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A deterministic, non-axis-trivial continuous classification problem."""
    size = 160
    row = np.arange(size)
    experiments = pd.DataFrame(
        {
            "load": (row + 0.5) / size,
            "resilience": ((row * 47) % size + 0.5) / size,
            "recovery": ((row * 89) % size + 0.5) / size,
        }
    )
    target = ((experiments["load"] >= 0.58) & (experiments["resilience"] <= 0.44)) | (
        (experiments["load"] >= 0.82) & (experiments["recovery"] >= 0.75)
    )
    return experiments, target.to_numpy(dtype=int)


def lenient2_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A problem on which the lenient objectives choose different pastes."""
    rng = np.random.default_rng(112)
    size = 180
    experiments = pd.DataFrame(
        {
            "x1": rng.random(size),
            "x2": rng.random(size),
            "x3": rng.random(size),
        }
    )
    probability = (
        0.03
        + 0.5 * ((experiments["x1"] > 0.5) & (experiments["x2"] < 0.6))
        + 0.3 * ((experiments["x3"] > 0.8) & (experiments["x1"] > 0.2))
    )
    target = (rng.random(size) < probability).astype(int)
    return experiments, target


def pasting_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A deterministic noisy problem whose trajectory performs three pastes."""
    rng = np.random.default_rng(6)
    size = 200
    x1 = rng.random(size)
    x2 = rng.random(size)
    probability = 0.04 + 0.72 * ((x1 > 0.48) & (x2 < 0.58)) + 0.12 * (x1 > 0.72)
    target = (rng.random(size) < probability).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2}), target


def mixed_data() -> tuple[pd.DataFrame, np.ndarray]:
    """A full-factorial problem exercising integer and categorical peels."""
    rows = [
        (capacity, demand, regime)
        for regime in ["baseline", "efficient", "fragile", "robust"]
        for capacity in range(10)
        for demand in range(10)
    ]
    experiments = pd.DataFrame(rows, columns=["capacity", "demand", "regime"])
    target = (
        (
            (experiments["regime"] == "fragile")
            & (experiments["capacity"] >= 5)
            & (experiments["demand"] <= 4)
        )
        | (
            (experiments["regime"] == "baseline")
            & (experiments["capacity"] >= 8)
            & (experiments["demand"] <= 2)
        )
        | (
            (experiments["regime"] == "efficient")
            & (experiments["capacity"] == 9)
            & (experiments["demand"] == 0)
        )
    )
    return experiments, target.to_numpy(dtype=int)


def feature_kind(series: pd.Series) -> str:
    if pd.api.types.is_float_dtype(series.dtype):
        return "continuous"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "integer"
    return "categorical"


def json_value(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def serialize_features(experiments: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "kind": feature_kind(experiments[name]),
            "values": [json_value(value) for value in experiments[name]],
        }
        for name in experiments.columns
    ]


def serialize_limit(name: str, kind: str, limits: pd.DataFrame) -> dict[str, Any]:
    if kind == "categorical":
        return {
            "name": name,
            "kind": kind,
            "categories": sorted(limits.at[0, name]),
        }
    return {
        "name": name,
        "kind": kind,
        "lower": json_value(limits.at[0, name]),
        "upper": json_value(limits.at[1, name]),
    }


def indices_in_box(experiments: pd.DataFrame, limits: pd.DataFrame) -> list[int]:
    inside = np.ones(len(experiments), dtype=bool)
    for name in experiments.columns:
        if feature_kind(experiments[name]) == "categorical":
            inside &= experiments[name].isin(limits.at[0, name]).to_numpy()
        else:
            inside &= (experiments[name] >= limits.at[0, name]).to_numpy()
            inside &= (experiments[name] <= limits.at[1, name]).to_numpy()
    return np.flatnonzero(inside).tolist()


def optional_p_value(value: float) -> float | None:
    return None if value == -1 else float(value)


def run_case(
    name: str,
    experiments: pd.DataFrame,
    target: np.ndarray,
    *,
    objective: prim.PRIMObjectiveFunctions,
    peel_alpha: float,
    paste_alpha: float,
    mass_min: float,
) -> dict[str, Any]:
    algorithm = prim.Prim(
        experiments,
        target,
        obj_function=objective,
        peel_alpha=peel_alpha,
        paste_alpha=paste_alpha,
        mass_min=mass_min,
    )
    box = algorithm.find_box()
    kinds = {column: feature_kind(experiments[column]) for column in experiments}

    trajectory = []
    for position, row in box.peeling_trajectory.iterrows():
        limits = box.box_lims[position]
        quasi_p = box.p_values[position]
        trajectory.append(
            {
                "stats": {
                    "coverage": float(row["coverage"]),
                    "density": float(row["density"]),
                    "mean": float(row["mean"]),
                    "mass": float(row["mass"]),
                    "restricted_dimensions": int(row["res_dim"]),
                    "points": int(row["n"]),
                    "cases_of_interest": int(row["k"]),
                },
                "limits": [
                    serialize_limit(column, kinds[column], limits)
                    for column in experiments.columns
                ],
                "indices": indices_in_box(experiments, limits),
                "quasi_p_values": [
                    {
                        "name": column,
                        "lower": optional_p_value(values[0]),
                        "upper": optional_p_value(values[1]),
                    }
                    for column, values in quasi_p.items()
                ],
            }
        )

    return {
        "name": name,
        "config": {
            "objective": objective.value,
            "peel_alpha": peel_alpha,
            "paste_alpha": paste_alpha,
            "mass_min": mass_min,
        },
        "features": serialize_features(experiments),
        "target": target.tolist(),
        "trajectory": trajectory,
    }


def generate() -> dict[str, Any]:
    continuous_x, continuous_y = continuous_data()
    lenient2_x, lenient2_y = lenient2_data()
    pasting_x, pasting_y = pasting_data()
    mixed_x, mixed_y = mixed_data()
    return {
        "reference": {
            "implementation": "EMA Workbench",
            "package": "ema-workbench",
            "version": ema_workbench.__version__,
        },
        "cases": [
            run_case(
                "continuous_lenient1",
                continuous_x,
                continuous_y,
                objective=prim.PRIMObjectiveFunctions.LENIENT1,
                peel_alpha=0.1,
                paste_alpha=0.1,
                mass_min=0.1,
            ),
            run_case(
                "continuous_lenient2",
                lenient2_x,
                lenient2_y,
                objective=prim.PRIMObjectiveFunctions.LENIENT2,
                peel_alpha=0.1,
                paste_alpha=0.1,
                mass_min=0.1,
            ),
            run_case(
                "continuous_original",
                continuous_x,
                continuous_y,
                objective=prim.PRIMObjectiveFunctions.ORIGINAL,
                peel_alpha=0.1,
                paste_alpha=0.1,
                mass_min=0.15,
            ),
            run_case(
                "continuous_with_pasting",
                pasting_x,
                pasting_y,
                objective=prim.PRIMObjectiveFunctions.LENIENT1,
                peel_alpha=0.1,
                paste_alpha=0.1,
                mass_min=0.1,
            ),
            run_case(
                "mixed_lenient1",
                mixed_x,
                mixed_y,
                objective=prim.PRIMObjectiveFunctions.LENIENT1,
                peel_alpha=0.1,
                paste_alpha=0.1,
                mass_min=0.05,
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(generate(), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
