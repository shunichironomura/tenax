#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "xy==0.0.5",
# ]
# ///
"""Render Tenax's lake-model PRIM exports with XY.

XY writes self-contained interactive HTML and static PNG versions of all plots.
Its Matplotlib-compatible pyplot layer composes the pairwise scatter-and-box
matrix without depending on Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import xy
import xy.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "target" / "lake_model"
PLOT_WIDTH = 900
PLOT_HEIGHT = 540


class PrimPhase(Enum):
    """Closed set of trajectory phases emitted by the Rust example."""

    INITIAL = "initial"
    PEEL = "peel"
    PASTE = "paste"


@dataclass(frozen=True)
class TrajectoryStep:
    """Statistics for one PRIM trajectory candidate."""

    index: int
    phase: PrimPhase
    coverage: float
    density: float
    mass: float
    restricted_dimensions: int
    points: int
    cases_of_interest: int


@dataclass(frozen=True)
class FeatureLimit:
    """Declared domain and sampled-box interval for one feature."""

    feature: str
    domain_lower: float
    domain_upper: float
    box_lower: float
    box_upper: float


@dataclass(frozen=True)
class Experiment:
    """One sampled model input row and its binary classification."""

    row_index: int
    values: dict[str, float]
    case_of_interest: bool


def parse_args() -> argparse.Namespace:
    """Parse paths and the optional trajectory candidate to inspect."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="directory written by the Rust example (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="plot directory (default: INPUT/plots)",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="trajectory step to inspect (default: final step)",
    )
    return parser.parse_args()


def read_trajectory(path: Path) -> list[TrajectoryStep]:
    """Load and validate trajectory statistics."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            TrajectoryStep(
                index=int(row["step"]),
                phase=PrimPhase(row["phase"]),
                coverage=float(row["coverage"]),
                density=float(row["density"]),
                mass=float(row["mass"]),
                restricted_dimensions=int(row["restricted_dimensions"]),
                points=int(row["points"]),
                cases_of_interest=int(row["cases_of_interest"]),
            )
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise ValueError(f"{path} contains no PRIM trajectory steps")
    expected = list(range(len(rows)))
    actual = [row.index for row in rows]
    if actual != expected:
        raise ValueError(f"{path} step indices are not contiguous: {actual}")
    return rows


def read_limits(path: Path) -> dict[int, list[FeatureLimit]]:
    """Load feature limits grouped by typed trajectory index."""

    grouped: dict[int, list[FeatureLimit]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            grouped.setdefault(step, []).append(
                FeatureLimit(
                    feature=row["feature"],
                    domain_lower=float(row["domain_lower"]),
                    domain_upper=float(row["domain_upper"]),
                    box_lower=float(row["box_lower"]),
                    box_upper=float(row["box_upper"]),
                )
            )
    if not grouped:
        raise ValueError(f"{path} contains no PRIM limits")
    return grouped


def parse_bool(value: str) -> bool:
    """Parse Rust's explicit CSV boolean representation."""

    match value:
        case "true":
            return True
        case "false":
            return False
        case _:
            raise ValueError(f"expected 'true' or 'false', got {value!r}")


def read_experiments(path: Path) -> list[Experiment]:
    """Load sampled inputs without assigning semantics to feature names."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        feature_names = [
            name
            for name in reader.fieldnames
            if name not in {"row_index", "case_of_interest"}
        ]
        rows = [
            Experiment(
                row_index=int(row["row_index"]),
                values={name: float(row[name]) for name in feature_names},
                case_of_interest=parse_bool(row["case_of_interest"]),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} contains no experiments")
    return rows


def select_step(
    trajectory: Sequence[TrajectoryStep], requested: int | None
) -> TrajectoryStep:
    """Select one candidate, defaulting to PRIM's final candidate."""

    index = trajectory[-1].index if requested is None else requested
    if not 0 <= index < len(trajectory):
        raise ValueError(f"step {index} is outside [0, {len(trajectory) - 1}]")
    return trajectory[index]


def limit_map(limits: Iterable[FeatureLimit]) -> dict[str, FeatureLimit]:
    """Index opaque feature names after checking uniqueness."""

    indexed: dict[str, FeatureLimit] = {}
    for limit in limits:
        if limit.feature in indexed:
            raise ValueError(f"duplicate limit for feature {limit.feature!r}")
        indexed[limit.feature] = limit
    return indexed


def restricted_feature_limits(
    initial_limits: Sequence[FeatureLimit],
    selected_limits: Sequence[FeatureLimit],
    selected: TrajectoryStep,
) -> list[FeatureLimit]:
    """Return selected limits that differ exactly from the initial box."""

    initial = limit_map(initial_limits)
    selected_by_feature = limit_map(selected_limits)
    if initial.keys() != selected_by_feature.keys():
        raise ValueError("initial and selected boxes contain different features")
    restricted = [
        selected_by_feature[feature]
        for feature, source in initial.items()
        if (
            selected_by_feature[feature].box_lower != source.box_lower
            or selected_by_feature[feature].box_upper != source.box_upper
        )
    ]
    if len(restricted) != selected.restricted_dimensions:
        raise ValueError(
            f"step {selected.index} reports {selected.restricted_dimensions} "
            f"restricted dimensions, but its limits imply {len(restricted)}"
        )
    return restricted


def is_box_member(experiment: Experiment, limits: Sequence[FeatureLimit]) -> bool:
    """Apply all inclusive continuous limits to one experiment."""

    return all(
        limit.box_lower <= experiment.values[limit.feature] <= limit.box_upper
        for limit in limits
    )


def validate_membership(
    selected: TrajectoryStep,
    limits: Sequence[FeatureLimit],
    experiments: Sequence[Experiment],
) -> list[bool]:
    """Cross-check exported statistics against membership implied by limits."""

    membership = [is_box_member(experiment, limits) for experiment in experiments]
    points = sum(membership)
    cases = sum(
        member and experiment.case_of_interest
        for member, experiment in zip(membership, experiments, strict=True)
    )
    if points != selected.points or cases != selected.cases_of_interest:
        raise ValueError(
            "selected limits disagree with trajectory statistics: "
            f"limits imply {points} points/{cases} cases, statistics report "
            f"{selected.points} points/{selected.cases_of_interest} cases"
        )
    return membership


def chart_theme() -> xy.Theme:
    """Return one shared XY theme for deterministic exports."""

    return xy.theme(
        plot_background="#ffffff",
        grid_color="#e5e7eb",
        axis_color="#6b7280",
        text_color="#111827",
        selection_color="#7c3aed",
        selection_fill="#7c3aed33",
    )


def tradeoff_chart(
    trajectory: Sequence[TrajectoryStep], selected: TrajectoryStep
) -> xy.Chart:
    """Build the coverage-density trade-off chart."""

    coverage = [step.coverage for step in trajectory]
    density = [step.density for step in trajectory]
    mass = [step.mass for step in trajectory]
    return xy.scatter_chart(
        xy.line(
            coverage,
            density,
            name="PRIM trajectory",
            color="#64748b",
            width=2.0,
        ),
        xy.scatter(
            coverage,
            density,
            color=mass,
            color_domain=(0.0, 1.0),
            colormap="viridis",
            size=6.0,
            name="Candidates (color = mass)",
            opacity=0.9,
            stroke="#ffffff",
            stroke_width=0.7,
        ),
        xy.scatter(
            [selected.coverage],
            [selected.density],
            color="#dc2626",
            size=13.0,
            symbol="star",
            name=f"Selected step {selected.index}",
            stroke="#7f1d1d",
            stroke_width=1.0,
        ),
        xy.x_axis(
            label="Coverage",
            domain=(0.0, 1.02),
            tick_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            tick_labels=["0%", "20%", "40%", "60%", "80%", "100%"],
        ),
        xy.y_axis(
            label="Density",
            domain=(0.0, 1.02),
            tick_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            tick_labels=["0%", "20%", "40%", "60%", "80%", "100%"],
        ),
        xy.legend(),
        chart_theme(),
        title="Lake model PRIM coverage-density trade-off",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
    )


def split_coordinates(
    experiments: Sequence[Experiment],
    membership: Sequence[bool],
    predicate: Callable[[Experiment, bool], bool],
) -> tuple[list[float], list[float]]:
    """Select b/q coordinates for one visual layer."""

    selected = [
        experiment
        for experiment, member in zip(experiments, membership, strict=True)
        if predicate(experiment, member)
    ]
    return (
        [experiment.values["b"] for experiment in selected],
        [experiment.values["q"] for experiment in selected],
    )


def experiment_chart(
    experiments: Sequence[Experiment],
    membership: Sequence[bool],
    selected: TrajectoryStep,
    selected_limits: Sequence[FeatureLimit],
) -> xy.Chart:
    """Build a b/q projection with selected-box members emphasized."""

    undesirable_x, undesirable_y = split_coordinates(
        experiments,
        membership,
        lambda experiment, member: not experiment.case_of_interest and not member,
    )
    desirable_x, desirable_y = split_coordinates(
        experiments,
        membership,
        lambda experiment, member: experiment.case_of_interest and not member,
    )
    selected_other_x, selected_other_y = split_coordinates(
        experiments,
        membership,
        lambda experiment, member: not experiment.case_of_interest and member,
    )
    selected_case_x, selected_case_y = split_coordinates(
        experiments,
        membership,
        lambda experiment, member: experiment.case_of_interest and member,
    )
    limits = limit_map(selected_limits)
    b_limit = limits["b"]
    q_limit = limits["q"]
    boundary_x = [
        b_limit.box_lower,
        b_limit.box_upper,
        b_limit.box_upper,
        b_limit.box_lower,
        b_limit.box_lower,
    ]
    boundary_y = [
        q_limit.box_lower,
        q_limit.box_lower,
        q_limit.box_upper,
        q_limit.box_upper,
        q_limit.box_lower,
    ]

    return xy.scatter_chart(
        xy.scatter(
            undesirable_x,
            undesirable_y,
            color="#cbd5e1",
            size=3.0,
            opacity=0.45,
            name="Other outcomes",
        ),
        xy.scatter(
            desirable_x,
            desirable_y,
            color="#2563eb",
            size=4.0,
            opacity=0.75,
            name="Cases of interest",
        ),
        xy.scatter(
            selected_other_x,
            selected_other_y,
            color="#f97316",
            size=7.0,
            opacity=0.95,
            name="Selected non-cases",
            stroke="#9a3412",
            stroke_width=0.7,
        ),
        xy.scatter(
            selected_case_x,
            selected_case_y,
            color="#16a34a",
            size=7.0,
            opacity=0.95,
            name="Selected cases",
            stroke="#14532d",
            stroke_width=0.7,
        ),
        xy.line(
            boundary_x,
            boundary_y,
            color="#dc2626",
            width=2.5,
            name="b/q projection of box",
        ),
        xy.x_axis(
            label="Lake phosphorus removal rate (b)",
            domain=(b_limit.domain_lower, b_limit.domain_upper),
        ),
        xy.y_axis(
            label="Lake phosphorus recycling exponent (q)",
            domain=(q_limit.domain_lower, q_limit.domain_upper),
        ),
        xy.legend(loc="lower left"),
        chart_theme(),
        title=(
            f"Lake experiments and PRIM step {selected.index} "
            f"({selected.points} rows, density {selected.density:.1%})"
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
    )


def normalized(value: float, lower: float, upper: float) -> float:
    """Map a value onto a non-degenerate sampled interval."""

    if not lower < upper:
        raise ValueError(f"cannot normalize degenerate interval [{lower}, {upper}]")
    return (value - lower) / (upper - lower)


def limits_chart(
    initial_limits: Sequence[FeatureLimit],
    selected_limits: Sequence[FeatureLimit],
    selected: TrajectoryStep,
) -> xy.Chart:
    """Build an EMA-style normalized view of restricted dimensions."""

    initial = limit_map(initial_limits)
    selected_by_feature = limit_map(selected_limits)
    feature_names = list(initial)
    positions = list(range(len(feature_names)))
    children: list[xy.Component] = []
    restricted_positions: list[int] = []
    restricted_lower: list[float] = []
    restricted_upper: list[float] = []

    for position, feature in enumerate(feature_names):
        source = initial[feature]
        chosen = selected_by_feature[feature]
        children.append(
            xy.line(
                [position, position],
                [0.0, 1.0],
                color="#d1d5db",
                width=4.0,
            )
        )
        restricted = (
            chosen.box_lower != source.box_lower or chosen.box_upper != source.box_upper
        )
        if restricted:
            lower = normalized(chosen.box_lower, source.box_lower, source.box_upper)
            upper = normalized(chosen.box_upper, source.box_lower, source.box_upper)
            children.append(
                xy.line(
                    [position, position],
                    [lower, upper],
                    color="#2563eb",
                    width=8.0,
                )
            )
            restricted_positions.extend([position, position])
            restricted_lower.append(lower)
            restricted_upper.append(upper)

    children.extend(
        [
            xy.scatter(
                restricted_positions,
                [
                    value
                    for pair in zip(restricted_lower, restricted_upper, strict=True)
                    for value in pair
                ],
                color="#1d4ed8",
                size=7.0,
                symbol="circle",
                stroke="#ffffff",
                stroke_width=0.8,
            ),
            xy.x_axis(
                label="Input feature",
                domain=(-0.5, len(feature_names) - 0.5),
                tick_values=positions,
                tick_labels=feature_names,
            ),
            xy.y_axis(
                label="Fraction of sampled range",
                domain=(-0.03, 1.03),
                tick_values=[0.0, 0.25, 0.5, 0.75, 1.0],
                tick_labels=["0%", "25%", "50%", "75%", "100%"],
            ),
            chart_theme(),
        ]
    )
    return xy.line_chart(
        *children,
        title=(
            f"PRIM step {selected.index} limits "
            f"({selected.restricted_dimensions} restricted dimensions in blue)"
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
    )


def padded_interval(limit: FeatureLimit) -> tuple[float, float]:
    """Add a small visual margin to one non-degenerate sampled interval."""

    width = limit.domain_upper - limit.domain_lower
    if not width > 0.0:
        raise ValueError(
            f"feature {limit.feature!r} has a degenerate declared domain "
            f"[{limit.domain_lower}, {limit.domain_upper}]"
        )
    padding = 0.04 * width
    return limit.domain_lower - padding, limit.domain_upper + padding


def export_pairs_figure(figure: plt.Figure, path: Path) -> None:
    """Write the composed XY matrix as PNG and self-contained HTML."""

    try:
        figure.savefig(path, dpi=140, facecolor="white")
        figure.savefig(path.with_suffix(".html"))
    finally:
        plt.close(figure)


def export_pairs_scatter(
    path: Path,
    experiments: Sequence[Experiment],
    initial_limits: Sequence[FeatureLimit],
    selected_limits: Sequence[FeatureLimit],
    selected: TrajectoryStep,
) -> None:
    """Compose an EMA-style pairwise matrix with XY's pyplot layer."""

    restricted = restricted_feature_limits(initial_limits, selected_limits, selected)
    if not restricted:
        figure, axis = plt.subplots(figsize=(7.0, 2.5), facecolor="white")
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            f"PRIM step {selected.index} has no restricted dimensions",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        export_pairs_figure(figure, path)
        return
    initial = limit_map(initial_limits)
    non_cases = [
        experiment for experiment in experiments if not experiment.case_of_interest
    ]
    cases = [experiment for experiment in experiments if experiment.case_of_interest]
    panel_count = len(restricted)
    figure, axes = plt.subplots(
        panel_count,
        panel_count,
        figsize=(3.1 * panel_count + 0.8, 3.0 * panel_count + 1.0),
        facecolor="white",
        squeeze=False,
    )
    non_case_color = "#1f77b4"
    case_color = "#ff7f0e"
    box_color = "#dc2626"
    legend_handles = []

    for row, y_limit in enumerate(restricted):
        for column, x_limit in enumerate(restricted):
            axis = axes[row][column]
            x_domain = padded_interval(x_limit)
            axis.set_xlim(*x_domain)
            axis.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
            axis.set_axisbelow(True)

            if row == column:
                source = initial[x_limit.feature]
                width = source.box_upper - source.box_lower
                if not width > 0.0:
                    raise ValueError(
                        f"feature {source.feature!r} has a degenerate sampled range"
                    )
                bins = [source.box_lower + width * index / 30 for index in range(31)]
                non_case_hist = axis.hist(
                    [experiment.values[x_limit.feature] for experiment in non_cases],
                    bins=bins,
                    # An absent class has zero bars, not an undefined density.
                    density=bool(non_cases),
                    color=non_case_color,
                    alpha=0.28,
                    edgecolor=non_case_color,
                    linewidth=0.7,
                )
                case_hist = axis.hist(
                    [experiment.values[x_limit.feature] for experiment in cases],
                    bins=bins,
                    density=bool(cases),
                    color=case_color,
                    alpha=0.35,
                    edgecolor=case_color,
                    linewidth=0.8,
                )
                axis.axvspan(
                    x_limit.box_lower,
                    x_limit.box_upper,
                    color=box_color,
                    alpha=0.07,
                )
                lower_boundary = axis.axvline(
                    x_limit.box_lower, color=box_color, linewidth=1.6
                )
                if row == 0:
                    legend_handles = [non_case_hist[2], case_hist[2], lower_boundary]
                axis.axvline(x_limit.box_upper, color=box_color, linewidth=1.6)
                axis.set_ylim(bottom=0.0)
            else:
                axis.scatter(
                    [experiment.values[x_limit.feature] for experiment in non_cases],
                    [experiment.values[y_limit.feature] for experiment in non_cases],
                    s=11.0,
                    color=non_case_color,
                    alpha=0.48,
                    edgecolors="white",
                    linewidths=0.2,
                )
                axis.scatter(
                    [experiment.values[x_limit.feature] for experiment in cases],
                    [experiment.values[y_limit.feature] for experiment in cases],
                    s=15.0,
                    color=case_color,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.25,
                )
                # A closed data-space line needs no Matplotlib Rectangle artist.
                axis.plot(
                    [
                        x_limit.box_lower,
                        x_limit.box_upper,
                        x_limit.box_upper,
                        x_limit.box_lower,
                        x_limit.box_lower,
                    ],
                    [
                        y_limit.box_lower,
                        y_limit.box_lower,
                        y_limit.box_upper,
                        y_limit.box_upper,
                        y_limit.box_lower,
                    ],
                    color=box_color,
                    linewidth=2.2,
                )
                axis.set_ylim(*padded_interval(y_limit))

            if row == panel_count - 1:
                axis.set_xlabel(x_limit.feature)
            else:
                axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel("Density" if row == column else y_limit.feature)
            else:
                axis.tick_params(labelleft=False)

    figure.suptitle(
        f"Lake model PRIM step {selected.index}: pairwise restricted dimensions",
        y=0.995,
    )
    figure.legend(
        handles=legend_handles,
        labels=["Not of interest", "Case of interest", "Selected box projection"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    export_pairs_figure(figure, path)


def export_chart(chart: xy.Chart, stem: Path) -> None:
    """Write one interactive HTML chart and one deterministic static PNG."""

    chart.to_html(stem.with_suffix(".html"))
    chart.to_png(str(stem.with_suffix(".png")), scale=2.0)


def main() -> None:
    """Load Rust exports, validate them, and render all analysis charts."""

    args = parse_args()
    output = args.output or args.input / "plots"
    trajectory = read_trajectory(args.input / "trajectory.csv")
    limits = read_limits(args.input / "limits.csv")
    experiments = read_experiments(args.input / "experiments.csv")
    selected = select_step(trajectory, args.step)
    if selected.index not in limits or 0 not in limits:
        raise ValueError(f"limits are missing step 0 or selected step {selected.index}")
    selected_limits = limits[selected.index]
    membership = validate_membership(selected, selected_limits, experiments)

    output.mkdir(parents=True, exist_ok=True)
    export_chart(tradeoff_chart(trajectory, selected), output / "prim_tradeoff")
    export_chart(
        experiment_chart(experiments, membership, selected, selected_limits),
        output / "experiments_b_q",
    )
    export_chart(
        limits_chart(limits[0], selected_limits, selected),
        output / "selected_box_limits",
    )
    export_pairs_scatter(
        output / "prim_pairs_scatter.png",
        experiments,
        limits[0],
        selected_limits,
        selected,
    )
    print(f"Plots for PRIM step {selected.index} written to {output}")


if __name__ == "__main__":
    main()
