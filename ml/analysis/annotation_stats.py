"""Compute and format annotation coverage statistics.

Each section emits three artifacts into ``out_dir``:
  * a row in the combined ``annotation_distribution.txt`` report
  * a per-analysis CSV (machine-readable)
  * a per-analysis PNG plot (skipped when ``make_plots=False``)
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from ICD_labels import load_labels_taxonomy

# Headless backend — no display server needed on Windows / CI.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMBINED_REPORT_NAME = "annotation_distribution.txt"


def run_analysis(
    annotation_csv: str,
    labels_csv: str,
    out_dir: str,
    make_plots: bool = True,
) -> None:
    """Compute annotation statistics and write per-analysis + combined artifacts."""
    rows = _load_csv(annotation_csv)
    taxonomy_groups = sorted({label.group for label in load_labels_taxonomy(labels_csv)})

    case_diag_count: Counter = Counter()
    case_group_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        cid = row["case_id"]
        case_diag_count[cid] += 1
        grp = row.get("matched_group", "").strip()
        if grp:
            case_group_counts[cid][grp] += 1

    all_cases = set(case_diag_count.keys())
    n_all = len(all_cases)
    n_labelled = sum(1 for c in all_cases if case_group_counts[c])

    group_case_count: Counter = Counter()
    for grp_counts in case_group_counts.values():
        for grp in grp_counts:
            group_case_count[grp] += 1

    groups_per_case: Counter = Counter(
        len(case_group_counts[c]) for c in all_cases
    )

    n_collision = 0
    collision_max: Counter = Counter()
    for grp_counts in case_group_counts.values():
        if not grp_counts:
            continue
        mx = max(grp_counts.values())
        if mx > 1:
            n_collision += 1
            collision_max[mx] += 1

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    text_blocks: list[str] = []
    text_blocks.append(_section_overview(n_all, n_labelled, taxonomy_groups, group_case_count))

    text_blocks.append(_section_cases_per_group(group_case_count, taxonomy_groups, n_labelled))
    _emit_cases_per_group(out, group_case_count, taxonomy_groups, n_labelled, make_plots)

    text_blocks.append(_section_diag_per_case(case_diag_count, n_all))
    _emit_diag_per_case(out, case_diag_count, n_all, make_plots)

    text_blocks.append(_section_groups_per_case(groups_per_case, n_all))
    _emit_groups_per_case(out, groups_per_case, n_all, make_plots)

    text_blocks.append(_section_collisions(n_all - n_collision, n_collision, n_all, collision_max))
    _emit_collisions(out, collision_max, n_collision, make_plots)

    combined = "\n\n".join(text_blocks) + "\n"
    (out / COMBINED_REPORT_NAME).write_text(combined, encoding="utf-8")
    print(combined, end="")


# ---------------------------------------------------------------------------
# Per-section emitters: write CSV (+ PNG if make_plots)
# ---------------------------------------------------------------------------

def _emit_cases_per_group(
    out: Path,
    group_case_count: Counter,
    taxonomy_groups: list[str],
    n_labelled: int,
    make_plots: bool,
) -> None:
    sorted_groups = sorted(taxonomy_groups, key=lambda g: (-group_case_count[g], g))
    rows = [
        (g, group_case_count[g], _pct(group_case_count[g], n_labelled))
        for g in sorted_groups
    ]
    _write_csv(out / "cases_per_group.csv", ["group", "cases", "pct_of_labelled"], rows)
    if not make_plots:
        return
    # Plot ascending so largest is at the top of the horizontal bar chart.
    plot_groups = [g for g, _, _ in rows][::-1]
    plot_counts = [c for _, c, _ in rows][::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(plot_groups))))
    ax.barh(plot_groups, plot_counts, color="steelblue")
    if max(plot_counts, default=0) > 0:
        ax.set_xscale("log")
    ax.set_xlabel("Cases (log scale)")
    ax.set_title(f"Cases per ICD group  (denominator: {n_labelled:,} labelled)")
    for i, c in enumerate(plot_counts):
        if c > 0:
            ax.text(c, i, f" {c:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cases_per_group.png", dpi=120)
    plt.close(fig)


def _emit_diag_per_case(out: Path, case_diag_count: Counter, n_all: int, make_plots: bool) -> None:
    dist = Counter(case_diag_count.values())
    _write_dist_csv(out / "diagnoses_per_case.csv", "diagnoses", dist, n_all)
    if make_plots:
        _plot_dist_bar(
            out / "diagnoses_per_case.png",
            dist,
            title=f"Diagnoses per case  (n={n_all:,})",
            xlabel="Diagnoses per case",
        )


def _emit_groups_per_case(out: Path, groups_per_case: Counter, n_all: int, make_plots: bool) -> None:
    _write_dist_csv(out / "groups_per_case.csv", "groups", groups_per_case, n_all)
    if make_plots:
        _plot_dist_bar(
            out / "groups_per_case.png",
            groups_per_case,
            title=f"Groups per case  (n={n_all:,}; 0 = no matched group)",
            xlabel="Groups per case",
        )


def _emit_collisions(
    out: Path,
    collision_max: Counter,
    n_collision: int,
    make_plots: bool,
) -> None:
    if n_collision > 0:
        rows = []
        cumulative = 0.0
        for mx in sorted(collision_max.keys()):
            n = collision_max[mx]
            pct = 100 * n / n_collision
            cumulative += pct
            rows.append((mx, n, f"{pct:.1f}", f"{cumulative:.1f}"))
    else:
        rows = []
    _write_csv(
        out / "collisions.csv",
        ["max_shared", "cases", "pct_of_collision_cases", "cumulative_pct"],
        rows,
    )
    if not make_plots:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if collision_max:
        keys = sorted(collision_max.keys())
        vals = [collision_max[k] for k in keys]
        ax.bar([str(k) for k in keys], vals, color="indianred")
        for x, v in zip([str(k) for k in keys], vals):
            ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No same-group collisions", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Max diagnoses sharing one group")
    ax.set_ylabel("Cases")
    ax.set_title(f"Same-group collisions  ({n_collision:,} cases)")
    fig.tight_layout()
    fig.savefig(out / "collisions.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_dist_csv(path: Path, count_col: str, dist: Counter, total: int) -> None:
    rows = []
    cumulative = 0.0
    for count in sorted(dist.keys()):
        n = dist[count]
        pct = 100 * n / total if total else 0.0
        cumulative += pct
        rows.append((count, n, f"{pct:.1f}", f"{cumulative:.1f}"))
    _write_csv(path, [count_col, "cases", "pct_of_all_cases", "cumulative_pct"], rows)


def _plot_dist_bar(path: Path, dist: Counter, title: str, xlabel: str) -> None:
    keys = sorted(dist.keys())
    vals = [dist[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(keys)), 5))
    ax.bar([str(k) for k in keys], vals, color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cases")
    ax.set_title(title)
    if vals and max(vals) > 0:
        ax.set_yscale("log")
        ax.set_ylabel("Cases (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}" if total else "0.0"


def _section_overview(
    n_all: int,
    n_labelled: int,
    taxonomy_groups: list[str],
    group_case_count: Counter,
) -> str:
    n_unlabelled = n_all - n_labelled
    n_tax = len(taxonomy_groups)
    n_covered = sum(1 for g in taxonomy_groups if group_case_count[g] > 0)
    return "\n".join([
        "=== 1. Overview ===",
        f"Total cases:                    {n_all:,}",
        f"Cases with ICD label:           {n_labelled:,}  ({_pct(n_labelled, n_all)}%)",
        f"Cases without ICD label:        {n_unlabelled:,}  ({_pct(n_unlabelled, n_all)}%)",
        "",
        f"Groups in taxonomy:             {n_tax}",
        f"Groups with annotated cases:    {n_covered}  ({_pct(n_covered, n_tax)}%)",
        f"Groups with no annotated cases: {n_tax - n_covered}  ({_pct(n_tax - n_covered, n_tax)}%)",
    ])


def _section_cases_per_group(
    group_case_count: Counter,
    taxonomy_groups: list[str],
    n_labelled: int,
) -> str:
    col_w = max(len(g) for g in taxonomy_groups) + 2
    sorted_groups = sorted(taxonomy_groups, key=lambda g: (-group_case_count[g], g))
    header = f"{'Group':<{col_w}} {'Cases':>8}    {'% of labelled':>13}"
    sep = "-" * (col_w + 26)
    rows = [header, sep]
    for grp in sorted_groups:
        count = group_case_count[grp]
        rows.append(f"{grp:<{col_w}} {count:>8,}    {_pct(count, n_labelled):>13}")
    return "\n".join([
        "=== 2. Cases per ICD group ===",
        f"(denominator: {n_labelled:,} labelled cases)",
        "",
        *rows,
    ])


def _section_diag_per_case(case_diag_count: Counter, n_all: int) -> str:
    dist = Counter(case_diag_count.values())
    return "\n".join([
        "=== 3. Diagnoses per case ===",
        f"Total cases in distribution: {n_all:,}  |  All cases: {n_all:,}",
        "",
        f"{'Count':<10} {'Cases':<10} {'% of all cases':<20} {'Cumulative %'}",
        "-" * 54,
        *_dist_rows(dist, n_all),
    ])


def _section_groups_per_case(groups_per_case: Counter, n_all: int) -> str:
    return "\n".join([
        "=== 4. Groups per case (0 = no matched group) ===",
        f"Total cases in distribution: {n_all:,}  |  All cases: {n_all:,}",
        "",
        f"{'Count':<10} {'Cases':<10} {'% of all cases':<20} {'Cumulative %'}",
        "-" * 54,
        *_dist_rows(groups_per_case, n_all),
    ])


def _section_collisions(
    n_no_collision: int,
    n_collision: int,
    n_all: int,
    collision_max: Counter,
) -> str:
    lines = [
        "=== 5. Cases with multiple diagnoses under the same group ===",
        f"  Cases with NO same-group collision:       {n_no_collision:,}  ({_pct(n_no_collision, n_all)}%)",
        f"  Cases with at least one collision:        {n_collision:,}  ({_pct(n_collision, n_all)}%)",
    ]
    if collision_max:
        lines += [
            "",
            "  Among cases WITH collisions -- max diagnoses sharing one group:",
            f"  {'Max shared':<12} {'Cases':<10} {'% of collision cases':<22} {'Cumulative %'}",
            "  " + "-" * 56,
        ]
        cumulative = 0.0
        for mx in sorted(collision_max.keys()):
            n = collision_max[mx]
            pct = 100 * n / n_collision
            cumulative += pct
            lines.append(f"  {mx:<12} {n:<10,} {pct:<22.1f} {cumulative:.1f}")
    return "\n".join(lines)


def _dist_rows(dist: Counter, total: int) -> list[str]:
    cumulative = 0.0
    rows = []
    for count in sorted(dist.keys()):
        n = dist[count]
        pct = 100 * n / total
        cumulative += pct
        rows.append(f"{count:<10} {n:<10,} {pct:<20.1f} {cumulative:.1f}")
    return rows
