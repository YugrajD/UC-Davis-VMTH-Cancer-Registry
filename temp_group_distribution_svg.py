import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


SUMMARY_PATH = Path("ml/output/annotation/llm/llm_summary.json")
ANNOTATION_PATH = Path("ml/output/annotation/llm/llm_annotation.csv")
ROW_OUTPUT_PATH = Path("ml/output/annotation/llm/group_distribution.svg")
CASE_OUTPUT_PATH = Path("ml/output/annotation/llm/group_distribution_cases.svg")


def render_chart(items, title, subtitle, output_path) -> None:
    label_width = 360
    chart_width = 1180
    bar_height = 18
    bar_gap = 8
    top_margin = 90
    bottom_margin = 80
    left_margin = 30
    right_margin = 220
    usable_width = chart_width - label_width - left_margin - right_margin
    max_count = max(count for _, count, _ in items) if items else 1
    height = top_margin + bottom_margin + len(items) * (bar_height + bar_gap)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_width}" height="{height}" viewBox="0 0 {chart_width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #4b5563; }",
        ".legend-text { font-size: 12px; font-weight: 600; fill: #111827; }",
        ".label { font-size: 12px; }",
        ".value { font-size: 12px; font-weight: 600; }",
        ".bar { fill: #2563eb; }",
        ".threshold-50 { stroke: #f59e0b; stroke-width: 2; stroke-dasharray: 5 4; }",
        ".threshold-100 { stroke: #ef4444; stroke-width: 2; stroke-dasharray: 5 4; }",
        ".threshold-200 { stroke: #7c3aed; stroke-width: 2; stroke-dasharray: 5 4; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{left_margin}" y="32" class="title">{escape(title)}</text>',
        f'<text x="{left_margin}" y="52" class="subtitle">Source: {escape(str(SUMMARY_PATH))}</text>',
        f'<text x="{left_margin}" y="68" class="subtitle">{escape(subtitle)}</text>',
    ]

    thresholds = [
        (50, "threshold-50", "#f59e0b"),
        (100, "threshold-100", "#ef4444"),
        (200, "threshold-200", "#7c3aed"),
    ]
    for value, css_class, color in thresholds:
        x = left_margin + label_width + (usable_width * value / max_count)
        if x <= left_margin + label_width + usable_width:
            parts.append(
                f'<line x1="{x:.1f}" y1="{top_margin - 12}" x2="{x:.1f}" y2="{height - bottom_margin}" class="{css_class}" />'
            )
            parts.append(
                f'<text x="{x + 4:.1f}" y="{height - bottom_margin + 18}" class="subtitle" fill="{color}">{value}</text>'
            )

    for index, (name, count, pct) in enumerate(items):
        y = top_margin + index * (bar_height + bar_gap)
        bar_width = 0 if max_count == 0 else usable_width * count / max_count
        bar_x = left_margin + label_width
        text_y = y + bar_height - 4
        parts.append(
            f'<text x="{left_margin + label_width - 10}" y="{text_y}" text-anchor="end" class="label">{escape(name)}</text>'
        )
        parts.append(
            f'<rect x="{bar_x}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" rx="3" class="bar" />'
        )
        value_x = min(bar_x + bar_width + 8, chart_width - right_margin + 20)
        parts.append(
            f'<text x="{value_x:.2f}" y="{text_y}" class="value">{count} ({pct:.2f}%)</text>'
        )

    legend_y = height - 36
    parts.extend(
        [
            f'<rect x="{left_margin - 8}" y="{legend_y - 13}" width="760" height="24" rx="6" fill="#f3f4f6" stroke="#d1d5db" />',
            f'<line x1="{left_margin + 8}" y1="{legend_y}" x2="{left_margin + 32}" y2="{legend_y}" class="threshold-50" />',
            f'<text x="{left_margin + 38}" y="{legend_y + 4}" class="legend-text">50 entries: bare minimum</text>',
            f'<line x1="{left_margin + 220}" y1="{legend_y}" x2="{left_margin + 244}" y2="{legend_y}" class="threshold-100" />',
            f'<text x="{left_margin + 250}" y="{legend_y + 4}" class="legend-text">100 entries: more stable</text>',
            f'<line x1="{left_margin + 430}" y1="{legend_y}" x2="{left_margin + 454}" y2="{legend_y}" class="threshold-200" />',
            f'<text x="{left_margin + 460}" y="{legend_y + 4}" class="legend-text">200 entries: strong support</text>',
        ]
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {output_path}")


def build_row_items(summary_data):
    groups = summary_data["group_distribution"]
    return sorted(
        ((name, values["count"], values["pct"]) for name, values in groups.items()),
        key=lambda item: (-item[1], item[0].lower()),
    )


def build_case_items(summary_data):
    group_cases = {}
    with ANNOTATION_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group = (row.get("matched_group") or "").strip()
            case_id = (row.get("case_id") or "").strip()
            if not group or not case_id:
                continue
            group_cases.setdefault(group, set()).add(case_id)

    all_groups = list(summary_data["group_distribution"].keys())
    total_cases = len({case_id for case_ids in group_cases.values() for case_id in case_ids}) or 1
    items = []
    for group in all_groups:
        case_count = len(group_cases.get(group, set()))
        items.append((group, case_count, case_count * 100 / total_cases))
    return sorted(items, key=lambda item: (-item[1], item[0].lower())), total_cases


def main() -> None:
    summary_data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    render_chart(
        build_row_items(summary_data),
        "Matched entries per ICD group",
        "Sorted by matched entry count from llm_summary.json",
        ROW_OUTPUT_PATH,
    )

    case_items, total_cases = build_case_items(summary_data)
    render_chart(
        case_items,
        "Matched cases per ICD group",
        f"Unique matched cases per group from llm_annotation.csv (denominator: {total_cases} matched cases)",
        CASE_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
