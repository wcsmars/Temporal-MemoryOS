#!/usr/bin/env python3
"""Render benchmark JSON reports as dependency-free SVG charts.

The charts follow the reader's light or dark color scheme and carry their
values as direct labels, so they stay readable without color.
"""

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

RESULTS = Path(__file__).resolve().parent / "results"

STYLE = """
<style>
  svg { --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #8a8984;
        --grid: #e4e3df; --s1: #2a78d6; --s2: #eb6834; }
  @media (prefers-color-scheme: dark) {
    svg { --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #8f8e88;
          --grid: #33332f; --s1: #3987e5; --s2: #d95926; }
  }
  .bg { fill: var(--surface); }
  .title { fill: var(--ink); font: 600 15px system-ui, -apple-system, sans-serif; }
  .sub { fill: var(--ink-2); font: 12px system-ui, -apple-system, sans-serif; }
  .tick { fill: var(--muted); font: 11px system-ui, -apple-system, sans-serif; }
  .label { fill: var(--ink); font: 600 12px system-ui, -apple-system, sans-serif; }
  .value { fill: var(--ink-2); font: 12px system-ui, -apple-system, sans-serif; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--muted); stroke-width: 1; }
  .s1 { stroke: var(--s1); fill: var(--s1); }
  .s2 { stroke: var(--s2); fill: var(--s2); }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; }
  .dot { stroke: var(--surface); stroke-width: 2; }
</style>
"""


def _svg(width, height, title, subtitle, body):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'role="img" aria-labelledby="t d">'
        '<title id="t">{title}</title><desc id="d">{sub}</desc>{style}'
        '<rect class="bg" width="{w}" height="{h}" rx="8"/>'
        '<text class="title" x="24" y="32">{title}</text>'
        '<text class="sub" x="24" y="52">{sub}</text>{body}</svg>\n'
    ).format(w=width, h=height, title=escape(title), sub=escape(subtitle), style=STYLE, body=body)


def _fmt_ms(value):
    return "{:,.0f} ms".format(value) if value >= 10 else "{:.1f} ms".format(value)


def scale_chart(current, baseline, metric, title, subtitle):
    """Log-log line chart of one latency metric against store size."""
    width, height = 720, 360
    left, right, top, bottom = 72, 150, 76, 300
    series = [("s2", "v" + baseline["version"], baseline["results"]),
              ("s1", "v" + current["version"], current["results"])]
    sizes = sorted({row["facts"] for _, _, rows in series for row in rows})
    values = [row[metric] for _, _, rows in series for row in rows]
    low = 10 ** math.floor(math.log10(min(values)))
    high = 10 ** math.ceil(math.log10(max(values)))

    def x(size):
        span = math.log10(sizes[-1]) - math.log10(sizes[0]) or 1
        return left + (math.log10(size) - math.log10(sizes[0])) / span * (width - left - right)

    def y(value):
        span = math.log10(high) - math.log10(low)
        return bottom - (math.log10(value) - math.log10(low)) / span * (bottom - top)

    parts = []
    decade = low
    while decade <= high:
        parts.append('<line class="grid" x1="{}" x2="{}" y1="{y:.1f}" y2="{y:.1f}"/>'.format(
            left, width - right, y=y(decade)))
        parts.append('<text class="tick" x="{}" y="{:.1f}" text-anchor="end">{}</text>'.format(
            left - 8, y(decade) + 4, _fmt_ms(decade)))
        decade *= 10
    for size in sizes:
        parts.append('<text class="tick" x="{:.1f}" y="{}" text-anchor="middle">{:,}</text>'.format(
            x(size), bottom + 20, size))
    parts.append('<text class="tick" x="{:.1f}" y="{}" text-anchor="middle">records in the store</text>'.format(
        (left + width - right) / 2, bottom + 40))
    parts.append('<line class="axis" x1="{l}" x2="{r}" y1="{b}" y2="{b}"/>'.format(
        l=left, r=width - right, b=bottom))
    for css, name, rows in series:
        points = [(x(row["facts"]), y(row[metric])) for row in rows]
        parts.append('<polyline class="line {}" points="{}"/>'.format(
            css, " ".join("{:.1f},{:.1f}".format(*point) for point in points)))
        for (px, py), row in zip(points, rows):
            parts.append('<circle class="dot {}" cx="{:.1f}" cy="{:.1f}" r="5">'
                         '<title>{}: {:,} records, {}</title></circle>'.format(
                             css, px, py, escape(name), row["facts"], _fmt_ms(row[metric])))
        last_x, last_y = points[-1]
        parts.append('<text class="label" x="{:.1f}" y="{:.1f}">{}</text>'.format(
            last_x + 12, last_y - 2, escape(name)))
        parts.append('<text class="value" x="{:.1f}" y="{:.1f}">{}</text>'.format(
            last_x + 12, last_y + 14, _fmt_ms(rows[-1][metric])))
    return _svg(width, height, title, subtitle, "".join(parts))


def accuracy_chart(report):
    """Horizontal bars of exact-version hits, one panel per workload."""
    labels = {
        "all_records_keyword": "Keyword over all versions",
        "latest_state_keyword": "Keyword over latest state",
        "valid_time_only": "Engine, valid time only",
        "temporal_hybrid": "Engine, valid + knowledge time",
    }
    panels = list(report["results"].items())
    width, bar_h, gap, panel_gap = 720, 22, 10, 40
    label_w, left = 210, 24
    plot_w = width - left - label_w - 90
    height = 76 + len(panels) * (28 + len(labels) * (bar_h + gap) + panel_gap)
    parts, cursor = [], 84
    names = {"state_changes": "State changes", "late_corrections": "Late corrections"}
    for workload, methods in panels:
        queries = next(iter(methods.values()))["queries"]
        parts.append('<text class="label" x="{}" y="{}">{} ({} dated queries)</text>'.format(
            left, cursor, escape(names.get(workload, workload)), queries))
        cursor += 14
        for method, title in labels.items():
            result = methods[method]
            share = result["correct"] / result["queries"]
            bar = max(2.0, share * plot_w)
            x0 = left + label_w
            parts.append('<text class="value" x="{}" y="{}">{}</text>'.format(
                left, cursor + bar_h - 6, escape(title)))
            parts.append('<rect class="grid" x="{}" y="{}" width="{}" height="{}" rx="4" '
                         'style="fill: var(--grid); stroke: none"/>'.format(x0, cursor, plot_w, bar_h))
            css = "s1" if method == "temporal_hybrid" else "s2" if method == "valid_time_only" else ""
            fill = "" if css else ' style="fill: var(--muted)"'
            parts.append('<rect class="{}" x="{}" y="{}" width="{:.1f}" height="{}" rx="4"{}>'
                         '<title>{}: {} of {}</title></rect>'.format(
                             css, x0, cursor, bar, bar_h, fill, escape(title),
                             result["correct"], result["queries"]))
            parts.append('<text class="label" x="{:.1f}" y="{}">{:.0%}</text>'.format(
                x0 + plot_w + 10, cursor + bar_h - 6, share))
            cursor += bar_h + gap
        cursor += panel_gap
    return _svg(width, height, "Exact-version retrieval accuracy",
                "Top-1 result must be the expected record version; synthetic workloads, 50 entities each",
                "".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    current = json.loads((args.results / "scale-current.json").read_text(encoding="utf-8"))
    baseline = json.loads((args.results / "scale-v0.1.json").read_text(encoding="utf-8"))
    charts = {
        "scale-write.svg": scale_chart(
            current, baseline, "write_ms_median", "Latency of one write",
            "Median of 20 writes into an existing store (log scales)"),
        "scale-search.svg": scale_chart(
            current, baseline, "search_ms_median", "Latency of one dated search",
            "Median of 20 hybrid searches with valid- and knowledge-time filters (log scales)"),
        "accuracy.svg": accuracy_chart(json.loads(
            (args.results / "synthetic-50.json").read_text(encoding="utf-8"))),
    }
    for name, svg in charts.items():
        (args.results / name).write_text(svg, encoding="utf-8")
        print(args.results / name)


if __name__ == "__main__":
    main()
