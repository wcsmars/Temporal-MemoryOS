#!/usr/bin/env python3
"""Render benchmark JSON reports as dependency-free SVG charts.

The charts follow the reader's light or dark color scheme and carry their
values as direct labels, so they stay readable without color. Titles and
subtitles take their counts (entities, writes, searches) from the reports.
"""

import argparse
import json
import math
import sys
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
  .dash { stroke-dasharray: 7 4; }
  .dot { stroke: var(--surface); stroke-width: 2; }
</style>
"""

# Vertical space for one end-of-line label (name above, value below).
LABEL_SPACING = 32
# Vertical space for each subtitle line after the first.
SUBTITLE_LINE = 16

# The reports the charts are drawn from, and how to produce each one.
REPORTS = {
    "scale-current.json": "python benchmarks/scale.py "
                          "--output benchmarks/results/scale-current.json",
    "scale-v0.1.json": "see benchmarks/README.md",
    "synthetic-50.json": "python benchmarks/run.py --entities 50 "
                         "--output benchmarks/results/synthetic-50.json",
}
SCALE_METRICS = ("write_ms_median", "warm_search_ms_median", "cold_search_ms_median")
METHOD_LABELS = {
    "all_records_keyword": "Keyword over all versions",
    "latest_state_keyword": "Keyword over latest state",
    "valid_time_only": "Engine, valid time only",
    "temporal_hybrid": "Engine, valid + knowledge time",
}


def _lines(subtitle):
    """A subtitle is one line of text or a sequence of lines."""
    return (subtitle,) if isinstance(subtitle, str) else tuple(subtitle)


def _svg(width, height, title, subtitle, body):
    lines = _lines(subtitle)
    sub = "".join('<text class="sub" x="24" y="{}">{}</text>'.format(
        52 + index * SUBTITLE_LINE, escape(line)) for index, line in enumerate(lines))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'role="img" aria-labelledby="t d">'
        '<title id="t">{title}</title><desc id="d">{desc}</desc>{style}'
        '<rect class="bg" width="{w}" height="{h}" rx="8"/>'
        '<text class="title" x="24" y="32">{title}</text>{sub}{body}</svg>\n'
    ).format(w=width, h=height, title=escape(title), desc=escape(" ".join(lines)), sub=sub,
             style=STYLE, body=body)


def _fmt_ms(value):
    return "{:,.0f} ms".format(value) if value >= 10 else "{:.1f} ms".format(value)


def _spread(positions, spacing, low, high):
    """Move label anchors apart by at least ``spacing``, keeping their order and
    staying between ``low`` and ``high`` where there is room."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    placed = [positions[i] for i in order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + spacing)
    if placed and placed[-1] > high:
        placed[-1] = high
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - spacing)
    if placed and placed[0] < low:
        shift = low - placed[0]
        placed = [value + shift for value in placed]
    result = [0.0] * len(positions)
    for i, value in zip(order, placed):
        result[i] = value
    return result


def line_chart(series, title, subtitle):
    """Log-log line chart of latency against store size.

    ``series`` holds ``(css, name, rows, metric, dashed)`` tuples; each line
    ends in a direct label with its name and last value. Each subtitle line
    after the first moves the plot down and makes the chart taller.
    """
    extra = SUBTITLE_LINE * (len(_lines(subtitle)) - 1)
    width, height = 720, 360 + extra
    left, right, top, bottom = 72, 150, 76 + extra, 300 + extra
    sizes = sorted({row["facts"] for _, _, rows, _, _ in series for row in rows})
    values = [row[metric] for _, _, rows, metric, _ in series for row in rows]
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
    ends = []
    for css, name, rows, metric, dashed in series:
        points = [(x(row["facts"]), y(row[metric])) for row in rows]
        parts.append('<polyline class="line {}{}" points="{}"/>'.format(
            css, " dash" if dashed else "",
            " ".join("{:.1f},{:.1f}".format(*point) for point in points)))
        for (px, py), row in zip(points, rows):
            parts.append('<circle class="dot {}" cx="{:.1f}" cy="{:.1f}" r="5">'
                         '<title>{}: {:,} records, {}</title></circle>'.format(
                             css, px, py, escape(name), row["facts"], _fmt_ms(row[metric])))
        ends.append((points[-1], name, rows[-1][metric]))
    anchors = _spread([point[1] for point, _, _ in ends], LABEL_SPACING, top, bottom)
    for ((last_x, _), name, value), anchor in zip(ends, anchors):
        parts.append('<text class="label" x="{:.1f}" y="{:.1f}">{}</text>'.format(
            last_x + 12, anchor - 2, escape(name)))
        parts.append('<text class="value" x="{:.1f}" y="{:.1f}">{}</text>'.format(
            last_x + 12, anchor + 14, _fmt_ms(value)))
    return _svg(width, height, title, subtitle, "".join(parts))


def scale_chart(current, baseline, metric, title, subtitle):
    """Log-log line chart of one latency metric for the baseline and current versions."""
    return line_chart([("s2", _name(baseline), baseline["results"], metric, False),
                       ("s1", _name(current), current["results"], metric, False)],
                      title, subtitle)


def _name(report):
    return "v" + report["version"]


def _count(reports, key):
    """The value of one run parameter, as recorded in every report."""
    values = {report["parameters"][key] for report in reports}
    return " or ".join(str(value) for value in sorted(values))


def _is_count(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _require(condition, name, problem):
    if not condition:
        raise ValueError("{} cannot be charted: {}. To produce it again: {}".format(
            name, problem, REPORTS[name]))


def _check_scale(report, name):
    """Raise ValueError unless ``report`` holds everything the scale charts read."""
    _require(isinstance(report.get("version"), str), name, "it has no 'version' text")
    parameters = report.get("parameters")
    for key in ("writes", "searches"):
        _require(isinstance(parameters, dict) and _is_count(parameters.get(key), 1), name,
                 "it does not record its '{}' parameter".format(key))
    rows = report.get("results")
    _require(isinstance(rows, list) and rows, name, "'results' is not a non-empty list")
    for row in rows:
        _require(isinstance(row, dict) and _is_count(row.get("facts"), 1), name,
                 "a result has no positive whole 'facts' count")
        for metric in SCALE_METRICS:
            _require(_is_positive(row.get(metric)), name,
                     "the result for {} records has no positive '{}' (the charts use log "
                     "scales)".format(row["facts"], metric))


def _check_accuracy(report, name):
    """Raise ValueError unless ``report`` holds everything the accuracy chart reads."""
    results, corpus = report.get("results"), report.get("corpus")
    _require(isinstance(results, dict) and results, name,
             "'results' does not map workloads to their methods")
    _require(isinstance(corpus, dict), name, "it has no 'corpus' section")
    for workload, methods in results.items():
        for method in METHOD_LABELS:
            result = methods.get(method) if isinstance(methods, dict) else None
            _require(isinstance(result, dict) and _is_count(result.get("queries"), 1)
                     and _is_count(result.get("correct"))
                     and result["correct"] <= result["queries"], name,
                     "'{}' has no 'correct' and 'queries' counts for '{}'".format(
                         workload, method))
        entry = corpus.get(workload)
        _require(isinstance(entry, dict) and _is_count(entry.get("entities"), 1), name,
                 "'corpus' has no 'entities' count for '{}'".format(workload))


def _load(results, name):
    """Read one report from ``results``; raise ValueError if it is missing or not JSON."""
    try:
        report = json.loads((results / name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError("{} was not found. To produce it: {}".format(
            name, REPORTS[name])) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("{} cannot be read as JSON ({})".format(name, error)) from None
    _require(isinstance(report, dict), name, "it is not a JSON object")
    return report


def accuracy_chart(report):
    """Horizontal bars of exact-version hits, one panel per workload."""
    labels = METHOD_LABELS
    panels = list(report["results"].items())
    width, bar_h, gap, panel_gap = 720, 22, 10, 40
    label_w, left = 210, 24
    plot_w = width - left - label_w - 90
    height = 76 + len(panels) * (28 + len(labels) * (bar_h + gap) + panel_gap)
    parts, cursor = [], 84
    names = {"state_changes": "State changes", "late_corrections": "Late corrections"}
    for workload, methods in panels:
        queries = methods["temporal_hybrid"]["queries"]
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
    entities = sorted({report["corpus"][workload]["entities"] for workload, _ in panels})
    if len(entities) == 1:
        scope = "{} entities each".format(entities[0])
    else:
        scope = "{} entities".format(" or ".join(str(count) for count in entities))
    return _svg(width, height, "Exact-version retrieval accuracy",
                "Top-1 result must be the expected record version; synthetic workloads, " + scope,
                "".join(parts))


def charts(results):
    """Render every chart from the reports in ``results``; return {file name: SVG}.

    Raises ValueError, naming the file and the problem, when a report is missing,
    is not JSON or lacks a field the charts need.
    """
    current = _load(results, "scale-current.json")
    baseline = _load(results, "scale-v0.1.json")
    accuracy = _load(results, "synthetic-50.json")
    _check_scale(current, "scale-current.json")
    _check_scale(baseline, "scale-v0.1.json")
    _check_accuracy(accuracy, "synthetic-50.json")
    both = (current, baseline)
    searches = _count(both, "searches")
    return {
        "scale-write.svg": scale_chart(
            current, baseline, "write_ms_median", "Latency of one write",
            "Median of {} writes into an existing store (log scales)".format(
                _count(both, "writes"))),
        "scale-search.svg": scale_chart(
            current, baseline, "warm_search_ms_median", "Latency of one dated search, warm",
            "Median of {} hybrid searches through one reused store and retriever "
            "(log scales)".format(searches)),
        "scale-search-cold.svg": line_chart(
            [("s2", _name(baseline) + " cold", baseline["results"], "cold_search_ms_median", True),
             ("s1", _name(current) + " cold", current["results"], "cold_search_ms_median", True),
             ("s1", _name(current) + " warm", current["results"], "warm_search_ms_median",
              False)],
            "Latency of one dated search, cold and warm",
            ("Cold opens the store and builds a new retriever for each search, like one CLI "
             "call, excluding interpreter start-up.",
             "Warm reuses one store and retriever for every search. Medians of {} searches "
             "each (log scales)".format(searches))),
        "accuracy.svg": accuracy_chart(accuracy),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--check", action="store_true",
                        help="Write nothing; exit with status 1 if a chart differs from the "
                             "reports.")
    args = parser.parse_args(argv)
    try:
        rendered = charts(args.results)
    except ValueError as error:
        parser.exit(2, "{}: error: {}\n".format(parser.prog, error))
    stale = []
    for name, svg in rendered.items():
        path = args.results / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != svg:
                stale.append(name)
            continue
        path.write_text(svg, encoding="utf-8")
        print(path)
    if stale:
        print("Charts out of date (run benchmarks/plot.py): " + ", ".join(stale), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
