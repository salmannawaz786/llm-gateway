"""Chaos benchmark runner.

    python -m chaos.run

Runs every scenario, prints a table, and writes `chaos/results/` --
`results.json` plus an SVG chart that drops straight into the README.

The chart is hand-generated SVG rather than matplotlib on purpose: it keeps the
benchmark dependency-free, and SVG renders natively on GitHub in both light and
dark themes, which a PNG does not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path

import structlog

from chaos.scenarios import (
    Outcome,
    scenario_degraded_provider,
    scenario_retry_storm,
    scenario_tail_latency,
)

RESULTS_DIR = Path(__file__).parent / "results"


def quiet_logs() -> None:
    """Silence the gateway's own logging during benchmarks.

    The executor logs every failover and hedge, which is exactly what you want
    in production and exactly what you do not want scrolling past the results
    table you are trying to read.
    """
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))


def _fmt_ms(v: float) -> str:
    """Latency percentiles are undefined when nothing succeeded."""
    return "-" if math.isnan(v) else f"{v:.0f}ms"


def _json_num(v: float) -> float | None:
    """JSON has no NaN literal.

    `json.dumps` will happily emit a bare `NaN`, which every strict parser then
    rejects. Percentiles over zero successful requests must serialise as null.
    """
    return None if math.isnan(v) else round(v, 1)


SCENARIOS = [
    (
        "Degraded provider (30% error rate)",
        "success_rate",
        scenario_degraded_provider,
    ),
    (
        "Tail latency: p95 (10% of requests take 2s)",
        "p95",
        scenario_tail_latency,
    ),
    (
        "Total outage: load inflicted upstream",
        "upstream",
        scenario_retry_storm,
    ),
]


def print_table(title: str, outcomes: list[Outcome]) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print(f"  {'':<34} {'success':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'upstream':>9}")
    for o in outcomes:
        print(
            f"  {o.label:<34} {o.success_rate:>8.1%} "
            f"{_fmt_ms(o.pct(0.50)):>9} {_fmt_ms(o.pct(0.95)):>9} "
            f"{_fmt_ms(o.pct(0.99)):>9} {o.upstream_calls:>9}"
        )


def bar_chart_svg(groups: list[tuple[str, list[tuple[str, float, str]]]]) -> str:
    """Render grouped horizontal bars.

    Each group is (title, [(label, normalised_0_to_1, display_value)]).
    Colours are chosen to be legible on both GitHub themes -- no fills that
    vanish against a dark background.
    """
    row_h, group_gap, pad, bar_w = 34, 46, 16, 380
    label_w = 250
    height = pad * 2 + sum(group_gap + row_h * len(rows) for _, rows in groups)
    width = pad * 2 + label_w + bar_w + 70

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">'
    ]
    y = pad

    for title, rows in groups:
        parts.append(
            f'<text x="{pad}" y="{y + 18}" font-size="14" font-weight="700" '
            f'fill="#3b82f6">{title}</text>'
        )
        y += group_gap
        for i, (label, frac, display) in enumerate(rows):
            # First row in each group is the baseline (grey), second is the
            # gateway (blue). The contrast is the entire point of the chart.
            colour = "#94a3b8" if i == 0 else "#3b82f6"
            w = max(2.0, frac * bar_w)
            cy = y + row_h / 2
            parts.append(
                f'<text x="{pad + label_w - 10}" y="{cy + 4}" font-size="12.5" '
                f'text-anchor="end" fill="#64748b">{label}</text>'
                f'<rect x="{pad + label_w}" y="{y + 6}" width="{w:.1f}" height="{row_h - 14}" '
                f'rx="3" fill="{colour}"/>'
                f'<text x="{pad + label_w + w + 8}" y="{cy + 4}" font-size="12.5" '
                f'font-weight="600" fill="#64748b">{display}</text>'
            )
            y += row_h
        y += 6

    parts.append("</svg>")
    return "".join(parts)


def build_chart(all_results: list[tuple[str, str, list[Outcome]]]) -> str:
    groups: list[tuple[str, list[tuple[str, float, str]]]] = []

    for title, metric, outcomes in all_results:
        rows: list[tuple[str, float, str]] = []
        if metric == "success_rate":
            for o in outcomes:
                rows.append((o.label, o.success_rate, f"{o.success_rate:.1%}"))
        elif metric == "p95":
            values = [o.pct(0.95) for o in outcomes]
            worst = max((v for v in values if not math.isnan(v)), default=1.0) or 1.0
            for o, v in zip(outcomes, values, strict=True):
                frac = 0.0 if math.isnan(v) else v / worst
                rows.append((o.label, frac, _fmt_ms(v)))
        else:  # upstream load -- lower is better
            worst = max(o.upstream_calls for o in outcomes) or 1
            for o in outcomes:
                rows.append((o.label, o.upstream_calls / worst, f"{o.upstream_calls} calls"))
        groups.append((title, rows))

    return bar_chart_svg(groups)


async def main() -> None:
    quiet_logs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[tuple[str, str, list[Outcome]]] = []

    print("\n\033[1mLLM Gateway - chaos benchmarks\033[0m")
    print("Every scenario is an A/B against an identical provider and seed.")

    for title, metric, fn in SCENARIOS:
        outcomes = await fn()
        print_table(title, outcomes)
        all_results.append((title, metric, outcomes))

    payload = {
        title: [
            {
                "label": o.label,
                "success_rate": round(o.success_rate, 4),
                "p50_ms": _json_num(o.pct(0.50)),
                "p95_ms": _json_num(o.pct(0.95)),
                "p99_ms": _json_num(o.pct(0.99)),
                "mean_ms": _json_num(o.mean_ms),
                "upstream_calls": o.upstream_calls,
                "requests": o.total,
            }
            for o in outcomes
        ]
        for title, _, outcomes in all_results
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / "chaos.svg").write_text(build_chart(all_results), encoding="utf-8")

    print(f"\n  wrote {RESULTS_DIR / 'results.json'}")
    print(f"  wrote {RESULTS_DIR / 'chaos.svg'}\n")


if __name__ == "__main__":
    asyncio.run(main())
