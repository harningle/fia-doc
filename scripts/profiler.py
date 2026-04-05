#!/usr/bin/env python
"""
Collects wall time (mean +/- std over `--repeat` runs) and top `--hotspots` cProfile hotspots for
every main method. Results are written to a Markdown file for easy review.

Usage
-----
    python profiler.py --repeat 5 --hotspots 10 --output report.md
"""
import argparse
import cProfile
import logging
import pstats
import statistics
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import paddlex
from paddleocr import logger

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fiadoc.parser import (
    EntryListParser,
    PitStopParser,
    PracticeParser,
    QualifyingParser,
    RaceParser,
)

# Suppress PaddleOCR logging
logger.setLevel(logging.ERROR)
paddlex.utils.logging._logger.setLevel(logging.ERROR)

PDF_DIR = Path('data/pdf')
_FIXTURES = {
    # EntryListParser
    'entry_list': '2026_1_entry_list.pdf',

    # PracticeParser
    'fp_classification': '2026_1_fp1_classification.pdf',
    'fp_lap_times': '2026_1_fp1_lap_times.pdf',

    # QualifyingParser
    'quali_classification': '2026_1_quali_final_classification.pdf',
    'quali_lap_times': '2026_1_quali_lap_times.pdf',

    # RaceParser
    'race_classification': '2026_1_race_classification.pdf',
    'race_lap_analysis': '2026_1_race_lap_analysis.pdf',
    'race_history_chart': '2026_1_race_history_chart.pdf',
    'race_lap_chart': '2026_1_race_lap_chart.pdf',
    'race_sector_analysis': '2026_1_race_sector_analysis.pdf',

    # PitStopParser
    'pit_stop': '2026_1_race_pit_stop_summary.pdf',
}
_F = {k: PDF_DIR / v for k, v in _FIXTURES.items()}  # Resolved absolute paths

_shared: dict = {}  # Mutable container for sharing parser instances between setup and run


def _entry_list_parse() -> None:
    """EntryListParser._parse (called inside __init__)."""
    p = EntryListParser(_F['entry_list'], year=2026, round_no=1)
    p._parse()


def _fp_parse_classification() -> None:
    p = PracticeParser(classification_file=_F['fp_classification'], lap_times_file=None,
                       year=2026, round_no=1, session='fp1')
    p._parse_classification()


def _fp_lap_times_setup() -> None:
    """
    Create `PracticeParser` and pre-parse classification so it's cached. Otherwise,
    `._parse_lap_times()` will pick `_parse_classification()` times
    """
    p = PracticeParser(classification_file=_F['fp_classification'],
                       lap_times_file=_F['fp_lap_times'],
                       year=2026, round_no=1, session='fp1')
    _ = p.classification_df  # populate cached_property
    _shared['fp_parser'] = p


def _fp_parse_lap_times() -> None:
    _shared['fp_parser']._parse_lap_times()


def _quali_parse_classification() -> None:
    p = QualifyingParser(classification_file=_F['quali_classification'], lap_times_file=None,
                         year=2026, round_no=1, session='quali')
    p._parse_classification()


def _quali_lap_times_setup() -> None:
    """Similar to `_fp_lap_times_setup()`"""
    p = QualifyingParser(classification_file=_F['quali_classification'],
                         lap_times_file=_F['quali_lap_times'],
                         year=2026, round_no=1, session='quali')
    _ = p.classification_df  # populate cached_property
    _shared['quali_parser'] = p


def _quali_parse_lap_times() -> None:
    _shared['quali_parser']._parse_lap_times()


def _race_parse_classification() -> None:
    p = RaceParser(classification_file=_F['race_classification'],
                   lap_analysis_file=None, history_chart_file=None, lap_chart_file=None,
                   sector_analysis_file=None,
                   year=2026, round_no=1, session='race')
    p._parse_classification()


def _race_parse_lap_analysis() -> None:
    p = RaceParser(classification_file=None,
                   lap_analysis_file=_F['race_lap_analysis'],
                   history_chart_file=None, lap_chart_file=None, sector_analysis_file=None,
                   year=2026, round_no=1, session='race',)
    p._parse_lap_analysis()


def _race_parse_history_chart() -> None:
    p = RaceParser(classification_file=None,
                   history_chart_file=_F['race_history_chart'],
                   lap_analysis_file=None, lap_chart_file=None, sector_analysis_file=None,
                   year=2026, round_no=1, session='race')
    p._parse_history_chart()


def _race_parse_lap_chart() -> None:
    p = RaceParser(classification_file=None,
                   lap_chart_file=_F['race_lap_chart'],
                   lap_analysis_file=None, history_chart_file=None, sector_analysis_file=None,
                   year=2026, round_no=1, session='race')
    p._parse_lap_chart()


def _race_parse_sector_analysis() -> None:
    p = RaceParser(classification_file=None,
                   sector_analysis_file=_F['race_sector_analysis'],
                   lap_analysis_file=None, history_chart_file=None, lap_chart_file=None,
                   year=2026, round_no=1, session='race')
    p._parse_sector_analysis()


def _pit_stop_parse() -> None:
    """PitStopParser._parse (called inside __init__)."""
    p = PitStopParser(_F['pit_stop'], year=2026, round_no=1, session='race')
    p._parse()


@dataclass
class Benchmark:
    parser_name: str
    method_name: str
    fn: Callable[[], None]
    required_files: list[str]
    setup: Callable[[], None] | None = None  # Called once before all runs (e.g. pre-parse deps)


@dataclass
class BenchmarkResult:
    label: str
    times: list[float] = field(default_factory=list)
    hotspots: list[dict] = field(default_factory=list)
    error: str | None = None


BENCHMARKS: list[Benchmark] = [
    Benchmark('EntryListParser',
              '_parse',
              _entry_list_parse,
              ['entry_list']),

    Benchmark('PracticeParser',
              '_parse_classification',
              _fp_parse_classification,
              ['fp_classification']),

    Benchmark('PracticeParser',
              '_parse_lap_times',
              _fp_parse_lap_times,
              ['fp_classification', 'fp_lap_times'],
              setup=_fp_lap_times_setup),

    Benchmark('QualifyingParser',
              '_parse_classification',
              _quali_parse_classification,
              ['quali_classification']),

    Benchmark('QualifyingParser',
              '_parse_lap_times',
              _quali_parse_lap_times,
              ['quali_classification', 'quali_lap_times'],
              setup=_quali_lap_times_setup),

    Benchmark('RaceParser',
              '_parse_classification',
              _race_parse_classification,
              ['race_classification']),

    Benchmark('RaceParser',
              '_parse_lap_analysis',
              _race_parse_lap_analysis,
              ['race_lap_analysis']),

    Benchmark('RaceParser',
              '_parse_history_chart',
              _race_parse_history_chart,
              ['race_history_chart']),

    Benchmark('RaceParser',
              '_parse_lap_chart',
              _race_parse_lap_chart,
              ['race_lap_chart']),

    Benchmark('RaceParser',
              '_parse_sector_analysis',
              _race_parse_sector_analysis,
              ['race_sector_analysis']),

    Benchmark('PitStopParser',
              '_parse',
              _pit_stop_parse,
              ['pit_stop'])
]


def _shorten_path(path: str) -> str:
    """Collapse long absolute paths into something readable"""
    for marker in ('fiadoc/', 'site-packages/'):
        idx = path.find(marker)
        if idx != -1:
            return path[idx:]
    return path.rsplit('/', 1)[-1] if '/' in path else path


def _extract_hotspots(pr: cProfile.Profile, n: int) -> list[dict]:
    """Return top `n` functions sorted by self-time (tottime)."""
    stats = pstats.Stats(pr)
    # stats.stats: {(file, line, name): (cc, nc, tt, ct, callers)}
    items = sorted(stats.stats.items(), key=lambda x: x[1][2], reverse=True)
    hotspots = []
    for (filepath, lineno, funcname), (_, ncalls, tottime, cumtime, _) in items[:n]:
        hotspots.append({
            'function': funcname,
            'source': f'{_shorten_path(filepath)}:{lineno}',
            'ncalls': ncalls,
            'tottime': tottime,
            'cumtime': cumtime,
        })
    return hotspots


def _run_benchmark(bm: Benchmark, n_repeat: int, n_hotspots: int) -> BenchmarkResult:
    """Execute a single benchmark: warm-up -> timing runs -> cProfile"""
    label = f'{bm.parser_name}.{bm.method_name}'
    result = BenchmarkResult(label=label)

    # Check that required PDF fixtures exist
    missing = [k for k in bm.required_files if not _F[k].exists()]
    if missing:
        result.error = f'Missing fixture(s): {", ".join(_FIXTURES[k] for k in missing)}'
        return result

    try:
        # Setup (e.g. pre-parse dependencies so they are cached)
        if bm.setup:
            bm.setup()

        # Warm-up (discard)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bm.fn()

        # Timing runs
        for _ in range(n_repeat):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                t0 = time.perf_counter()
                bm.fn()
                t1 = time.perf_counter()
            result.times.append(t1 - t0)

        # cProfile run
        pr = cProfile.Profile()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pr.enable()
            bm.fn()
            pr.disable()
        result.hotspots = _extract_hotspots(pr, n_hotspots)

    except:  # noqa: E722
        result.error = traceback.format_exc()

    return result


def _fmt(val: float) -> str:
    """Format seconds to a human-friendly millisecond"""
    return f'{val * 1000:.1f} ms'


def _write_markdown(results: list[BenchmarkResult], path: Path, n_repeat: int) -> None:
    lines: list[str] = []
    w = lines.append

    w('# Profiler Results\n')
    w(f'- **Generated**: {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}')
    w(f'- **Python**: {sys.version.split()[0]}')
    w(f'- **Repeats per method**: {n_repeat}\n')

    w('## Summary\n')
    w('| Parser | Method | Mean | Std | Min | Max |')
    w('|--------|--------|------|-----|-----|-----|')
    for r in results:
        parser, method = r.label.split('.', 1)
        if r.error:
            w(f'| {parser} | `{method}` | — | — | — | — | error |')
            continue
        if not r.times:
            w(f'| {parser} | `{method}` | — | — | — | — | skipped |')
            continue
        mean = statistics.mean(r.times)
        std = statistics.stdev(r.times) if len(r.times) > 1 else 0.0
        mn, mx = min(r.times), max(r.times)
        w(f'| {parser} | `{method}` | {_fmt(mean)} | {_fmt(std)} '
          f'| {_fmt(mn)} | {_fmt(mx)} |')

    w('\n## Hotspots (by self-time)\n')
    for r in results:
        if r.error or not r.hotspots:
            continue

        w(f'### {r.label}\n')
        w('| # | Function | Source | Calls | Self-time | Cum-time |')
        w('|---|----------|--------|------:|----------:|---------:|')
        for i, h in enumerate(r.hotspots, 1):
            w(f'| {i} | `{h["function"]}` | `{h["source"]}` '
              f'| {h["ncalls"]} | {_fmt(h["tottime"])} | {_fmt(h["cumtime"])} |')
        w('')

    errors = [r for r in results if r.error]
    if errors:
        w('## Errors\n')
        for r in errors:
            w(f'### {r.label}\n')
            w('```')
            w(r.error.rstrip())
            w('```\n')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeat', type=int, default=5,
                    help='Number of timing runs per method (default: 5)')
    ap.add_argument('--output', type=str, default='profiler_results.md',
                    help='Path to the output Markdown file (default: profiler_results.md)')
    ap.add_argument('--hotspots', type=int, default=10,
                    help='Number of top hotspots to show per method (default: 10)')
    args = ap.parse_args()

    total = len(BENCHMARKS)
    results: list[BenchmarkResult] = []

    print(f'Running profiler  --  {args.repeat} repeats, {args.hotspots} hotspots')
    print(f'Output: {args.output}')
    print('=' * 72)

    for i, bm in enumerate(BENCHMARKS, 1):
        label = f'{bm.parser_name}.{bm.method_name}'
        print(f'[{i}/{total}] {label}... ', end='', flush=True)

        result = _run_benchmark(bm, n_repeat=args.repeat, n_hotspots=args.hotspots)
        results.append(result)

        if result.error:
            print('ERROR')
        elif result.times:
            mean = statistics.mean(result.times)
            std = statistics.stdev(result.times) if len(result.times) > 1 else 0.0
            print(f'{_fmt(mean).removesuffix(" ms")} +/- {_fmt(std)}')
        else:
            print('skipped')

    print('=' * 72)

    out = Path(args.output)
    _write_markdown(results, out, n_repeat=args.repeat)
    print(f'Results written to {out.resolve()}')


if __name__ == '__main__':
    main()
