#!/usr/bin/env python
"""Benchmark public parser outputs against the correctness-suite PDFs.

Run from an activated Python 3.12 environment: python scripts/profiler.py --output temp/perf/run
Each case uses five fresh processes and five warm parses with new parser instances. Downloads
are prepared before timing. Cold means a fresh interpreter, with warm filesystem/model caches.
Reports include complete outputs, warnings, dependency versions, RSS and separate diagnostics.
"""
import argparse
import cProfile
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the correctness suite's URLs, metadata and expected complete JSON outputs.
CASES = {
    'practice': ('fp', '2026_1_fp1_classification.json'),
    'qualifying': ('quali', '2026_20_quali_classification.json'),
    'race': ('race', '2026_13_race_classification.json'),
    'sprint': ('race', '2025_13_sprint_classification.json'),
    'ocr': ('race', '2025_11_race_classification.json'),
    'entry_list': ('entry_list', '2026_1_entry_list.json'),
    'pit_stop': ('pit_stop', '2023_13_race_pit_stop.json'),
}
DEFAULT_CASES = ['practice', 'qualifying', 'race', 'sprint', 'ocr']


def prepare_case(name, directory):
    """Resolve/download fixtures outside the measured region. Fail on missing/invalid inputs."""
    from fiadoc import parser  # noqa: PLC0415
    from fiadoc.utils import download_pdf  # noqa: PLC0415

    kind, fixture = CASES[name]
    cases = importlib.import_module(f'fiadoc.tests.test_parse_{kind}').race_list
    case = next(c for c in cases if fixture in c)
    directory.mkdir(parents=True, exist_ok=True)
    if kind == 'race':
        urls, metadata, expected = case[:5], case[5:8], case[8:10]
        cls, properties = parser.RaceParser, ['classification_df', 'lap_times_df']
    elif kind == 'quali':
        # Lap-times input is deprecated; qualifying now consumes sector analysis.
        urls, metadata, expected = (case[0], None, case[2]), case[3:6], case[6:8]
        cls, properties = parser.QualifyingParser, ['classification_df', 'lap_times_df']
    elif kind == 'fp':
        urls, metadata, expected = case[:2], case[2:5], case[5:7]
        cls, properties = parser.PracticeParser, ['classification_df', 'lap_times_df']
    else:
        urls, metadata, expected = case[:1], case[1:-2], case[-2:-1]
        cls = parser.EntryListParser if kind == 'entry_list' else parser.PitStopParser
        properties = ['df']
        if kind == 'entry_list':
            base = ('https://www.fia.com/sites/default/files/decision-document/'
                    if metadata[0] <= 2024 else  # noqa: PLR2004
                    'https://www.fia.com/system/files/decision-document/')
            urls = [base + urls[0]]
    paths = []
    for i, url in enumerate(urls):
        if url is None:
            paths.append(None)
            continue
        resolved_url = (url if url.startswith('https://') else
                        'https://www.fia.com/sites/default/files/' + url)
        path = directory / f'{i}.pdf'
        download_pdf(resolved_url, path)
        paths.append(path)
    expected = [json.loads((ROOT / 'fiadoc/tests/fixtures' / p).read_text()) for p in expected]
    return cls, paths, metadata, properties, expected


class Phases:
    """Exclusive phase wall times; nested rendering/extraction is not counted twice."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = []
        self.ocr_crops = []
        self.crop_context = None

    def wrap(self, fn, name):
        @wraps(fn)
        def timed(*args, **kwargs):
            previous_context = self.crop_context
            if name in {'text_and_ocr_preparation', 'search_and_ocr_preparation'}:
                page = args[0]
                clip = kwargs.get('clip')
                self.crop_context = {'file': str(page.file), 'page': page.number,
                                     'rotation': page.rotation, 'method': fn.__name__,
                                     'clip': tuple(clip) if clip is not None else None}
            self.stack.append([time.perf_counter(), 0.0])
            try:
                return fn(*args, **kwargs)
            finally:
                start, nested = self.stack.pop()
                elapsed = time.perf_counter() - start
                self.seconds[name] += elapsed - nested
                self.calls[name] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed
                self.crop_context = previous_context
        return timed

    def instrument(self, patches):
        import pymupdf  # noqa: PLC0415

        from fiadoc.parser import page, parser  # noqa: PLC0415

        native_owner, native_method = ((page.Page, '_get_native_text')
                                       if hasattr(page.Page, '_get_native_text') else
                                       (pymupdf.Page, 'get_text'))
        for obj, method, name in [
            (pymupdf.Page, 'get_pixmap', 'rendering'),
            (native_owner, native_method, 'native_extraction'),
            (page.Page, '_get_full_page_pixmap', 'render_cache_and_copy'),
            (page.Page, 'get_text', 'text_and_ocr_preparation'),
            (page.Page, 'search_for', 'search_and_ocr_preparation'),
            (parser.BaseParser, '_detect_cols', 'column_detection'),
            (parser, 'label', 'connected_components'),
            (parser, 'find_objects', 'component_bounds'),
            (page.Page, 'search_for_black_lines', 'line_detection'),
            (page.Page, 'search_for_white_strips', 'white_strips'),
            (page.Page, 'search_for_grey_white_rows', 'grey_rows'),
        ]:
            wrapped = self.wrap(getattr(obj, method), name)
            if obj is parser.BaseParser:
                wrapped = staticmethod(wrapped)
            patches.enter_context(patch.object(obj, method, wrapped))

        original = page.get_ocr_instance
        seen = set()

        def get_ocr():
            ocr = original()
            if id(ocr) not in seen:
                seen.add(id(ocr))
                predict = ocr.predict

                def record(image, *args, **kwargs):
                    self.ocr_crops.append({'context': self.crop_context,
                                          'shape': image.shape, 'sha256': hashlib.sha256(
                        image.tobytes()).hexdigest()})
                    return predict(image, *args, **kwargs)

                patches.enter_context(patch.object(ocr, 'predict',
                                                  self.wrap(record, 'ocr_inference')))
            return ocr

        patches.enter_context(patch.object(page, 'get_ocr_instance',
                                          self.wrap(get_ocr, 'ocr_initialisation')))


def parse_case(case):
    cls, paths, metadata, properties, _ = case
    start = time.perf_counter()
    instance = cls(*paths, *metadata)
    frames = [getattr(instance, p) for p in properties]
    parsed = time.perf_counter() - start
    start = time.perf_counter()
    output = [frame.to_json() for frame in frames]
    serialised = time.perf_counter() - start
    return output, {'parsing_s': parsed, 'serialisation_s': serialised,
                    'total_s': parsed + serialised}


def metadata():
    import psutil  # noqa: PLC0415

    packages = ['pymupdf', 'numpy', 'scipy', 'pandas', 'paddleocr', 'paddlex',
                'paddlepaddle', 'paddlepaddle-gpu', 'psutil']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'platform': platform.platform(),
            'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
            'threads': psutil.Process().num_threads(), 'dependencies': versions,
            'thread_environment': {n: os.environ.get(n) for n in
                                   ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']}}


def peak_rss():
    if sys.platform != 'win32':
        import resource  # noqa: PLC0415

        scale = 1 if sys.platform == 'darwin' else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    import psutil  # noqa: PLC0415

    return psutil.Process().memory_info().peak_wset


def worker(args):
    started = time.perf_counter()
    importlib.import_module('fiadoc.parser')
    imported = time.perf_counter() - started
    from fiadoc.utils import sort_json  # noqa: PLC0415

    case = prepare_case(args.worker, args.inputs / args.worker)
    result = {'import_s': imported, 'runs': [], 'warnings': []}
    phases = Phases()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        # Match the suite's filter for this optional Paddle build-tool warning. With lazy OCR
        # loading it otherwise moves from import-time stderr into the parser warning capture.
        warnings.filterwarnings('ignore', message='.*ccache.*', category=UserWarning)
        with ExitStack() as patches:
            if args.repeat:
                phases.instrument(patches)
            output, timing = parse_case(case)
        result['first_parse'] = timing
        print('First parse:', timing, flush=True)
        assert sort_json(output) == sort_json(case[-1]), 'JSON differs from correctness fixtures'
        for _ in range(args.repeat):
            current, timing = parse_case(case)
            assert current == output, 'JSON differs between repeated parses'
            result['runs'].append(timing)
            print('Warm parse:', timing, flush=True)
        if args.repeat:
            profile = cProfile.Profile()
            profile.enable()
            current, _ = parse_case(case)
            profile.disable()
            assert current == output
            profile.dump_stats(str(args.output.with_suffix('.pstats')))
            stats = pstats.Stats(profile)
            result['hotspots'] = [
                {'function': str(key), 'calls': val[1], 'self_s': val[2], 'cumulative_s': val[3]}
                for key, val in sorted(stats.stats.items(), key=lambda x: x[1][2], reverse=True)
                [:args.hotspots]
            ]
        result['warnings'] = sorted({str(w.message) for w in caught})
    result.update({'output': output, 'phases_s': dict(phases.seconds),
                   'phase_calls': dict(phases.calls), 'ocr_crops': phases.ocr_crops,
                   'peak_rss_bytes': peak_rss(), 'environment': metadata(),
                   'inputs': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in case[1] if p is not None}})
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def describe(values):
    if not values:
        return '-'
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f'{statistics.mean(values):.3f} +/- {std:.3f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--cold-repeat', type=int,
                    help='Fresh-process parses per case (default: --repeat); 0 for warm-only runs')
    ap.add_argument('--cases', nargs='+', choices=CASES, default=DEFAULT_CASES)
    ap.add_argument('--output', type=Path, required=True, help='New report directory')
    ap.add_argument('--inputs', type=Path, default=ROOT / 'temp/perf-inputs')
    ap.add_argument('--compare', type=Path,
                    help='Previous report directory; require identical JSON')
    ap.add_argument('--hotspots', type=int, default=15)
    ap.add_argument('--worker', choices=CASES, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeat < 1:
        ap.error('--repeat must be positive')
    cold_repeat = args.repeat if args.cold_repeat is None else args.cold_repeat
    if cold_repeat < 0:
        ap.error('--cold-repeat must be non-negative')
    args.output.mkdir(parents=True, exist_ok=False)
    # Preparation runs separately so downloads do not contaminate cold import measurements.
    for name in args.cases:
        prepare_case(name, args.inputs / name)
    lines = ['# Parser benchmark', '',
             'Seconds: mean +/- sample standard deviation. RSS is process peak, including imports.',
             ('Cold = fresh interpreter; filesystem and models are cached. Warm = new parser '
              'instances in one process. Diagnostics are separate from warm timing runs.'), '',
             '| Case | Import | Cold parse + JSON | Warm parse + JSON | Peak RSS MiB |',
             '| --- | --- | --- | --- | --- |']
    for name in args.cases:
        cold = []
        workers = [(f'cold-{i}', 0) for i in range(cold_repeat)] + [('warm', args.repeat)]
        for label, count in workers:
            output = args.output / f'{name}-{label}.json'
            with output.with_suffix('.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name,
                                '--repeat', str(count), '--inputs', str(args.inputs),
                                '--output', str(output), '--hotspots', str(args.hotspots)],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            result = json.loads(output.read_text())
            if args.compare:
                previous = json.loads((args.compare / f'{name}-warm.json').read_text())
                assert result['output'] == previous['output'], f'{name}: baseline JSON differs'
                assert result['inputs'] == previous['inputs'], f'{name}: input PDF hashes differ'
                assert result['warnings'] == previous['warnings'], f'{name}: warnings differ'
            if label != 'warm':
                cold.append(result)
        lines.append(f'| {name} | {describe([r["import_s"] for r in cold])} | '
                     f'{describe([r["first_parse"]["total_s"] for r in cold])} | '
                     f'{describe([r["total_s"] for r in result["runs"]])} | '
                     f'{max(r["peak_rss_bytes"] for r in [*cold, result]) / 1024**2:.1f} |')
        print(lines[-1], flush=True)
        (args.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
