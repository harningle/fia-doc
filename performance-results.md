# Parser performance measurements

Measured on 10 November 2026 against freshly fetched `main` (`8faa946`). The retained changes reuse a page's native display list while extracting each clip with MuPDF, defer OCR imports until first use, avoid recursive classification parsing through the existing cached property, and reduce white-strip raster work. Public constructors, DataFrame properties, JSON schemas and OCR models are preserved.

## Hosted CPU results

The [GitHub Actions comparison](https://github.com/harningle/fia-doc/actions/runs/42873362417) passed at `183f60b`. Each row includes five warm parses with new parser instances and complete public JSON generation. Values are mean +/- sample standard deviation in seconds. RSS is the observed peak of the worker containing the diagnostic parse, five warm runs and a separate cProfile run; it includes imports and native allocations.

| Case | Main seconds | Perf seconds | Time reduction | Main peak MiB | Perf peak MiB |
| --- | --- | --- | --- | --- | --- |
| Practice, 2026 round 1 FP1 | 5.113 +/- 0.077 | 3.460 +/- 0.082 | 32.3% | 1603.2 | 1615.8 |
| Qualifying, 2026 round 20 | 31.010 +/- 0.576 | 27.671 +/- 0.266 | 10.8% | 5256.5 | 5281.9 |
| Race, 2026 round 13 | 23.109 +/- 0.339 | 12.456 +/- 0.109 | 46.1% | 2306.2 | 2073.7 |
| Sprint, 2025 round 13 | 8.972 +/- 0.108 | 4.999 +/- 0.114 | 44.3% | 1969.7 | 1771.3 |
| Image-based race, 2025 round 11 | 62.238 +/- 0.899 | 28.721 +/- 0.511 | 53.9% | 2933.0 | 2711.6 |
| Entry list, 2026 round 1 | 0.251 +/- 0.004 | 0.209 +/- 0.025 | 16.6% | 566.0 | 442.2 |
| Pit stops, 2023 round 13 | 3.888 +/- 0.050 | 3.378 +/- 0.024 | 13.1% | 883.7 | 761.9 |

The sum of the first five case means fell from 130.44 to 77.31 seconds (40.7%). Memory savings vary: practice and qualifying peaks increased by 0.8% and 0.5%, while the other five cases used less memory. These are measurements of this workload, rather than a predicted ingestion-job speedup.

The hosted worker had four logical CPUs and 18 process threads after OCR. Python was 3.12.16; PyMuPDF 1.26.0, PaddleOCR/PaddlePaddle 3.2.0, PaddleX 3.2.0, NumPy 2.3.2, SciPy 1.16.1 and pandas 3.0.3 were identical before and after. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were each 2; CUDA was disabled. Full per-case metadata, samples, phase timings, call counts and input/output hashes are in [performance-results.json](performance-results.json). The run artefact contains complete JSON outputs, logs, profiles and the dependency freeze.

Five fresh interpreters were also measured for each native-only case, with filesystem/PDF caches warm and downloads excluded. Entry-list import time fell from 4.031 +/- 0.359 to 0.563 +/- 0.013 seconds; import plus first parse/JSON fell from 4.302 +/- 0.354 to 0.768 +/- 0.013 seconds. The pit-stop equivalent fell from 7.883 +/- 0.431 to 4.005 +/- 0.036 seconds.

## Independent local experiments

Local CPU measurements used Python 3.12.3 under WSL2 on a Ryzen 7 9700X with 16 logical CPUs, the same core dependency versions, and the same thread settings. Five warm runs before/after averaged 6.419/2.796 seconds for practice, 22.809/13.774 for qualifying, 29.205/10.377 for race, 11.300/4.120 for sprint and 60.482/21.480 for the image-based race. Raw samples and deviations are recorded in the JSON report.

- Native reuse alone reduced practice from 6.419 to 3.062 seconds and the image-based race from 60.482 to 31.489 seconds. A cached full-page TextPage was rejected because PyMuPDF ignores a later clip. Replaying a display list into a fresh clipped text device preserves native glyph selection, ordering and superscript flags.
- Lazy OCR loading reduced native-only import time from 3.966 +/- 0.336 to 0.598 +/- 0.008 seconds, and mean peak RSS in that check from 216.14 to 160.85 MiB. Native-only parsing does not import Paddle, PaddleOCR or PaddleX. The existing model singleton is retained.
- Classification requested the starting grid, which requested laps, which re-entered classification before its cached property was populated. Publishing the already parsed classification to that existing property removes this re-entry. With the other optimisations fixed, restoring the recursive behaviour cost 32.059 +/- 0.266 seconds versus 21.480 +/- 0.109 seconds. OCR calls fell from 308 to 155, with the same 154 distinct prepared-image hashes and identical public output. This avoids another OCR-result cache or coordinate remapping.
- White-strip detection now reduces byte pixels directly, avoiding a temporary full RGB boolean array. Five batches of 20 calls measured 3.173 +/- 0.089 ms before and 0.777 +/- 0.014 ms after. The reduction identity preserves empty-width clipping behaviour; the final form was regression-tested and remeasured.
- A column-mask rewrite measured 24.360 +/- 0.706 ms before and 24.428 +/- 0.831 ms after. It was discarded. Connected-component labelling remains unchanged.
- A separate 300-DPI column-detection prototype measured practice at 2.732 +/- 0.048 seconds versus 2.796 +/- 0.019 at 600 DPI. The small difference and additional full-page cache entry did not justify retaining it. Layout, OCR and strikeout resolutions are preserved.

For the hosted image-based race diagnostic, native `get_text` extraction fell from 16.333 to 1.764 seconds, OCR inference from 32.785 to 15.547 seconds, and white-strip detection from 0.677 to 0.260 seconds. Diagnostic phase values are exclusive wall times and are separate from the uninstrumented warm measurements. The OCR-initialisation phase includes deferred dependency imports on `perf`.

## RTX 3060 diagnostic

The [documented GPU 3.2.0 indexes](https://www.paddleocr.ai/v3.5.0/en/version3.x/paddlepaddle_installation.html) returned HTTP 502 during this session. The hardware comparison therefore used matching PaddlePaddle 3.1.1 CPU/GPU runtimes in separate external Python 3.12 environments, with PaddleOCR/PaddleX 3.2.0 and the same existing OCR models. Implementation and hosted CPU validation used PaddlePaddle 3.2.0.

Five fresh processes per device each constructed one model, discarded a prediction warm-up, then timed five predictions per crop. Inference deviations below are across the five process means. Model construction excludes dependency import and prediction warm-up.

| Operation | CPU seconds | RTX 3060 seconds |
| --- | --- | --- |
| Model construction | 1.541 +/- 0.085 | 2.685 +/- 0.674 |
| Small crop, 83 x 417 RGB pixels | 0.0408 +/- 0.0006 | 0.0134 +/- 0.0004 |
| Table crop, 1667 x 2250 RGB pixels | 3.149 +/- 0.068 | 0.820 +/- 0.030 |

Input hashes, recognised text and boxes matched exactly across both devices and all processes: one text on the small crop and 219 on the table crop. These crop measurements do not establish a GPU parser end-to-end speedup, and did not determine which optimisations were retained.

## Validation and reproduction

The [implementation CI run](https://github.com/harningle/fia-doc/actions/runs/42873361893) passed 256 tests with 192 opt-in `--full` cases skipped. The final local source, including the empty-width reduction identity, passed 257 tests with the same 192 skips; Ruff and `git diff --check` passed. Every hosted benchmark parse matched the complete baseline JSON, including list order, PDF hashes and parser warning sets. Regression coverage compares raw extraction with PyMuPDF for clipping, partial glyphs, superscripts, images, flags, sorting and all four rotations; existing strikeout and OCR tests pass. Additional tests verify both race-property access orders, a single classification parse, lazy OCR imports, the singleton and empty-width white strips.

The profiler refuses to overwrite an existing report directory and fails on output differences or parsing errors. It measures public output generation, rather than repeatedly calling private parse methods on an already parsed object. The historical `profiler_results.md` and all pre-existing user scratch files were preserved. No production workflow or scheduled-repository changes are retained.

Run from an activated environment under `~/.virtualenvs/`, with development requirements installed. For a matching before/after CPU comparison:

```bash
source "$HOME/.virtualenvs/fia-doc-perf-cpu/bin/activate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1
git worktree add --detach temp/perf/baseline 8faa946
cp scripts/profiler.py temp/perf/baseline/scripts/profiler.py
python temp/perf/baseline/scripts/profiler.py --cold-repeat 0 --inputs temp/perf/inputs --output temp/perf/before
python scripts/profiler.py --cold-repeat 0 --inputs temp/perf/inputs --output temp/perf/after --compare temp/perf/before
```

Omit `--cold-repeat 0` to collect five cold parses as well. Add `--cases entry_list pit_stop` for the native-only cases. Local complete outputs, profiling files and experiment logs remain under the ignored `temp/perf` directory.
