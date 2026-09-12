# fia-doc

PDF parsers to extract structured data from FIA F1 documents. The output is ready to be imported into [jolpica/jolpica-f1](https://github.com/jolpica/jolpica-f1). This started as part of [theOehrly/Fast-F1#445](https://github.com/theOehrly/Fast-F1/issues/445).


## Supported documents

| Parser | Sessions | Input PDFs | Output |
|---|---|---|---|
| `EntryListParser` | - | Entry List | `RoundEntry`, plus `Driver` for drivers new to Jolpica and `TeamDriver` for non-regular (junior, reserve, etc.) drivers |
| `PracticeParser` | `'fp1'`, `'fp2'`, `'fp3'` (`'fp'` is treated as `'fp1'`) | Practice Session Classification, Practice Session Lap Times | `SessionEntry`, `Lap` |
| `QualifyingParser` | `'quali'`, `'sprint_quali'` | Qualifying Session Classification, Qualifying Session Sector Analysis | `SessionEntry`, `Lap` (for Q1/Q2/Q3 or SQ1/SQ2/SQ3) |
| `RaceParser` | `'race'`, `'sprint'` | Race Classification, Race Lap Analysis, Race History Chart, Race Lap Chart, Race Sector Analysis | `SessionEntry`, `Lap` |
| `PitStopParser` | `'race'`, `'sprint'` | Race Pit Stop Summary | `PitStop` |

The PDFs can be found on at FIA website, e.g. [Event&Timing Information](https://www.fia.com/events/fia-formula-one-world-championship/season-2026/miami-grand-prix/eventtiming-information) or [Decision documents](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/season/season-2026-2072/event/Miami%20Grand%20Prix). The parsers are tested against documents from year 2025 onwards, and *may* also work for earlier years (though not very well tested).


## Installation

Python 3.12 is required.

```bash
pip install git+https://github.com/harningle/fia-doc
```

Some PDFs, or parts of them, are images rather than text. These are read with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), which downloads its models the first time it is used.


## Usage

Each parser takes the path(s) to the PDF(s), the year, the round number, and the session. The parsed data are exposed as pandas DataFrames, and `.to_json()` on each DataFrame converts them to a list of dicts in the [jolpica-schemas](https://github.com/jolpica/jolpica-f1/tree/schemaslibrary/libraries/jolpica-schemas) format.

```python
import json

from fiadoc.parser import (
    EntryListParser,
    PitStopParser,
    PracticeParser,
    QualifyingParser,
    RaceParser,
)

# Entry list
parser = EntryListParser('2025_10_entry_list.pdf', 2025, 10)
entry_list = parser.df.to_json()

# Free practice
parser = PracticeParser(
    classification_file='2025_10_fp1_classification.pdf',
    lap_times_file='2025_10_fp1_lap_times.pdf',  # Optional
    year=2025,
    round_no=10,
    session='fp1'
)
fp1_classification = parser.classification_df.to_json()
fp1_lap_times = parser.lap_times_df.to_json()

# Qualifying (use `session='sprint_quali'` for sprint quali.)
parser = QualifyingParser(
    classification_file='2025_10_quali_classification.pdf',
    lap_times_file=None,  # Deprecated and ignored. Always pass `None`
    sector_analysis_file='2025_10_quali_sector_analysis.pdf',  # Optional
    year=2025,
    round_no=10,
    session='quali'
)
quali_classification = parser.classification_df.to_json()
quali_lap_times = parser.lap_times_df.to_json()

# Race (use `session='sprint'` for sprint race)
parser = RaceParser(
    classification_file='2025_10_race_classification.pdf',
    lap_analysis_file='2025_10_race_lap_analysis.pdf',        # Optional
    history_chart_file='2025_10_race_history_chart.pdf',      # Optional
    lap_chart_file='2025_10_race_lap_chart.pdf',              # Optional
    sector_analysis_file='2025_10_race_sector_analysis.pdf',  # Optional
    year=2025,
    round_no=10,
    session='race'
)
race_classification = parser.classification_df.to_json()
race_lap_times = parser.lap_times_df.to_json()

# Pit stops
parser = PitStopParser('2025_10_race_pit_stop_summary.pdf', 2025, 10, 'race')
race_pit_stops = parser.df.to_json()

with open('2025_10_race_lap_times.json', 'w', encoding='utf-8') as f:
    json.dump(race_lap_times, f, indent=4)
```

An example output, i.e. one element of `race_lap_times` above, looks like:

```json
{
    "object_type": "Lap",
    "foreign_keys": {"year": 2025, "round": 10, "car_number": 1, "session": "R"},
    "objects": [
        {
            "number": 1,
            "position": 2,
            "time": {"_type": "timedelta", "milliseconds": 82549},
            "is_entry_fastest_lap": false
        }
    ]
}
```

`fiadoc.utils.download_pdf(url, out_path)` can be used to download the PDFs. Downloads are cached, so the same URL is only fetched once.

### Parsing PDFs jointly

Some data are spread over several PDFs, so one session may require multiple PDFs:

* **Qualifying**: the classification PDF only has each driver's fastest lap time in Q1/Q2/Q3. The sector analysis PDF has every lap, and the classification PDF is used to assign these laps to Q1/Q2/Q3 and to check the fastest laps
* **Race**: lap times come from the lap analysis PDF. Lap 1 in the lap analysis PDF is shown as calendar time, so the actual lap 1 time is taken from the history chart PDF. Positions on each lap and the starting grid come from the lap chart PDF. The fastest lap is cross-checked against the classification PDF

### Missing PDFs

Not every PDF is published for every session. When an optional PDF is missing (i.e. `None`), the parsers fall back as follows:

* **Practice** without lap times PDF: only the fastest lap of each driver, taken from the classification PDF, with lap number set to `None`
* **Qualifying** without sector analysis PDF: only the fastest lap of each driver in each of Q1/Q2/Q3, taken from the classification PDF, with lap number set to `None`
* **Race** without lap analysis PDF: lap times are taken from the sector analysis PDF instead. If both are missing, `lap_times_df` raises `FileNotFoundError`, while `classification_df` still works
* **Race** without history chart PDF: lap 1 is dropped from the lap times
* **Race** without lap chart PDF: positions on each lap and the starting grid are set to `None`

### Cache

Downloaded PDFs and the driver name to Jolpica driver ID mapping (fetched from the [Jolpica API](https://api.jolpi.ca/ergast/f1/drivers/)) are cached in the `FIADOC_CACHE_DIR` environment variable if set, otherwise in:

* Windows: `%LocalAppData%\fiadoc\Cache`
* macOS: `~/Library/Caches/fiadoc`
* Linux: `~/.cache/fiadoc`


## Development

```bash
pip install -r requirements-dev.txt

pytest fiadoc                               # All tests
pytest fiadoc/tests/test_parse_quali.py     # A single test file
ruff check .
```

The tests download real FIA PDFs and compare the parsed output against the expected JSON in `fiadoc/tests/fixtures/`. When running tests locally, PDFs are cached in a `test/` subfolder of the cache directory above, so they are only downloaded once.

When adding a new season, first update the team and regular driver mappings and the number of drivers in Q2/Q3 in `fiadoc/_constants.py`.


## Other related projects

* [marcll/f1-fia-doc-parser](https://github.com/marcll/f1-fia-doc-parser): this project tries to parse text data, e.g. penalties, using LLMs, which we don't cover here

