# fia-doc

PDF parsers to extract structured data from FIA F1 documents. This is part of [theOehrly/Fast-F1#445](https://github.com/theOehrly/Fast-F1/issues/445) and [jolpica/jolpica-f1](https://github.com/jolpica/jolpica-f1).


## Parsing procedure

1. Wed/Thu: get tyre compound from ["Event Notes/Pirelli Preview"](https://www.fia.com/sites/default/files/decision-document/2023%20United%20States%20Grand%20Prix%20-%20Event%20Notes%20-%20Pirelli%20Preview.pdf)
1. Thu/Fri: get driver entry list, from ["Entry List"](https://www.fia.com/sites/default/files/decision-document/2023%20United%20States%20Grand%20Prix%20-%20Entry%20List.pdf)
    * the data dump is going to [`RoundEntry`](https://github.com/jolpica/jolpica-f1/blob/main/jolpica/formula_one/models/database.svg) table
1. Sat: get quali. lap times and classification, from ["Quali. Lap Times"](https://www.fia.com/sites/default/files/2023_19_usa_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf) and ["Quali. Final Classification"](https://www.fia.com/sites/default/files/doc_20_-_2023_united_states_grand_prix_-_final_qualifying_classification.pdf). This gives two data dumps:
    * classification, to be inserted into [`SessionEntry`](https://github.com/jolpica/jolpica-f1/blob/main/jolpica/formula_one/models/database.svg) table
    * lap times, to be inserted into [`Lap`](https://github.com/jolpica/jolpica-f1/blob/main/jolpica/formula_one/models/database.svg) table
    * the two PDFs have to be parsed *__jointly__*! Because ["Quali. Final Classification"](https://www.fia.com/sites/default/files/doc_20_-_2023_united_states_grand_prix_-_final_qualifying_classification.pdf) only tells us the lap time of the fastest lap, but doesn't tell us which lap number it is. We need to combine this with ["Quali. Lap Times"](https://www.fia.com/sites/default/files/2023_19_usa_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf) to get the full info.
1. Sun: get race lap times, pit stops, and classification, from ["Race History Chart"](https://www.fia.com/sites/default/files/2023_19_usa_f1_r0_timing_racehistorychart_v01.pdf), ["Pit Stop Summary"](https://www.fia.com/sites/default/files/2023_19_usa_f1_r0_timing_racepitstopsummary_v01.pdf), and ["Race Final Classification"](https://www.fia.com/sites/default/files/doc_66_-_2023_united_states_grand_prix_-_final_race_classification.pdf). We get three data dumps from them:
    * classification, to be inserted into [`SessionEntry`](https://github.com/jolpica/jolpica-f1/blob/main/jolpica/formula_one/models/database.svg) table
    * lap times, to be inserted into [`Lap`](https://github.com/jolpica/jolpica-f1/blob/main/jolpica/formula_one/models/database.svg) table
    * pit stops, to be inserted into [`PitStop`](https://github.com/jolpica/jolpica-f1/blob/main/jolpica/formula_one/models/database.svg) table
    * the three PDFs can be parsed individually. But to enable cross validation, we parse them *__jointly__*?

In case of sprint weekend, sprint quali./shootout can be parsed as if it's a usual quali. session, and sprint race, as a usual race.


## Use

```python
from fiadoc.parser import EntryListParser, PitStopParser, QualifyingParser, RaceParser
import json

parser = EntryListParser('data/pdf/2023_18_entry_list.pdf', 2023, 18)
with open('data/dump/2023_18_entry_list.json', 'w') as f:
    json.dump(parser.df.to_json(), f)

parser = QualifyingParser('data/pdf/2023_18_quali_classification.pdf',
                          'data/pdf/2023_18_quali_lap_times.pdf',
                          2023,
                          18,
                          'quali')
with open('data/dump/2023_18_quali_classification.json', 'w') as f:
    json.dump(parser.classification_df.to_json(), f)
with open('data/dump/2023_18_quali_lap_times.json', 'w') as f:
    json.dump(parser.lap_times_df.to_json(), f)

parser = QualifyingParser('data/pdf/2023_18_sprint_quali_classification.pdf',
                          'data/pdf/2023_18_sprint_quali_lap_times.pdf',
                          2023,
                          18,
                          'sprint_quali')
with open('data/dump/2023_18_sprint_quali_classification.json', 'w') as f:
    json.dump(parser.classification_df.to_json(), f)
with open('data/dump/2023_18_sprint_quali_lap_times.json', 'w') as f:
    json.dump(parser.lap_times_df.to_json(), f)

parser = RaceParser('data/pdf/2023_18_sprint_classification.pdf',
                    'data/pdf/2023_18_sprint_lap_analysis.pdf',
                    'data/pdf/2023_18_sprint_history_chart.pdf',
                    'data/pdf/2023_18_sprint_lap_chart.pdf',
                    2023,
                    18,
                    'sprint_race')
with open('data/dump/2023_18_sprint_classification.json', 'w') as f:
    json.dump(parser.classification_df.to_json(), f)
with open('data/dump/2023_18_sprint_lap_times.json', 'w') as f:
    json.dump(parser.lap_times_df.to_json(), f)
    
parser = RaceParser('data/pdf/2023_18_race_final_classification.pdf',
                    'data/pdf/2023_18_race_lap_analysis.pdf',
                    'data/pdf/2023_18_race_history_chart.pdf',
                    'data/pdf/2023_18_race_lap_chart.pdf',
                    2023,
                    18,
                    'race')
with open('data/dump/2023_18_race_classification.json', 'w') as f:
    json.dump(parser.classification_df.to_json(), f)
with open('data/dump/2023_18_race_lap_times.json', 'w') as f:
    json.dump(parser.lap_times_df.to_json(), f)

parser = PitStopParser('data/pdf/2023_18_race_pit_stop_summary.pdf', 2023, 18, 'race')
with open('data/dump/2023_18_race_pit_stops.json', 'w') as f:
    json.dump(parser.df.to_json(), f)
```
