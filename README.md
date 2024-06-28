# fia-doc

PDF parsers to extract structured data from FIA F1 documents. This is part of theOehrly/Fast-F1#445.

We plan to get the following data from the PDFs:

* lap times: for each race/sprint/qualifying, the lap times of each driver in each lap
* tyre compounds: for each race, the mapping from "Soft", "Medium", "Hard" to the actual tyre compounds C1, C2, etc.
* penalties: in progress

Currently we have the "table" like data, e.g. lap time, pit stop, etc., more or less parsed. The "less table" ones, such as penalties, are not started yet. Please check [marcll/f1-fia-doc-parser](https://github.com/marcll/f1-fia-doc-parser), who uses LLMs to get these unstructured info. from PDFs, with a very good quality!
