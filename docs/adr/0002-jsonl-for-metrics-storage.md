# JSONL as the metrics storage format

Metrics are written as newline-delimited JSON (JSONL) — one object per turn, with per-call data embedded as a list field. This format is trivially appendable (both the scripted benchmark and the REPL logger just call `file.write(json.dumps(turn) + "\n")`), requires no schema migration, and is read directly by pandas (`pd.read_json(path, lines=True)`).

## Considered Options

**CSV** — matches the existing notebook's input format. Rejected: the two-level data model (turn contains N calls) does not flatten cleanly into a single CSV; two separate files (turns.csv + calls.csv) add join complexity with no benefit over JSONL.

**SQLite** — proper relational model with two tables. Rejected: the only consumer is the Jupyter notebook, which reads the whole dataset at once — a query engine adds no value over a flat file at this scale.
