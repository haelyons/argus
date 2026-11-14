# argus

## Output
- Code and data only. No specs, plans, reports, figures or tables.
- The repo describes only what exists. No TODOs or planned work.
- No smoke tests or test suites. Run the real thing and fix what breaks.
- Reduced data for the full sample; dense captures only for chosen proteins.
- Filter feature descriptions in code; never read them in bulk.

## Code
- Keep `scripts/s_<operation>.py` minimal. Its docstring is its question, nothing else. Its constants stay in it.
- Used twice, it moves to the core. Scripts never import from each other.
- Anything stages are compared on has one implementation, in the core.
- Fix defects in finished scripts. Do not refactor them for taste.
- Follow /andrej-karpathy-skills:karpathy-guidelines.

## Runs
- Launch only from a clean committed tree.
- After a core change, the next run includes `s_checks`: our forward pass, edits and feature values must still match Biohub's exactly.
- Every run records tok/s, op cost in forward-pass equivalents and peak memory.
- The next commit message carries the run's evidence (checks, key numbers, op costs, failures and fixes) and names the commit that ran.

## Data
- Results go to HF as `<input>/<operation>/<date>-<commit>/`, indexed by `runs.jsonl`. No prose.

## Limits
- Never print credentials.
- Spend limit: $500 without asking.
