"""Select the test week and its midday snapshots for the example-week figure
(truth only).

Among the 36 test weeks, the week with the largest number of true
voltage-violation bus-snapshots (sum of column v_true of the per-snapshot
table), ties to the earlier week. Reads only the truth-derived columns; no
estimate enters the choice.

Writes selected_ids.json in the --snapshot_ids format of run_cell.py:
  ids.week   = the 336 snapshots of the selected week
  ids.midday = its half-hours 10:00-14:00 (steps 20-28 of each day; 63 ids)

Usage: python select_week.py WEEK_IDS_JSON PER_SNAPSHOT_CSV OUT_JSON"""

import csv
import json
import sys


def main():
    week_json, per_csv, out_json = sys.argv[1:4]
    with open(week_json) as fh:
        weeks = {k: v for k, v in json.load(fh)["ids"].items() if k.startswith("w")}
    v_true, i_true = {}, {}
    with open(per_csv) as f:
        for row in csv.DictReader(f):
            s = int(row["snapshot"])
            v_true[s] = float(row["v_true"])
            i_true[s] = float(row["i_true"])
    table = []
    for key in sorted(weeks):
        ids = weeks[key]
        missing = [s for s in ids if s not in v_true]
        if missing:
            raise SystemExit(f"{key}: {len(missing)} snapshots missing in {per_csv}")
        table.append((key, sum(v_true[s] for s in ids), sum(i_true[s] for s in ids)))
    best = max(table, key=lambda r: (r[1], -int(r[0][1:])))  # ties: earlier week
    ids = weeks[best[0]]
    midday = [s for s in ids if 20 <= s % 48 <= 28]
    out = {
        "rule": "test week with the most true voltage-violation "
        "bus-snapshots (sum of v_true, stage-1 WLS cells, truth only), ties to the "
        "earlier week; midday = half-hours 10:00-14:00",
        "selected_week": best[0],
        "per_week": [{"week": k, "v_true": v, "i_true": i} for k, v, i in table],
        "ids": {"week": ids, "midday": midday},
    }
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v, i in table:
        print(
            f"{k}  v_true {v:6.0f}  i_true {i:6.0f}{'   <== selected' if k == best[0] else ''}"
        )


if __name__ == "__main__":
    main()
