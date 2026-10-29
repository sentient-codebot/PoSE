"""Merge the shards of one cell (run_cell.py --shard i --num_shards n) into
one cell directory with per_snapshot.csv and metrics.json, as an unsharded run
would write them (seeds are per snapshot). Runtime is the snapshot-weighted
mean of the shards; the shard directories and their file hashes are recorded.

    python scripts/merge_shards.py --shards DIR_0 DIR_1 ... --output_dir CELL_DIR"""

import argparse
import json
import os

import pandas as pd
from run_cell import summarize

from pose.experiment import file_sha256, git_state, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    infos, tables = [], []
    for d in args.shards:
        with open(os.path.join(d, "metrics.json")) as f:
            infos.append(json.load(f))
        tables.append(pd.read_csv(os.path.join(d, "per_snapshot.csv")))
    shards = sorted(range(len(infos)), key=lambda i: infos[i]["shard"]["shard"])
    infos = [infos[i] for i in shards]
    tables = [tables[i] for i in shards]
    dirs = [args.shards[i] for i in shards]
    n = infos[0]["shard"]["num_shards"]
    if [i["shard"]["shard"] for i in infos] != list(range(n)):
        raise SystemExit(f"need shards 0..{n - 1}, got {[i[shard] for i in infos]}")
    for key in ("label", "setting", "num_samples", "num_steps", "sampler", "design"):
        if len({json.dumps(i.get(key)) for i in infos}) != 1:
            raise SystemExit(f"shards differ in {key}")
    if len({i["git"]["commit"] for i in infos}) != 1:
        raise SystemExit("shards ran at different commits")
    if len({i["data"]["sha256"] for i in infos}) != 1:
        raise SystemExit("shards used different data")

    table = pd.concat(tables, ignore_index=True)
    if table["snapshot"].duplicated().any():
        raise SystemExit("shards overlap")
    metric_cols = infos[0]["metric_cols"]
    summary, dso = summarize(table, metric_cols, infos[0]["num_samples"])
    counts = [len(t) for t in tables]
    info = {k: v for k, v in infos[0].items() if k != "shard"}
    info.update(
        {
            "summary": summary,
            "dso": dso,
            "num_snapshots": len(table),
            "runtime_per_snapshot_s": sum(
                i["runtime_per_snapshot_s"] * c
                for i, c in zip(infos, counts, strict=False)
            )
            / len(table),
            "setup_time_s": max(i["setup_time_s"] for i in infos),
            "shards": [
                {
                    "dir": d,
                    "num_snapshots": c,
                    "hardware": i["hardware"],
                    "per_snapshot_sha256": file_sha256(
                        os.path.join(d, "per_snapshot.csv")
                    ),
                    "metrics_sha256": file_sha256(os.path.join(d, "metrics.json")),
                }
                for d, i, c in zip(dirs, infos, counts, strict=False)
            ],
            "merge_git": git_state(),
        }
    )
    os.makedirs(args.output_dir, exist_ok=True)
    table.to_csv(os.path.join(args.output_dir, "per_snapshot.csv"), index=False)
    write_json(os.path.join(args.output_dir, "metrics.json"), info)
    print(f"merged {n} shards, {len(table)} snapshots -> {args.output_dir}")


if __name__ == "__main__":
    main()
