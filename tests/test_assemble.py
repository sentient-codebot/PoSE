"""Data assembly on a fake Fluvius archive (same long format as the real
CSV export), with 2 homes per node so the test stays small."""

import importlib.util
import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest


HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "scripts", "data", "assemble_fluvius.py")


def load_module():
    spec = importlib.util.spec_from_file_location("assemble_fluvius", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_archive(path: str, num_meters: int = 64, days: int = 31) -> None:
    rng = np.random.default_rng(0)
    stamps = pd.date_range("2024-01-01", periods=days * 96, freq="15min")
    rows = []
    for m in range(num_meters):
        pv = int(m % 2 == 0)
        hours = stamps.hour + stamps.minute / 60
        load = 0.1 + 0.05 * rng.random(len(stamps))
        gen = pv * 0.3 * np.clip(np.cos(np.pi * (hours - 12.5) / 8), 0, None)
        net = load - gen
        frame = pd.DataFrame(
            {
                "EAN_ID": f"meter{m:03d}",
                "Datum": stamps.strftime("%Y-%m-%d"),
                "Datum_Startuur": stamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Volume_Afname_kWh": np.clip(net, 0, None) / 4,
                "Volume_Injectie_kWh": np.clip(-net, 0, None) / 4,
                "Warmtepomp_Indicator": 0,
                "Elektrisch_Voertuig_Indicator": 0,
                "PV-Installatie_Indicator": pv,
                "Contract_Categorie": "Residentieel",
            }
        )
        if m == 3:  # a few missing readings
            frame = frame.drop(frame.index[100:110])
        rows.append(frame)
    buf = io.StringIO()
    pd.concat(rows).to_csv(buf, index=False)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("part1.csv", buf.getvalue())


@pytest.mark.parametrize("scenario", ["base", "high_pv"])
def test_assemble_on_fake_archive(tmp_path, monkeypatch, scenario):
    mod = load_module()
    monkeypatch.setattr(mod, "HOMES_PER_NODE", 2)
    monkeypatch.setattr(mod, "PV_PER_HEAVY_NODE_BASE", 1)
    raw = tmp_path / "raw"
    raw.mkdir()
    fake_archive(str(raw / "fake.zip"))
    meters = mod.read_meters(str(raw))
    assert len(meters["meter_id"]) == 64 and meters["attrs"][:, 0].sum() == 32
    data = mod.build(meters, scenario, seed=1)
    # January only; the 10-step gap of meter 3 leaves 1 of 2 homes (< 90%)
    assert data.valid.sum() == 31 * 96 - 10
    jan = data.p_kw[: 31 * 96]
    assert np.isfinite(jan[data.valid[: 31 * 96]]).all()
    pv_nodes = [
        mod.node_of_bus(b)
        for b in (
            mod.PV_BUSES_HIGH if scenario == "high_pv" else mod.PV_HEAVY_BUSES_BASE
        )
    ]
    pv_count = np.asarray(data.meta["pv_homes_per_node"])
    assert (pv_count[pv_nodes] >= 1).all()
    train = data.valid & (data.context["split"] == 0)
    gross_like = np.quantile(np.clip(data.p_kw[train], 0, None), 0.99, axis=0)
    assert np.all(gross_like <= np.asarray(mod.P_NOMINAL_KW) * 1.0001)
    assert (
        data.meta["rescaled_node_steps"] == 0
    )  # the gap at meter 3 invalidates its steps


def test_select_prior_link_selected(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys

    import torch

    from pose.data import synthetic_snapshots
    from pose.prior import FlowPrior, PriorConfig, save_prior

    path = os.path.join(HERE, "..", "scripts", "select_prior.py")
    spec = importlib.util.spec_from_file_location("select_prior", path)
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)

    data = synthetic_snapshots(0)
    data_path = str(tmp_path / "data" / "snapshots.npz")
    data.save(data_path)
    tr = data.indices("train")
    mean, std = data.p_kw[tr].mean(0), data.p_kw[tr].std(0)
    sweep = tmp_path / "sweep"
    for name, width in (("w16", 16), ("w32", 32)):
        torch.manual_seed(width)
        cfg = PriorConfig(width=width, num_blocks=1, emb_dim=16, context="cyclic")
        prior = FlowPrior(cfg, mean, std)
        run = sweep / name
        run.mkdir(parents=True)
        save_prior(str(run / "prior.pt"), prior, {"seed": 0})
        meta = {
            "run_name": name,
            "num_parameter": sum(p.numel() for p in prior.net.parameters()),
            "config": {k: getattr(cfg, k) for k in sp.SWEPT},
            "best_step": 0,
            "best_val_loss_ema": 0.0,
        }
        (run / "metadata.json").write_text(json.dumps(meta))

    def run_main(output, extra):
        argv = ["select_prior.py", "--data", data_path, "--sweep_dir", str(sweep)]
        argv += ["--output", output, "--report_only", "--num_draws", "1"]
        monkeypatch.setattr(sys, "argv", argv + ["--device", "cpu"] + extra)
        sp.main()
        with open(output) as f:
            return json.load(f)

    link = tmp_path / "out" / "selected_prior.pt"
    report = run_main(str(tmp_path / "sel.json"), ["--link_selected", str(link)])
    chosen = sweep / report["selected_run"] / "prior.pt"
    assert os.path.islink(link)
    assert os.path.realpath(link) == os.path.realpath(chosen)
    assert report["linked_prior"] == str(link)
    assert os.path.realpath(report["linked_prior_target"]) == os.path.realpath(chosen)
    # rerun replaces the link
    report = run_main(str(tmp_path / "sel2.json"), ["--link_selected", str(link)])
    assert os.path.realpath(link) == os.path.realpath(chosen)
    # without the flag: no link and no link keys
    link.unlink()
    report = run_main(str(tmp_path / "sel3.json"), [])
    assert not os.path.lexists(link) and "linked_prior" not in report
