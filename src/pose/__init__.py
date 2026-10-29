"""Posterior sampling for distribution system state estimation (PoSE).

Modules:
    feeder: IEEE 33-bus data, measurement sets and noise levels.
    powerflow: batched AC power flow (backward/forward sweep) in torch.
    lindistflow: linearized power flow operator and its error statistics.
    data: snapshot data set (injections, context, split) and ground truth.
    prior: conditional flow matching prior with an MLP velocity network.
    guidance: endpoint-corrected samplers (M0-M2) and DPS (B4).
    baselines: classical, Bayesian and learned baselines (B1-B7).
    metrics: accuracy, calibration and paired bootstrap comparisons.
"""
