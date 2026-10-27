#!/usr/bin/env python3
"""A GYRO-INPUT policy (2026-10-27, Harsh: "train the model like with gyro
also"): a model exactly one input wider than the 146 of the LiDAR + frontier
policy gets the measured turn rate as input 144 (between the frontier rays and
the last action). Every other model must run exactly as before, and a width
that is neither must still be refused.

    python3 tests/test_gyro_input.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pi_deploy as pd                                    # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or not detail else f"\n        {detail}"))


def model(width: int, folder: Path) -> Path:
    """A policy of that input width: obs -> (v, omega) = (0, 0)."""
    w = numpy_helper.from_array(np.zeros((width, 2), np.float32), "W")
    g = helper.make_graph([helper.make_node("MatMul", ["obs", "W"], ["act"])], "p",
                          [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", width])],
                          [helper.make_tensor_value_info("act", TensorProto.FLOAT, ["batch", 2])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path = folder / f"policy_{width}.onnx"
    onnx.save(m, str(path))
    return path


tmp = Path(tempfile.mkdtemp())

print("which models get the gyro")
cfg = pd.RobotConfig()
pd.PolicyRunner(model(146, tmp), cfg, threads=1)
check("a 146-input model: no gyro input, width 146", not cfg.gyro_input and cfg.obs_dim == 146)
cfg = pd.RobotConfig()
pd.PolicyRunner(model(147, tmp), cfg, threads=1)
check("a 147-input model: gyro input on, width 147", cfg.gyro_input and cfg.obs_dim == 147)
for width in (148, 74):
    cfg = pd.RobotConfig()
    try:
        pd.PolicyRunner(model(width, tmp), cfg, threads=1)
        refused = False
    except ValueError:
        refused = True
    check(f"a {width}-input model is still refused", refused and not cfg.gyro_input)

print("where the gyro goes")
n = 72
for gyro in (False, True):
    cfg = pd.RobotConfig()
    cfg.gyro_input = gyro
    stub = SimpleNamespace(cfg=cfg, _obs=np.zeros(cfg.obs_dim, np.float32),
                           last_action=np.array([0.25, -0.5], np.float32), _gyro_obs=0.75,
                           map=SimpleNamespace(update_and_frontier=lambda s: np.full(n, 2.5, np.float32)))
    obs = pd.ControlLoop._build_observation(stub, np.full(n, 1.0, np.float32))
    if gyro:
        check("gyro model: scan, frontier, the turn rate at 144, last action at the end",
              obs.shape == (147,) and abs(obs[0] - 0.2) < 1e-6 and abs(obs[n] - 0.5) < 1e-6
              and obs[2 * n] == np.float32(0.75) and list(obs[-2:]) == [0.25, -0.5], str(obs[140:]))
    else:
        check("plain model: the same 146 as before, no turn rate anywhere",
              obs.shape == (146,) and list(obs[-2:]) == [0.25, -0.5] and 0.75 not in obs, str(obs[140:]))

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
