#!/usr/bin/env python3
"""Teach Friday to drive any GTA vehicle from Harsh's recordings (helper 19,
2026-10-25: "train a good model in my laptop for all vehicles in gta 5").

Runs ON THE LAPTOP, beside friday_laptop_agent.py, with the FridayML Python
(PyTorch on the RTX 3050):

    %USERPROFILE%\\FridayML\\py\\python.exe train_driver_net.py --vehicle car

Behaviour cloning with a small convolutional network:
  input  = two game pictures (160x90, now and NET_GAP ticks before -- so it
           can see how fast it is going) + the minimap (64x45, the GPS route)
  output = one number per axis of that vehicle (VEHICLES in the helper), -1..1:
           how much of each tick he held that axis's key, averaged over 0.9 s
           (he TAPS keys) and taken `lag` ticks AHEAD (he reacts late, so the
           picture comes before his key).
Every picture is also learned mirrored left-right with the turning axes
negated, so a left turn teaches a right turn too.

Recordings: ~/Friday_Recordings/<session>/ (helper 17+); a session without a
"vehicle" in info.json is a car. Pause-menu/big-map frames and long stretches
with no key held (on foot, cutscenes, phone) are left out. The drive is cut
into one-minute blocks: 7 in 10 train, 1 picks the best epoch and lag, 2 are
the honest test it never saw (scattered over the whole drive, because his
last minutes are not like the rest: on 10-21 the final 8 minutes were 70%
no key at all, and a "last 20%" test graded it on not driving). A few
seconds at each block's edge are dropped so a test minute never touches a
training one. Writes friday_driver_<vehicle>
.onnx + .json beside this script; the helper drives with it (mode "net")
after POST /model {"reload": true}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import friday_laptop_agent as fa                          # noqa: E402

VIEW = (160, 90)          # (width, height), = the recorder's REC_SIZE
MAP = (64, 45)
GAP = 2                   # the older picture: this many ticks (0.2 s) back
SMOOTH = 9                # ticks a target is averaged over (0.9 s)
IDLE_TICKS = 30           # no key within this many ticks either side -> left out
BLOCK = 600               # ticks in one split block (a minute)
SPLIT_GAP = 20            # ticks dropped at a block's edge (2 s)
TEST_BLOCKS, PICK_BLOCKS = (3, 8), (5,)   # of every 10 blocks


def menu_on_screen(pic: np.ndarray) -> bool:
    """GTA's pause menu / big map on a [3, h, w] picture: the white 'MAP' tab
    with the dark tab bar beside it (same test as the drive supervisor)."""
    a = pic.mean(axis=0)
    h, w = a.shape
    tab = a[int(0.17 * h):int(0.20 * h) + 1, int(0.17 * w):int(0.26 * w)]
    bar = a[int(0.17 * h):int(0.20 * h) + 1, int(0.40 * w):int(0.48 * w)]
    return bool((tab > 200).mean() > 0.6 and (bar < 70).mean() > 0.6)


def smooth(v: np.ndarray, n: int = SMOOTH) -> np.ndarray:
    if len(v) < n:
        return v.astype(np.float32)
    pad = n // 2
    return np.convolve(np.pad(v.astype(np.float32), pad, mode="edge"),
                       np.ones(n, np.float32) / n, mode="valid")


def load_session(folder: Path, axes) -> dict | None:
    """Pictures, minimaps and per-axis key signals of one recording, in
    stretches of unbroken recording (the game left the front in between)."""
    views, maps, raw, t = [], [], [], []
    for lab in sorted(folder.glob("labels_*.jsonl")):
        blob = (folder / lab.name.replace("labels_", "frames_").replace(".jsonl", ".bin")).read_bytes()
        for line in lab.read_text().splitlines():
            row = json.loads(line)
            v, m = row["view"], row["map"]
            views.append(fa.net_picture(v["w"], v["h"], zlib.decompress(blob[v["off"]:v["off"] + v["len"]]), VIEW))
            maps.append(fa.net_picture(m["w"], m["h"], zlib.decompress(blob[m["off"]:m["off"] + m["len"]]), MAP))
            held = set(row["keys"])                     # 'wd' (helper 17) or ['w', 'num8']
            raw.append([(1.0 if pos in held else 0.0) - (1.0 if neg in held else 0.0)
                        for _, neg, pos in axes])
            t.append(float(row["t"]))
    if not views:
        return None
    return {"views": np.stack(views), "maps": np.stack(maps),
            "raw": np.array(raw, np.float32), "t": np.array(t)}


def targets(raw: np.ndarray, t: np.ndarray, lag: int):
    """Per frame: each axis averaged over SMOOTH ticks, LAG ticks ahead, inside
    its own unbroken stretch; and the index of the picture GAP ticks before."""
    n = len(raw)
    starts = [0] + [i for i in range(1, n) if t[i] - t[i - 1] > 0.5] + [n]
    y = np.zeros_like(raw)
    prev = np.zeros(n, np.int64)
    for a, b in zip(starts[:-1], starts[1:]):
        seg = np.stack([smooth(raw[a:b, k]) for k in range(raw.shape[1])], axis=1)
        idx = np.minimum(np.arange(b - a) + lag, b - a - 1)
        y[a:b] = seg[idx]
        prev[a:b] = np.maximum(np.arange(a, b) - GAP, a)
    return y, prev


class DriverNet(nn.Module):
    def __init__(self, n_out: int) -> None:
        super().__init__()

        def block(cin: int, cout: int, k: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, k, 2, k // 2), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True))
        # 90x160 -> 45x80 -> 23x40 -> 12x20 -> 6x10 -> 3x5
        self.view = nn.Sequential(block(6, 24, 5), block(24, 32, 5), block(32, 48, 3),
                                  block(48, 64, 3), nn.AvgPool2d(2), nn.Flatten())
        # 45x64 -> 23x32 -> 12x16 -> 6x8 -> 2x2
        self.map = nn.Sequential(block(3, 16, 3), block(16, 32, 3), block(32, 32, 3),
                                 nn.AvgPool2d((3, 4)), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 * 15 + 32 * 4, 128),
                                  nn.ReLU(inplace=True), nn.Dropout(0.2),
                                  nn.Linear(128, n_out), nn.Tanh())

    def forward(self, view: torch.Tensor, mini: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.view(view), self.map(mini)], dim=1))


def scores(pred: np.ndarray, y: np.ndarray, base: np.ndarray, names) -> dict:
    out = {}
    for k, name in enumerate(names):
        p, t = pred[:, k], y[:, k]
        clear = np.abs(t) > 0.3
        out[name] = {
            "error": round(float(np.abs(p - t).mean()), 3),
            "constant_guess": round(float(np.abs(base[k] - t).mean()), 3),
            "correlation": round(float(np.corrcoef(p, t)[0, 1]) if t.std() > 0 and p.std() > 0 else 0.0, 3),
            "right_direction": round(float((np.sign(p[clear]) == np.sign(t[clear])).mean()) if clear.any() else 0.0, 3),
            "clear_frames": int(clear.sum()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="car", choices=sorted(fa.VEHICLES))
    ap.add_argument("--root", default=str(Path.home() / "Friday_Recordings"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lags", default="0,2,4", help="ticks ahead to try; the pick set chooses")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    axes = fa.VEHICLES[args.vehicle]
    names = [a[0] for a in axes]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"vehicle {args.vehicle}, axes {names}, on {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'the processor'}")

    sessions = []
    for folder in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        try:
            info = json.loads((folder / "info.json").read_text())
        except (OSError, ValueError):
            continue
        if info.get("vehicle", "car") != args.vehicle:
            continue
        t0 = time.time()
        got = load_session(folder, axes)
        if got is None:
            continue
        menu = np.array([menu_on_screen(v) for v in got["views"]])
        active = np.abs(got["raw"]).sum(axis=1) > 0
        near = np.convolve(active.astype(np.float32), np.ones(2 * IDLE_TICKS + 1), mode="same") > 0
        got["keep"] = ~menu & near
        print(f"{folder.name}: {len(got['raw'])} frames, {int(menu.sum())} menu, "
              f"{int((~near).sum())} idle, {int(got['keep'].sum())} kept ({time.time() - t0:.0f} s)")
        sessions.append(got)
    if not sessions:
        print(f"no {args.vehicle} recordings in {args.root}")
        return 1

    # everything on the graphics card as uint8; frames indexed globally
    views = torch.from_numpy(np.concatenate([s["views"] for s in sessions])).to(dev)
    maps = torch.from_numpy(np.concatenate([s["maps"] for s in sessions])).to(dev)
    print(f"{len(views)} frames on {dev} ({(views.numel() + maps.numel()) / 1e6:.0f} MB)")
    mirror = torch.tensor([1.0 if n in fa.MIRRORED_AXES else 0.0 for n in names], device=dev)

    def split_ids(lag: int):
        parts = {"train": [], "pick": [], "test": []}
        ys, prevs, off = [], [], 0
        for s in sessions:
            y, prev = targets(s["raw"], s["t"], lag)
            ys.append(y)
            prevs.append(prev + off)
            n = len(y)
            for lo in range(0, n, BLOCK):
                k = (lo // BLOCK) % 10
                name = "test" if k in TEST_BLOCKS else "pick" if k in PICK_BLOCKS else "train"
                ids = np.arange(lo + SPLIT_GAP, min(n, lo + BLOCK) - SPLIT_GAP)
                if len(ids):
                    parts[name].append(ids[s["keep"][ids]] + off)
            off += n
        return ({k: np.concatenate(v) for k, v in parts.items()},
                torch.from_numpy(np.concatenate(ys)).to(dev), torch.from_numpy(np.concatenate(prevs)).to(dev))

    def batch(ids: torch.Tensor, prev: torch.Tensor, augment: bool):
        v = torch.cat([views[ids], views[prev[ids]]], dim=1).float() / 255.0
        m = maps[ids].float() / 255.0
        if augment:
            k = len(ids)
            gain = torch.empty(k, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
            shift = torch.empty(k, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
            v = (v * gain + shift).clamp(0, 1)
            m = (m * gain + shift).clamp(0, 1)
        return v, m

    def predict(net, ids, prev):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), 512):
                part = torch.from_numpy(ids[i:i + 512]).to(dev)
                v, m = batch(part, prev, False)
                out.append(net(v, m).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(names)), np.float32)

    best = None
    for lag in [int(x) for x in args.lags.split(",")]:
        ids, Y, prev = split_ids(lag)
        ytr = Y[torch.from_numpy(ids["train"]).to(dev)]
        # turns are rarer than straight driving: weigh a frame by how hard it turns
        weight_axis = 1.0 + 2.0 * mirror
        torch.manual_seed(0)
        net = DriverNet(len(names)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = args.epochs * max(1, len(ids["train"]) // args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.2)
        train_ids = torch.from_numpy(ids["train"]).to(dev)
        y_pick = Y[torch.from_numpy(ids["pick"]).to(dev)].cpu().numpy()
        run_best = (1e9, None, 0)
        t0 = time.time()
        for epoch in range(args.epochs):
            net.train()
            order = train_ids[torch.randperm(len(train_ids), device=dev)]
            for i in range(0, len(order) - args.batch + 1, args.batch):
                b = order[i:i + args.batch]
                v, m = batch(b, prev, True)
                y = Y[b]
                flip = torch.rand(len(b), device=dev) < 0.5
                v = torch.where(flip[:, None, None, None], v.flip(-1), v)
                m = torch.where(flip[:, None, None, None], m.flip(-1), m)
                y = torch.where(flip[:, None] & (mirror[None] > 0), -y, y)
                w = 1.0 + weight_axis * y.abs()
                loss = (loss_fn(net(v, m), y) * w).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            pick = predict(net, ids["pick"], prev)
            pick_err = float(np.abs(pick - y_pick).mean())
            if pick_err < run_best[0]:
                run_best = (pick_err, {k: v.detach().clone() for k, v in net.state_dict().items()}, epoch)
        net.load_state_dict(run_best[1])
        print(f"lag {lag}: best epoch {run_best[2] + 1}/{args.epochs}, pick-set error {run_best[0]:.3f} "
              f"({time.time() - t0:.0f} s)")
        if best is None or run_best[0] < best["pick_err"]:
            best = {"pick_err": run_best[0], "lag": lag, "net": net, "ids": ids, "Y": Y, "prev": prev,
                    "base": ytr.mean(dim=0).cpu().numpy()}

    net, ids, Y, prev = best["net"], best["ids"], best["Y"], best["prev"]
    y_train = Y[torch.from_numpy(ids["train"]).to(dev)].cpu().numpy()
    fit = scores(predict(net, ids["train"], prev), y_train, best["base"], names)
    print("on the minutes it trained on (does it learn at all?): " + "; ".join(
        f"{n} correlation {sc['correlation']:+.2f}, right direction {sc['right_direction']:.0%}"
        for n, sc in fit.items()))
    y_test = Y[torch.from_numpy(ids["test"]).to(dev)].cpu().numpy()
    result = scores(predict(net, ids["test"], prev), y_test, best["base"], names)
    print(f"chosen lag {best['lag']} ticks. On {len(ids['test'])} frames of his driving it never saw:")
    for name, sc in result.items():
        print(f"  {name}: error {sc['error']} (always the same guess: {sc['constant_guess']}), "
              f"correlation {sc['correlation']:+.2f}, right direction on {sc['right_direction']:.0%} "
              f"of {sc['clear_frames']} clear frames")

    # export: float32 0..1 inputs, the helper divides by 255 exactly like batch()
    net = net.to("cpu").eval()
    onnx_path = HERE / fa.NET_FILE.format(args.vehicle)
    tmp = onnx_path.with_suffix(".tmp.onnx")
    dummy = (torch.zeros(1, 6, VIEW[1], VIEW[0]), torch.zeros(1, 3, MAP[1], MAP[0]))
    torch.onnx.export(net, dummy, str(tmp), input_names=["view", "map"], output_names=["axes"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    sample = torch.from_numpy(ids["test"][:16])
    views_cpu, maps_cpu, prev_cpu = views.cpu(), maps.cpu(), prev.cpu()
    v = torch.cat([views_cpu[sample], views_cpu[prev_cpu[sample]]], dim=1).float() / 255.0
    m = maps_cpu[sample].float() / 255.0
    with torch.no_grad():
        want = net(v, m).numpy()
    got = np.concatenate([sess.run(None, {"view": v[i:i + 1].numpy(), "map": m[i:i + 1].numpy()})[0]
                          for i in range(len(sample))])
    diff = float(np.abs(want - got).max())
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, {"view": v[:1].numpy(), "map": m[:1].numpy()})
    ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"onnx matches torch to {diff:.1e}; {ms:.1f} ms a tick on the processor")
    if diff > 1e-3:
        print("the exported network does not match -- not saved")
        return 1
    tmp.replace(onnx_path)
    meta = {"vehicle": args.vehicle, "axes": [list(a) for a in axes], "view": list(VIEW), "map": list(MAP),
            "gap": GAP, "lag": best["lag"], "smooth": SMOOTH, "made": time.strftime("%Y-%m-%d %H:%M"),
            "frames": {k: int(len(v)) for k, v in ids.items()}, "scores": result, "ms": round(ms, 2)}
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {onnx_path.name} + .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
