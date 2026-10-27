#!/usr/bin/env python3
"""Fine-tune Harsh's PPO exploration model on Friday's real body (2026-10-25).

Runs ON THE LAPTOP with the FridayML Python, beside friday_body_env.py:

    %USERPROFILE%\\FridayML\\py\\python.exe finetune.py --timesteps 3000000
    %USERPROFILE%\\FridayML\\py\\python.exe finetune.py --compare <run folder>

It drives HIS train.py (main, PPO, checkpoints, export) with two things swapped:
the environment is friday_body_env.FridayExplorationEnv, and EVERY file it
writes goes to a new run folder under FridayML\\explore_runs\\. It starts from
COPIES of his ppo_exploration_model.zip and vecnormalize.pkl: the originals in
his home folder, his checkpoints\\ and his exploration_model.onnx are never
written. --compare scores the original and the fine-tuned policy on the same
Friday-body rooms (same seeds): area seen, collisions, how far she got.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path.home()))

import friday_body_env as fb                              # noqa: E402
import friday_body_env3 as fb3                            # noqa: E402
import train as T                                         # noqa: E402

ENVS = {1: (fb.FridayExplorationEnv, fb.make_env_fn), 3: (fb3.Friday3Env, fb3.make_env_fn)}

RUNS = Path.home() / "FridayML" / "explore_runs"
FINETUNE_LR = 1e-4        # his runs used 3e-4 from scratch; a fine-tune nudges, it does not relearn


def lr_schedule(progress_remaining: float) -> float:
    return FINETUNE_LR * progress_remaining


def clip_schedule(_progress_remaining: float) -> float:
    return 0.2


def safe_ppo_load() -> None:
    """His zip carries its schedules as pickled bytecode from the Python he trained
    with; FridayML's 3.12 crashes reading it ("Windows fatal exception: access
    violation" in train.py's f). Every PPO.load gets fresh schedules instead --
    the weights and the normaliser are untouched."""
    from stable_baselines3 import PPO
    original = PPO.load.__func__

    def load(cls, path, *a, **k):
        k.setdefault("custom_objects", {"learning_rate": lr_schedule, "lr_schedule": lr_schedule,
                                        "clip_range": clip_schedule})
        return original(cls, path, *a, **k)
    PPO.load = classmethod(load)


ORIGINAL = Path.home() / "ppo_exploration_model.zip"
ORIGINAL_NORM = Path.home() / "vecnormalize.pkl"


def point_train_at(run: Path, env: int = 1) -> None:
    """Every path train.py writes, inside `run`; the Friday body for every env."""
    T.MODEL_PATH = run / "ppo_exploration_model.zip"
    T.ONNX_PATH = run / "exploration_model.onnx"
    T.VECNORM_PATH = run / "vecnormalize.pkl"
    T.RUN_DIR = run / "runs"
    T.CKPT_DIR = run / "checkpoints"
    T.DiffDriveExplorationEnv, T.make_env_fn = ENVS[env]    # --check-env and anything else


def widen(src: Path, dst: Path) -> bool:
    """Round 3's gyro input: a 146-input policy becomes a 147-input one whose
    new input (GYRO_AT, before the last action) starts with zero weight, so it
    begins exactly where `src` was and learns to use the gyro from there.
    True if it was widened (its VecNormalize statistics no longer fit)."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    old = PPO.load(str(src), device="cpu")
    if old.observation_space.shape[0] != fb3.GYRO_AT + 2:
        shutil.copy2(src, dst)
        return False
    env = DummyVecEnv([lambda: fb3.Friday3Env(T.EnvConfig())])
    new = PPO("MlpPolicy", env, policy_kwargs=old.policy_kwargs, learning_rate=lr_schedule,
              n_steps=old.n_steps, batch_size=old.batch_size, n_epochs=old.n_epochs, gamma=old.gamma,
              gae_lambda=old.gae_lambda, clip_range=clip_schedule, ent_coef=old.ent_coef,
              vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm, target_kl=old.target_kl,
              device="cpu", seed=0)
    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
    for key, w in old_sd.items():
        if new_sd[key].shape == w.shape:
            new_sd[key] = w.clone()
        elif w.dim() == 2 and new_sd[key].shape == (w.shape[0], w.shape[1] + 1):
            wide = torch.zeros_like(new_sd[key])
            wide[:, :fb3.GYRO_AT] = w[:, :fb3.GYRO_AT]
            wide[:, fb3.GYRO_AT + 1:] = w[:, fb3.GYRO_AT:]
            new_sd[key] = wide
        else:
            raise ValueError(f"cannot widen {key}: {tuple(w.shape)} -> {tuple(new_sd[key].shape)}")
    new.policy.load_state_dict(new_sd)
    new.save(str(dst))
    env.close()
    return True


def score(model_zip: Path, episodes: int, seed: int, steps: int, env_no: int = 1) -> dict:
    """The deterministic policy on `episodes` Friday-body rooms. A 146-input
    policy in round 3's rooms is simply not shown the gyro."""
    import numpy as np
    from stable_baselines3 import PPO
    model = PPO.load(str(model_zip), device="cpu")
    width = model.observation_space.shape[0]
    cfg = T.EnvConfig(max_episode_steps=steps, coverage_target=1.0)    # never "succeeds" early
    env = ENVS[env_no][0](cfg)
    seen, crashes, driven, still, backing, house = [], 0, [], [], [], []
    by_scene: dict[str, list] = {}
    stops = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        start = env.pose[:2].copy()
        path, prev, idle, back = 0.0, env.pose[:2].copy(), 0, 0
        for _ in range(steps):
            if obs.shape[0] > width:
                obs = np.delete(obs, fb3.GYRO_AT)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            back += bool(getattr(env, "_reversed", False))
            step = float(np.linalg.norm(env.pose[:2] - prev))
            path += step
            idle += step < 0.002
            prev = env.pose[:2].copy()
            if term or trunc:
                crashes += bool(info.get("collision"))
                break
        seen.append(env._known_free_count * env.cell_area)
        driven.append(path)
        still.append(idle / max(1, env._steps))
        backing.append(back / max(1, env._steps))
        if getattr(env, "house", False):
            house.append(seen[-1])
        stops.append(getattr(env, "interventions", 0) / max(1, env._steps))
        crashed = bool(info.get("collision")) if (term or trunc) else False
        by_scene.setdefault(getattr(env, "scene", "room"), []).append((seen[-1], crashed))
    out = {"area_m2": round(float(np.mean(seen)), 2), "crashes": crashes, "episodes": episodes,
           "driven_m": round(float(np.mean(driven)), 2), "still": round(float(np.mean(still)), 2),
           "reversing": round(float(np.mean(backing)), 3),
           "house_area_m2": round(float(np.mean(house)), 2) if house else None, "houses": len(house)}
    if env_no == 3:
        out["safety_stops"] = round(float(np.mean(stops)), 3)
        out["by_room"] = {k: {"n": len(v), "area_m2": round(float(np.mean([a for a, _ in v])), 1),
                              "crashes": sum(c for _, c in v)} for k, v in sorted(by_scene.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--max-episode-steps", type=int, default=1500)
    ap.add_argument("--compare", default="", help="a run folder: score its best model against --against")
    ap.add_argument("--against", default=str(ORIGINAL), help="the model --compare scores it against")
    ap.add_argument("--from", dest="start", default=str(ORIGINAL),
                    help="the model to fine-tune (round 2: round 1's best_model.zip)")
    ap.add_argument("--from-norm", default=str(ORIGINAL_NORM), help="its VecNormalize statistics")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--env", type=int, default=3, choices=sorted(ENVS),
                    help="1 = rounds 1-2's body; 3 = round 3: gyro input, the safety layers, the 2-min rooms")
    args = ap.parse_args()
    safe_ppo_load()

    if args.compare:
        run = Path(args.compare)
        best = run / "checkpoints" / "best" / "best_model.zip"
        new = best if best.exists() else run / "ppo_exploration_model.zip"
        result = {"against": args.against,
                  "before": score(Path(args.against), args.episodes, 5000, args.max_episode_steps, args.env),
                  "fine_tuned": score(new, args.episodes, 5000, args.max_episode_steps, args.env),
                  "rooms": f"{args.episodes} Friday-body rooms, seeds 5000+, {args.max_episode_steps} steps"}
        (run / "compare.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return 0

    run = RUNS / time.strftime("run-%Y%m%d-%H%M")
    run.mkdir(parents=True, exist_ok=False)
    widened = widen(Path(args.start), run / "ppo_exploration_model.zip") if args.env == 3 else False
    if not widened:
        if args.env != 3:
            shutil.copy2(args.start, run / "ppo_exploration_model.zip")
        shutil.copy2(args.from_norm, run / "vecnormalize.pkl")    # a widened model starts a fresh one
    point_train_at(run, args.env)
    (run / "about.txt").write_text(
        f"fine-tune of {args.start} on Friday's body (env {args.env}, widened {widened}), started "
        f"{time.strftime('%Y-%m-%d %H:%M')}, {args.timesteps} steps\n")
    print(f"run folder: {run}", flush=True)
    return T.main(["--resume", "--timesteps", str(args.timesteps), "--n-envs", str(args.n_envs),
                   "--max-episode-steps", str(args.max_episode_steps), "--coverage-target", "0.7",
                   "--eval-freq", "100000", "--checkpoint-freq", "250000"])


if __name__ == "__main__":
    raise SystemExit(main())
