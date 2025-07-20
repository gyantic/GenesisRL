import argparse
import os
import pickle
from importlib import metadata

import torch

"""
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
"""
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--imitations", type=str, default="trot", help="カンマ区切りで模倣報酬名を指定（例: trot,pace）")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # 指定されたimitationのみ報酬スケールを0.0に
    imitation_modes = [s.strip() for s in args.imitations.split(",") if s.strip()]
    reward_cfg["reward_scales"] = {f"{imit}_imitation": 0.0 for imit in imitation_modes}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        imitation_modes=imitation_modes
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy()
    env.cam.start_recording()

    obs, _ = env.reset()
    max_steps = 1000
    step_count = 0
    with torch.no_grad():
        while step_count < max_steps:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            step_count += 1
            env.cam.render()

    # 動画ファイル名に実験名を含める
    video_filename = f"video_eval_{args.exp_name}.mp4"
    env.cam.stop_recording(save_to_filename=video_filename, fps=30)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
