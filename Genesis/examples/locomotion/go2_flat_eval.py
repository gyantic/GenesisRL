import argparse
import pickle
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser(description="Go2 平地速度・エネルギー評価")
    parser.add_argument("--model_path", type=str, required=True, help="評価するモデルのパス(.pt)")
    parser.add_argument("--exp_name", type=str, required=True, help="設定ファイルの実験名")
    parser.add_argument("--max_steps", type=int, default=1000, help="評価ステップ数")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)
    device = gs.device

    # --- 設定ファイルロード ---
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_orig = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # --- 坂道環境用設定例（コメントアウトで残す） ---
    # env_cfg["terrain_type"] = "slope"
    # env_cfg["slope_angle"] = 10.0
    # env_cfg["terrain_length"] = 5.0

    # --- 平地用 ---
    # 何も terrain/slope を指定しなければデフォルトで平地

    # --- 平地専用にterrain/slope関連を除外 ---
    for k in list(env_cfg.keys()):
        if "slope" in k or "terrain" in k:
            del env_cfg[k]

    # --- 環境初期化 ---
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        imitation_modes=[]
    )

    # --- モデルロード ---
    runner = OnPolicyRunner(env, train_cfg_orig, log_dir, device=device)
    runner.load(args.model_path)
    policy = runner.get_inference_policy(device=device)

    # --- 評価ループ ---
    obs, _ = env.reset()
    velocities = []
    torques = []
    with torch.no_grad():
        for step in range(args.max_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            velocities.append(env.base_lin_vel[0, 0].item())
            # トルク（絶対値和）
            if hasattr(env.robot, 'get_dofs_force'):
                torque = env.robot.get_dofs_force()
                torques.append(torch.abs(torch.tensor(torque)).sum().item())
            elif hasattr(env.robot, 'control_dofs_force'):
                torque = env.robot.control_dofs_force
                torques.append(torch.abs(torch.tensor(torque)).sum().item())
    mean_speed = sum(velocities) / len(velocities)
    mean_torque = sum(torques) / len(torques) if torques else 0.0
    print(f"平均速度: {mean_speed:.3f} m/s, 平均トルク絶対値和: {mean_torque:.3f}")

if __name__ == "__main__":
    main()
