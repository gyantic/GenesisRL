import argparse
import pickle
import time, copy

import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner # OnPolicyRunnerをインポート
from go2_env import Go2Env
from collections import deque

def main():
    parser = argparse.ArgumentParser(description="Go2 Gait Switching Evaluation")
    parser.add_argument("--trot_model_path", type=str, required=True, help="Path to the trained climber trot model (.pt)")
    parser.add_argument("--pace_model_path", type=str, required=True, help="Path to the trained flat-ground pace model (.pt)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name to load base config from")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)
    device = gs.device # 'cpu' or 'cuda'

    # --- 1. 設定ファイルのロード ---
    print("--- Loading configs ---")
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_orig = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    env_cfg["termination_if_pitch_greater_than"] = 30.0

    # --- 2. 環境の初期化 ---
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        imitation_modes=[]
    )

    # --- 3. 2つの専門家モデルをOnPolicyRunner経由でロード ---
    print("--- Loading models ---")

    train_cfg_trot = copy.deepcopy(train_cfg_orig)
    trot_runner = OnPolicyRunner(env, train_cfg_trot, log_dir, device=device)
    trot_runner.load(args.trot_model_path)
    trot_policy = trot_runner.get_inference_policy(device=device)

    # ペース用のRunnerとPolicyを準備
    train_cfg_pace = copy.deepcopy(train_cfg_orig)
    pace_runner = OnPolicyRunner(env, train_cfg_pace, log_dir, device=device)
    pace_runner.load(args.pace_model_path)
    pace_policy = pace_runner.get_inference_policy(device=device)

    print("--- Models loaded successfully ---")

    # --- 4. メインの制御ループ ---
    obs, _ = env.reset()
    env.cam.start_recording()

    # --- 元の閾値・条件に戻す ---
    slope_detection_threshold = 6.0  # 坂道とみなす閾値
    min_slope_steps_for_trot = 5   # トロットに切り替えるための最小ステップ数
    min_flat_steps_for_pace = 10  # 坂の終わりでの復帰を早くする

    # 坂道検出用の変数
    consecutive_slope_steps = 0
    # 坂道状態の追跡
    is_on_slope = False
    slope_entry_steps = 0
    slope_exit_steps = 0

    # 現在の歩容状態を追跡
    current_gait = "pace"
    gait_switch_cooldown = 0

    max_steps = env.max_episode_length
    step_count = 0
    active_policy = pace_policy
    transition_countdown = 0
    pitch_history = deque(maxlen=5)
    with torch.no_grad():
        gait_switched_to_trot = False
        while step_count < max_steps:
            pitch_angle = env.base_euler[0, 1].item()
            pitch_history.append(pitch_angle)
            avg_pitch = sum(pitch_history) / len(pitch_history)
            previous_policy = active_policy
            previous_gait = current_gait

            # 坂道検出ロジック
            if abs(avg_pitch) > slope_detection_threshold:
                consecutive_slope_steps += 1
                slope_exit_steps = 0
                if not is_on_slope:
                    is_on_slope = True
                    slope_entry_steps = 0
                    print(f"--- ENTERING SLOPE at avg_pitch {avg_pitch:.2f}° ---")
            else:
                consecutive_slope_steps = 0
                if is_on_slope:
                    is_on_slope = False
                    slope_exit_steps = 0
                    print(f"--- EXITING SLOPE at avg_pitch {avg_pitch:.2f}° (was on slope for {slope_entry_steps} steps) ---")
                else:
                    slope_exit_steps += 1

            # 坂道にいる間はステップ数をカウント
            if is_on_slope:
                slope_entry_steps += 1

            # クールダウン期間を減らす
            if gait_switch_cooldown > 0:
                gait_switch_cooldown -= 1

            # 歩容切り替えロジック
            if gait_switch_cooldown == 0:
                if current_gait == "pace":
                    if is_on_slope and consecutive_slope_steps >= min_slope_steps_for_trot:
                        current_gait = "trot"
                        active_policy = trot_policy
                        gait_switch_cooldown = 50
                        print(f"avg_pitch: {avg_pitch:.2f} deg -> Switching to TROT (Slope confirmed)")
                        gait_switched_to_trot = True
                else:
                    if (not is_on_slope and slope_exit_steps >= min_flat_steps_for_pace):
                        current_gait = "pace"
                        active_policy = pace_policy
                        gait_switch_cooldown = 50
                        print(f"avg_pitch: {avg_pitch:.2f} deg -> Switching to PACE (Flat ground confirmed for {slope_exit_steps} steps)")

            # 中間アクションを3ステップ送る
            if gait_switched_to_trot:
                for _ in range(3):
                    actions = (pace_policy(obs) + trot_policy(obs)) / 2
                    obs, _, dones, _ = env.step(actions)
                gait_switched_to_trot = False
            else:
                actions = active_policy(obs)
                obs, _, dones, _ = env.step(actions)

            # デバッグ情報（5ステップごと）
            if step_count % 5 == 0:
                slope_status = "ON_SLOPE" if is_on_slope else "FLAT"
                print(f"Step {step_count}: Pitch={pitch_angle:.2f}°, Gait={current_gait}, Slope={slope_status}, SlopeSteps={consecutive_slope_steps}, TotalSlopeSteps={slope_entry_steps}, FlatSteps={slope_exit_steps}, Cooldown={gait_switch_cooldown}")

            env.cam.render()
            step_count += 1

    video_filename = "video_eval_switcher_demo.mp4"
    env.cam.stop_recording(save_to_filename=video_filename, fps=50)
    print(f"--- Video saved to {video_filename} ---")


if __name__ == "__main__":
    main()
