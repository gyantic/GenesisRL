import argparse
import pickle
import time
import copy
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from go2_env import Go2Env

def main():
    parser = argparse.ArgumentParser(description="Go2 Gait Switching Evaluation (Run until Goal Reached)")
    parser.add_argument("--trot_model_path", type=str, required=True, help="Path to the trained climber trot model (.pt)")
    parser.add_argument("--pace_model_path", type=str, required=True, help="Path to the trained flat-ground pace model (.pt)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name to load base config from")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)
    device = gs.device

    # --- Config and Model Loading ---
    print("--- Loading models and configs ---")
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_orig = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    env_cfg["termination_if_pitch_greater_than"] = 30.0

    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False, imitation_modes=[])

    train_cfg_trot = copy.deepcopy(train_cfg_orig)
    trot_runner = OnPolicyRunner(env, train_cfg_trot, log_dir, device=device)
    trot_runner.load(args.trot_model_path)
    trot_policy = trot_runner.get_inference_policy(device=device)

    train_cfg_pace = copy.deepcopy(train_cfg_orig)
    pace_runner = OnPolicyRunner(env, train_cfg_pace, log_dir, device=device)
    pace_runner.load(args.pace_model_path)
    pace_policy = pace_runner.get_inference_policy(device=device)
    print("--- Models loaded successfully ---")

    # --- ★★★ コースのゴールラインを設定 ★★★ ---
    GOAL_X_POSITION = 17.0 # 最後の平地（中心15.0, 長さ5.0）の終端少し手前

    while True:
        print("\n--- Starting new evaluation run... ---")
        obs, _ = env.reset()
        env.cam.start_recording()

        total_energy = 0.0
        start_time = time.time()
        goal_reached = False # ゴールしたかどうかを判定するフラグ

        active_policy = pace_policy
        transition_countdown = 0
        step_count = 0
        SLOPE_THRESHOLD = 6.8

        with torch.no_grad():
            while step_count < env.max_episode_length:
                pitch_angle = env.base_euler[0, 1].item()
                previous_policy = active_policy

            previous_policy = active_policy

            if abs(pitch_angle) > SLOPE_THRESHOLD:
                active_policy = trot_policy
            else:
                active_policy = pace_policy

            if active_policy is not previous_policy:
                print(f"--- POLICY SWITCHED! Resetting action memory. ---")
                # 観測情報(obs)の最後の12個の要素（=直前の行動）をゼロにする
                obs[0, -12:] = 0.0


            # 4. 選択したポリシーで行動を予測・実行
            actions = active_policy(obs)

            obs, _, dones, infos = env.step(actions)

            total_energy += torch.sum(torch.square(infos["torques"][0])).item() * env.dt
            step_count += 1

                # --- ★★★ ゴール判定ロジック ★★★ ---
            current_x_pos = env.base_pos[0, 0].item()
            if current_x_pos > GOAL_X_POSITION:
                print(f"--- GOAL REACHED at X={current_x_pos:.2f}! Run successful. ---")
                goal_reached = True
                break # ゴールしたので、この試行は成功としてループを抜ける

            if dones[0]:
                print(f"--- Episode terminated prematurely at step {step_count}. Run failed. ---")
                break

        if goal_reached:
            print("\n---!!! SUCCESSFUL RUN CAPTURED !!!---")
            end_time = time.time()
            traversal_time = end_time - start_time

            print("\n--- PERFORMANCE EVALUATION RESULTS ---")
            print(f"  Total Traversal Time: {traversal_time:.2f} seconds")
            print(f"  Total Energy Consumed: {total_energy:.2f} J (approx.)")
            print("------------------------------------")

            video_filename = "video_eval_SUCCESSFUL_RUN.mp4"
            env.cam.stop_recording(save_to_filename=video_filename, fps=50)
            print(f"--- Success video saved to {video_filename}. Exiting. ---")

            break # 成功したので、外側の 'while True' ループを抜ける
        else:
            print("--- Run failed before reaching goal. Retrying in 3 seconds... ---")
            env.cam.discard_recording()
            time.sleep(3)

if __name__ == "__main__":
    main()
