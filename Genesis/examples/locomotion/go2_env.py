import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, imitation_modes=None):
        if imitation_modes is None:
            imitation_modes = ["trot"]
        self.imitation_modes = imitation_modes
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.cam = self.scene.add_camera(
            res = (640, 480),
            pos = (-3.5,-2.0,2.5),
            lookat = (0,0,1.0),
            fov = 60,
            GUI = False,
        )
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        # --- トロット報酬関数を追加 ---
        self.reward_functions["trot_imitation"] = self._reward_trot_imitation
        self.episode_sums["trot_imitation"] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        # trot_imitationのスケールも追加（デフォルト値1.0）
        if "trot_imitation" not in self.reward_scales:
            self.reward_scales["trot_imitation"] = 1.0 * self.dt
        # trot_imitationのスケールも追加
        if "trot_imitation" not in self.reward_scales:
            self.reward_scales["trot_imitation"] = 0.0
        self.reward_scales["trot_imitation"] *= self.dt

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()
        # --- ここからトロット用 ---
        self.time = 0.0  # 経過時間を管理
        # トロット報酬の重み（必要に応じてenv_cfgやreward_cfgに移動可）
        self.trot_w1 = reward_cfg.get("trot_w1", 1.0)  # 前進速度の重み
        self.trot_w2 = reward_cfg.get("trot_w2", 1.0)  # お手本誤差の重み
        self.pace_w1 = reward_cfg.get("pace_w1", 1.0)
        self.pace_w2 = reward_cfg.get("pace_w2", 1.0)
        self.bound_w1 = reward_cfg.get("bound_w1", 1.0)
        self.bound_w2 = reward_cfg.get("bound_w2", 1.0)
        self.gallop_w1 = reward_cfg.get("gallop_w1", 1.0)
        self.gallop_w2 = reward_cfg.get("gallop_w2", 1.0)
        # --- ここまで歩行調整報酬用 ---

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # --- トロット用: 時間を進める ---
        self.time += self.dt

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        # imitation_modesで指定された模倣報酬をすべて加算
        for imitation in self.imitation_modes:
            key = f"{imitation}_imitation"
            if key in self.reward_functions:
                rew = self.reward_functions[key]() * self.reward_scales.get(key, 1.0)
                self.rew_buf += rew
                self.episode_sums[key] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _get_trot_reference_angles(self):
        # サイン波でトロット歩行の目標関節角度を生成
        amplitude = 0.2  # 振幅を小さく（0.5 → 0.2）
        frequency = 1.0  # 周波数を下げる（1.5 → 1.0 Hz）
        # 12自由度: FR, FL, RR, RLの順で3関節ずつ
        # トロット歩行の位相: FR+RL（対角）とFL+RR（対角）が逆位相
        phase_offsets = torch.tensor([
            0, 0, 0,         # FR: 0度
            math.pi, math.pi, math.pi, # FL: 180度（FRと逆位相）
            math.pi, math.pi, math.pi, # RR: 180度（FRと逆位相）
            0, 0, 0          # RL: 0度（FRと同位相）
        ], device=self.device)
        t = torch.tensor(self.time, device=self.device)
        # (num_envs, 12)の目標角度
        trot_ref = amplitude * torch.sin(2 * math.pi * frequency * t + phase_offsets)
        # 複数環境対応
        return trot_ref.repeat(self.num_envs, 1)

    def _reward_trot_imitation(self):
        # お手本（サイン波）との誤差をペナルティ
        reference_angles = self._get_trot_reference_angles()
        # 実際の関節角度: self.dof_pos (num_envs, 12)
        imitation_penalty = torch.sum((self.dof_pos - reference_angles) ** 2, dim=1)
        # 前進速度: x方向
        forward_vel = self.base_lin_vel[:, 0]
        # 安定性報酬: 高さが適切な範囲にある場合に報酬
        base_height = self.base_pos[:, 2]
        height_reward = torch.exp(-torch.abs(base_height - 0.3) / 0.1)
        # 報酬 = w1*前進速度 + 安定性報酬 - w2*誤差
        return self.trot_w1 * forward_vel + height_reward - self.trot_w2 * imitation_penalty

    def _get_pace_reference_angles(self):
        # パース：同側が同時に動く
        amplitude = 0.2
        frequency = 1.0
        phase_offsets = torch.tensor([
            0, 0, 0,             # FR: 0度
            math.pi, math.pi, math.pi, # FL: 180度
            0, 0, 0,             # RR: 0度
            math.pi, math.pi, math.pi  # RL: 180度
        ], device=self.device)
        t = torch.tensor(self.time, device=self.device)
        pace_ref = amplitude * torch.sin(2 * math.pi * frequency * t + phase_offsets)
        return pace_ref.repeat(self.num_envs, 1)

    def _reward_pace_imitation(self):
        reference_angles = self._get_pace_reference_angles()
        imitation_penalty = torch.sum((self.dof_pos - reference_angles) ** 2, dim = 1)
        forward_vel = self.base_lin_vel[:, 0] #前進速度
        height_reward = torch.exp(-torch.abs(self.base_pos[:, 2] - 0.3) / 0.1) #安定性報酬
        return self.pace_w1 * forward_vel + height_reward - self.pace_w2 * imitation_penalty

    def _get_gallop_reference_angles(self):
        amplitude = 0.2
        frequency = 1.0
        # ギャロップ: 前脚（FR, FL）: 0度, 後脚（RR, RL）: π
        phase_offsets = torch.tensor([
            0, 0, 0,             # FR: 0度
            0, 0, 0,             # FL: 0度
            math.pi, math.pi, math.pi, # RR: 180度
            math.pi, math.pi, math.pi # RL: 180度
        ], device=self.device)
        t = torch.tensor(self.time, device=self.device)
        gallop_ref = amplitude * torch.sin(2 * math.pi * frequency * t + phase_offsets)
        return gallop_ref.repeat(self.num_envs, 1)

    def _reward_gallop_imitation(self):
        reference_angles = self._get_gallop_reference_angles()
        imitation_penalty = torch.sum((self.dof_pos - reference_angles) ** 2, dim = 1)
        forward_vel = self.base_lin_vel[:, 0] #前進速度
        height_reward = torch.exp(-torch.abs(self.base_pos[:, 2] - 0.3) / 0.1) #安定性報酬
        return self.gallop_w1 * forward_vel + height_reward - self.gallop_w2 * imitation_penalty

    def _get_bound_reference_angles(self):
        amplitude = 0.2
        frequency = 1.0
        # バウンド: 前脚（FR, FL）: 0度, 後脚（RR, RL）: π
        phase_offsets = torch.tensor([
            0, 0, 0,             # FR: 0度
            0, 0, 0,             # FL: 0度
            math.pi, math.pi, math.pi, # RR: 180度
            math.pi, math.pi, math.pi  # RL: 180度
        ], device=self.device)
        t = torch.tensor(self.time, device=self.device)
        bound_ref = amplitude * torch.sin(2 * math.pi * frequency * t + phase_offsets)
        return bound_ref.repeat(self.num_envs, 1)

    def _reward_bound_imitation(self):
        reference_angles = self._get_bound_reference_angles()
        imitation_penalty = torch.sum((self.dof_pos - reference_angles) ** 2, dim = 1)
        forward_vel = self.base_lin_vel[:, 0] #前進速度
        height_reward = torch.exp(-torch.abs(self.base_pos[:, 2] - 0.3) / 0.1) #安定性報酬
        return self.bound_w1 * forward_vel + height_reward - self.bound_w2 * imitation_penalty
