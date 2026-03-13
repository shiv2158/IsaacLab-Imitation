from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import FastSACRLOptConfig
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_POLICY_OBS_KEYS,
    G1_VALUE_OBS_KEYS,
)


@configclass
class G1ImitationRLOptFastSACConfig(FastSACRLOptConfig):
    """RLOpt FastSAC configuration for G1 imitation.
    (holosoma/src/holosoma/holosoma/config_values/loco/g1/experiment.py).
    """

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, FastSACRLOptConfig)

        # Observation keys
        self.policy.input_keys = list(G1_POLICY_OBS_KEYS)
        self.q_function.input_keys = list(G1_VALUE_OBS_KEYS)

        # Network architecture (holosoma: actor_hidden_dim=512, critic_hidden_dim=768)
        self.policy.num_cells = [512, 512]
        self.q_function.num_cells = [768, 768]

        # Collector (holosoma: learning_starts=10, num_learning_iterations=50000)
        self.collector.init_random_frames = 10
        self.collector.frames_per_batch = 24
        self.collector.total_frames = 50000 * 4096 * 24

        # Replay buffer (holosoma: buffer_size=1024 per env, ~4096 envs)
        self.replay_buffer.size = 1024 * 4096

        # Loss (holosoma: gamma=0.97, batch_size=8192)
        self.loss.gamma = 0.97
        self.loss.mini_batch_size = 8192

        # Optimizer (holosoma: lr=3e-4, weight_decay=0.001, tau=0.125)
        self.optim.lr = 3e-4
        self.optim.weight_decay = 0.001
        self.optim.target_update_polyak = 1.0 - 0.125  # tau = 0.125
        self.optim.scheduler = None
        self.optim.max_grad_norm = 1.0

        # SAC (holosoma: alpha_init=0.001, log_std_min=-5.0, log_std_max=0.0)
        self.sac.alpha_init = 0.001
        self.sac.clip_log_std = True
        self.sac.log_std_min = -5.0
        self.sac.log_std_max = 0.0
        self.sac.num_qvalue_nets = 2

        # FastSAC scheduling (holosoma: num_updates=8, policy_frequency=4)
        self.sac.feature_update_ratio = 8
        self.sac.actor_update_freq = 4
        self.sac.target_update_freq = 1

        # Compilation (holosoma: compile=True)
        self.compile.compile = True
        self.save_interval = 500
