from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import IPMDFastSACRLOptConfig

VANILLA_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_ori_b"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

VANILLA_Q_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "expert_motion"),
    ("critic", "expert_anchor_pos_b"),
    ("critic", "expert_anchor_ori_b"),
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "last_action"),
]

LATENT_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "latent_command"),
    ("policy", "projected_gravity"),
    ("policy", "base_lin_vel"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

LATENT_Q_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "latent_command"),
    ("critic", "expert_motion"),
    ("critic", "expert_anchor_pos_b"),
    ("critic", "expert_anchor_ori_b"),
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "projected_gravity"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "joint_pos"),
    ("critic", "joint_vel"),
    ("critic", "last_action"),
]

EXPERT_INPUT_KEYS: list[tuple[str, str]] = [
    ("expert_state", "joint_pos"),
    ("expert_state", "joint_vel"),
    ("expert_state", "root_pos"),
    ("expert_state", "root_quat"),
    ("expert_state", "root_lin_vel"),
    ("expert_state", "root_ang_vel"),
]


@configclass
class _G1ImitationRLOptIPMDFastSACBaseConfig(IPMDFastSACRLOptConfig):
    """Shared FastSAC+IPMD configuration for G1 imitation."""

    _use_latent_obs: bool = False

    def sync_input_keys(self) -> None:
        if self._use_latent_obs:
            self.policy.input_keys = list(LATENT_POLICY_INPUT_KEYS)
            self.q_function.input_keys = list(LATENT_Q_INPUT_KEYS)
        else:
            self.policy.input_keys = list(VANILLA_POLICY_INPUT_KEYS)
            self.q_function.input_keys = list(VANILLA_Q_INPUT_KEYS)
        self.ipmd.reward_input_keys = list(EXPERT_INPUT_KEYS)

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.q_function is not None, "Q-function configuration must be provided."

        self.sync_input_keys()

        # FastSAC rollout/optimizer shape tuned for Isaac Lab vectorized runs.
        self.collector.frames_per_batch = 1
        self.collector.total_frames = 100_000_000
        self.collector.init_random_frames = 0
        self.collector.warmup_collects = 5
        self.collector.no_cuda_sync = True
        self.replay_buffer.size = 200_000

        self.fastsac.batch_size = 8
        self.fastsac.num_updates = 8
        self.fastsac.policy_frequency = 4
        self.fastsac.actor_hidden_dim = 512
        self.fastsac.critic_hidden_dim = 768
        self.fastsac.num_q_networks = 2
        self.fastsac.num_atoms = 101
        self.fastsac.v_min = -20.0
        self.fastsac.v_max = 20.0
        self.fastsac.tau = 0.03
        self.fastsac.gamma = 0.97
        self.fastsac.learning_starts = 10
        self.fastsac.alpha_init = 0.002
        self.fastsac.use_autotune = True
        self.fastsac.target_entropy_ratio = 0.05
        self.fastsac.log_std_min = -5.0
        self.fastsac.log_std_max = 0.0
        self.fastsac.use_layer_norm = True
        self.fastsac.norm_obs = True
        self.fastsac.max_grad_norm = 0.0
        self.fastsac.num_steps = 1
        self.fastsac.amp = True
        self.fastsac.compile_updates = False

        self.optim.optimizer = "adamw"
        self.optim.lr = 3.0e-4
        self.optim.weight_decay = 0.001
        self.optim.target_update_polyak = 0.995

        self.save_interval = 5_000_000
        self.trainer.progress_bar = True
        self.trainer.log_interval = 819
        self.compile.compile = False
        self.log_level = "critical"
        self.logger.project_name = "G1-Imitation-RLOpt-IPMD-FastSAC"

        self.ipmd.reward_input_type = "s'"
        self.ipmd.reward_num_cells = (256, 256)
        self.ipmd.reward_activation = "elu"
        self.ipmd.reward_output_activation = "tanh"
        self.ipmd.reward_output_scale = 0.25
        self.ipmd.reward_loss_coeff = 1.0
        self.ipmd.reward_l2_coeff = 0.5
        self.ipmd.reward_grad_penalty_coeff = 0.0
        self.ipmd.reward_detach_features = True
        self.ipmd.estimated_reward_clamp_min = 0.0
        self.ipmd.estimated_reward_clamp_max = 0.25
        self.ipmd.est_reward_weight = 0.2
        self.ipmd.env_reward_weight = 1.0


@configclass
class G1ImitationRLOptIPMDFastSACConfig(_G1ImitationRLOptIPMDFastSACBaseConfig):
    """FastSAC+IPMD configuration for vanilla G1 imitation."""

    _use_latent_obs: bool = False


@configclass
class G1ImitationLatentRLOptIPMDFastSACConfig(_G1ImitationRLOptIPMDFastSACBaseConfig):
    """FastSAC+IPMD configuration for latent G1 imitation."""

    _use_latent_obs: bool = True
