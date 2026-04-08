import os
import tempfile
from pathlib import Path

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
        # LayerNorm disabled: adds complexity without proven benefit at our UTD range (0.003).
        # Re-enable once base training is stable.
        self.q_function.use_layer_norm = False

        # Collector — warmup_collects is expanded in scripts/rlopt/train.py to
        # init_random_frames = warmup_collects * frames_per_batch (after × num_envs).
        self.collector.warmup_collects = 5
        self.collector.frames_per_batch = 24
        self.collector.total_frames = 50_000 * 4096 * 24

        # Replay buffer — match the run that reached ep_ret=9.7 (500k).
        self.replay_buffer.size = 500_000
        if self.replay_buffer.scratch_dir is None and self.collector.scratch_dir is None:
            scratch_root_env = os.environ.get("RLOPT_FASTSAC_REPLAY_SCRATCH_DIR")
            if scratch_root_env is not None and len(scratch_root_env) > 0:
                scratch_root = Path(scratch_root_env)
            elif Path("/data").is_dir():
                scratch_root = Path("/data/rlopt_replay")
            else:
                scratch_root = Path(tempfile.gettempdir()) / "rlopt_replay"

            slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
            replay_scratch_dir = scratch_root / f"rlopt_fastsac_replay_{slurm_job_id}"
            try:
                replay_scratch_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                replay_scratch_dir = Path(tempfile.gettempdir()) / (
                    f"rlopt_fastsac_replay_{slurm_job_id}"
                )
                replay_scratch_dir.mkdir(parents=True, exist_ok=True)

            self.replay_buffer.scratch_dir = str(replay_scratch_dir)

        # Loss — 512 matches the run that reached ep_ret=9.7. Larger batches (2048) combined
        # with τ=0.005 caused ep_ret to plateau at -5 in testing.
        self.loss.gamma = 0.97
        self.loss.mini_batch_size = 512

        # Optimizer — τ=0.05 is a middle ground: responsive enough for early bootstrap
        # (τ=0.005 was too slow; target barely moved over 128 steps) but less oscillatory
        # than τ=0.125 (which caused ep_ret variance ±3-4).
        self.optim.lr = 3e-4
        self.optim.weight_decay = 0.001
        self.optim.target_update_polyak = 1.0 - 0.05
        self.optim.scheduler = None
        self.optim.max_grad_norm = 1.0

        # SAC — alpha is auto-tuned toward H_target ≈ -14.5 (= -action_dim/2 via TorchRL auto).
        # alpha_init = 0.01: proven to prevent multi-joint violations in early training on this
        # task. High alpha (0.1) combined with many actor updates per batch caused Q-explosion
        # in testing; auto-tuning will raise alpha appropriately once the critic is stable.
        self.sac.alpha_init = 0.01
        self.sac.clip_log_std = True
        self.sac.log_std_min = -5.0
        self.sac.log_std_max = 2.0
        self.sac.num_qvalue_nets = 2

        # FastSAC scheduling — exact values from the run that reached ep_ret=9.7.
        # 64 critic steps / 4 = 16 actor updates per collection step.
        self.sac.feature_update_ratio = 64
        self.sac.actor_update_freq = 4
        self.sac.target_update_freq = 1

        self.compile.compile = True
        self.save_interval = 500
