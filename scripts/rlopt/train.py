# Feiyang Wu (feiyangwu@gatech.edu), based on sb3/trian.py

"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# Isaac Sim's kit Python ships a stale `rlopt` in site-packages that can shadow the
# project's RLOpt submodule (missing FastSACRLOptConfig, etc.). Prefer the repo checkout.
_RLOPT_REPO = Path(__file__).resolve().parent.parent.parent / "RLOpt"
if (_RLOPT_REPO / "rlopt").is_dir():
    _RLOPT_ROOT = str(_RLOPT_REPO)
    if _RLOPT_ROOT not in sys.path:
        sys.path.insert(0, _RLOPT_ROOT)

import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--video_width",
    type=int,
    default=None,
    help="Optional video render width override (applies to env viewer resolution).",
)
parser.add_argument(
    "--video_height",
    type=int,
    default=None,
    help="Optional video render height override (applies to env viewer resolution).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rlopt_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--log_interval", type=int, default=100_000, help="Log data every n timesteps."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Continue the training from checkpoint.",
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="PPO",
    choices=["PPO", "SAC", "FASTSAC", "IPMD", "GAIL", "AMP", "ASE"],
    help="RLOpt algorithm to train (must match the agent config).",
)
parser.add_argument(
    "--ray-proc-id",
    "-rid",
    type=int,
    default=None,
    help="Automatically configured by Ray integration, otherwise None.",
)
# Optional RLOpt / FastSAC hyperparameters for cluster sweeps (avoid editing Hydra YAML per job).
parser.add_argument(
    "--rlopt-target-tau",
    type=float,
    default=None,
    help=(
        "Soft target update rate τ for Polyak averaging: sets "
        "optim.target_update_polyak = 1.0 - τ. "
        "Typical SAC uses τ≈0.005; your G1 config used τ=0.125 (very fast targets)."
    ),
)
parser.add_argument(
    "--rlopt-replay-size",
    type=int,
    default=None,
    help="Override replay_buffer.size (e.g. 2000000 for a larger reservoir).",
)
parser.add_argument(
    "--rlopt-feature-update-ratio",
    type=int,
    default=None,
    help="Override sac.feature_update_ratio (FastSAC critic steps per collector step).",
)
parser.add_argument(
    "--rlopt-mini-batch-size",
    type=int,
    default=None,
    help="Override loss.mini_batch_size for SGD batches from the replay buffer.",
)
parser.add_argument(
    "--wandb-exp-suffix",
    type=str,
    default=None,
    help="Appended to logger.exp_name so parallel sweeps are easy to tell apart in W&B.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import random
import time
from datetime import datetime

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import numpy as np
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, PPO, SAC, FastSAC
from rlopt.config_base import RLOptConfig, TrainerConfig
from torchrl.envs import (
    Compose,
    ExcludeTransform,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.csv import CSVLogger

torch.set_float32_matmul_precision("high")

# import logger
logger = logging.getLogger(__name__)

WANDB_BACKEND = "wandb"
WANDB_PROJECT = "FastSAC"
WANDB_ENTITY = "dheddesheimer3-georgia-institute-of-technology"
WANDB_GROUP = "g1_fastsac"

ALGORITHM_CLASS_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "FASTSAC": FastSAC,
    "IPMD": IPMD,
    "GAIL": GAIL,
    "AMP": AMP,
    "ASE": ASE,
}


def _render_frame_to_numpy(frame):
    """Convert render outputs to contiguous CPU uint8 arrays for video logging."""
    if isinstance(frame, list):
        if len(frame) == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        frame = frame[-1]
    if isinstance(frame, torch.Tensor):
        frame = frame.detach()
        if frame.is_cuda:
            frame = frame.to("cpu")
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)
        frame = frame.numpy()
    else:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
    return np.ascontiguousarray(frame)


def _infer_render_fps(env: object, default_fps: int = 30) -> int:
    """Infer render FPS from env metadata (falls back to default_fps)."""
    stack: list[object] = [env]
    visited: set[int] = set()
    while len(stack) > 0:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        metadata = getattr(current, "metadata", None)
        if isinstance(metadata, dict):
            fps = metadata.get("render_fps")
            try:
                if fps is not None and float(fps) > 0:
                    return max(1, int(round(float(fps))))
            except Exception:
                pass

        for attr_name in ("base_env", "env", "_env", "unwrapped"):
            try:
                next_obj = getattr(current, attr_name, None)
            except Exception:
                continue
            if next_obj is None:
                continue
            if isinstance(next_obj, (list, tuple)):
                stack.extend(next_obj)
            else:
                stack.append(next_obj)
    return max(1, int(default_fps))


def resolve_agent_cfg_entry_point(
    task_name: str | None, agent_entry_point: str, algorithm: str
) -> str:
    """Resolve the agent config entry point based on algorithm and task registry."""
    if agent_entry_point != "rlopt_cfg_entry_point" or task_name is None:
        return agent_entry_point
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    try:
        spec = gym.spec(task_id)
    except Exception as exc:
        logger.warning("Could not resolve task '%s' from registry: %s", task_id, exc)
        return agent_entry_point
    if spec.kwargs.get(algo_entry_point) is not None:
        if algo_entry_point != agent_entry_point:
            print(f"[INFO] Using agent config entry point: {algo_entry_point}")
        return algo_entry_point
    if algorithm != "PPO":
        logger.warning(
            "No algorithm-specific agent config for '%s' (expected '%s'); using '%s'.",
            task_id,
            algo_entry_point,
            agent_entry_point,
        )
    return agent_entry_point


args_cli.agent = resolve_agent_cfg_entry_point(
    args_cli.task, args_cli.agent, args_cli.algorithm
)


def apply_rlopt_cli_hyperparams(agent_cfg: RLOptConfig, args_cli: argparse.Namespace) -> None:
    """Apply optional CLI overrides after Hydra loads the agent config (for cluster sweeps)."""
    if args_cli.rlopt_target_tau is not None:
        tau = float(args_cli.rlopt_target_tau)
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"--rlopt-target-tau must be in (0, 1], got {tau}")
        agent_cfg.optim.target_update_polyak = 1.0 - tau
    if args_cli.rlopt_replay_size is not None:
        agent_cfg.replay_buffer.size = int(args_cli.rlopt_replay_size)
    if args_cli.rlopt_feature_update_ratio is not None:
        sac = getattr(agent_cfg, "sac", None)
        if sac is None or not hasattr(sac, "feature_update_ratio"):
            logger.warning(
                "Ignoring --rlopt-feature-update-ratio: config has no "
                "sac.feature_update_ratio (use FastSAC)."
            )
        else:
            sac.feature_update_ratio = int(args_cli.rlopt_feature_update_ratio)
    if args_cli.rlopt_mini_batch_size is not None:
        agent_cfg.loss.mini_batch_size = int(args_cli.rlopt_mini_batch_size)
    if args_cli.wandb_exp_suffix:
        base = agent_cfg.logger.exp_name
        agent_cfg.logger.exp_name = f"{base}_{args_cli.wandb_exp_suffix}"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RLOptConfig,
):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if agent_cfg.trainer is None:
        agent_cfg.trainer = TrainerConfig()
    agent_cfg.trainer.log_interval = max(1, int(args_cli.log_interval))
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg.collector.total_frames = (
            args_cli.max_iterations
            * agent_cfg.collector.total_frames
            * env_cfg.scene.num_envs
        )
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    # Convert warmup_collects → init_random_frames now that frames_per_batch is finalized.
    # warmup_collects is set in task configs; the base CollectorConfig default (1000) is
    # far too small for vectorized envs (1 batch = num_envs * fpb >> 1000).
    if agent_cfg.collector.warmup_collects is not None:
        agent_cfg.collector.init_random_frames = (
            agent_cfg.collector.warmup_collects * agent_cfg.collector.frames_per_batch
        )
    apply_rlopt_cli_hyperparams(agent_cfg, args_cli)
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(
        os.path.join("logs", "rlopt", args_cli.algorithm.lower(), args_cli.task)
    )
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    agent_cfg.logger.log_dir = log_dir
    agent_cfg.logger.backend = WANDB_BACKEND
    agent_cfg.logger.project_name = WANDB_PROJECT
    agent_cfg.logger.entity = WANDB_ENTITY
    agent_cfg.logger.group_name = WANDB_GROUP
    # log command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError(
            "DirectMARLEnv is not supported for RLOpt training yet."
        )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  # type: ignore
    start_time = time.time()

    env = IsaacLabWrapper(env)  # type: ignore
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )  # type: ignore
    )
    # NOTE: ObservationNorm (running mean/std) is strongly recommended for 29-DoF humanoid
    # (PHC, AMP, ASE all use it). However, G1 uses concatenate_terms=False so observations
    # are nested tensors per term (("policy","joint_vel_rel") etc.), not a single "policy"
    # tensor. Correct implementation requires either: (a) set concatenate_terms=True in
    # the policy obs group and update policy.input_keys=["policy"], or (b) apply separate
    # ObservationNorm per nested key. Deferred to avoid a runtime crash.
    env = TransformedEnv(
        env=env,
        transform=Compose(
            RewardSum(),  # type: ignore
            StepCounter(1000),  # type: ignore
        ),
    )

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(
        env=env,
        config=agent_cfg,  # type: ignore
    )

    # run training
    agent.train()

    # close the simulator
    env.close()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()  # type: ignore
