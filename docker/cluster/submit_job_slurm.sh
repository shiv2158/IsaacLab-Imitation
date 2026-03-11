#!/usr/bin/env bash

# Load cluster config from synced workspace if available.
if [ -f "$1/docker/cluster/.env.cluster" ]; then
	# shellcheck disable=SC1090
	source "$1/docker/cluster/.env.cluster"
fi

slurm_gpus_per_node="${CLUSTER_SLURM_GPUS_PER_NODE:-rtx_6000:1}"
slurm_nodes="${CLUSTER_SLURM_NODES:-1}"
slurm_mem_per_gpu="${CLUSTER_SLURM_MEM_PER_GPU:-24G}"
slurm_time="${CLUSTER_SLURM_TIME:-8:00:00}"
slurm_partition="${CLUSTER_SLURM_PARTITION:-}"
slurm_account="${CLUSTER_SLURM_ACCOUNT:-}"
slurm_mail_type="${CLUSTER_SLURM_MAIL_TYPE:-}"
slurm_mail_user="${CLUSTER_SLURM_MAIL_USER:-}"

cat <<EOT > job.sh
#!/bin/bash

<<<<<<< HEAD
#SBATCH --gpus-per-node=l40s:1
#SBATCH -N1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=16:00:00
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output="output_%j.log"
#SBATCH --error="error_%j.log"
=======
#SBATCH --gpus-per-node=${slurm_gpus_per_node}
#SBATCH -N${slurm_nodes}
#SBATCH --mem-per-gpu=${slurm_mem_per_gpu}
#SBATCH --time=${slurm_time}
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
$( [ -n "$slurm_partition" ] && echo "#SBATCH --partition=${slurm_partition}" )
$( [ -n "$slurm_account" ] && echo "#SBATCH --account=${slurm_account}" )
$( [ -n "$slurm_mail_type" ] && [ -n "$slurm_mail_user" ] && echo "#SBATCH --mail-type=${slurm_mail_type}" )
$( [ -n "$slurm_mail_type" ] && [ -n "$slurm_mail_user" ] && echo "#SBATCH --mail-user=${slurm_mail_user}" )
>>>>>>> donald_20260304

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT
sbatch < job.sh
rm job.sh
