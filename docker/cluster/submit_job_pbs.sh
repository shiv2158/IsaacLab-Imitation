#!/usr/bin/env bash

# Load cluster config from synced workspace if available.
if [ -f "$1/docker/cluster/.env.cluster" ]; then
	# shellcheck disable=SC1090
	source "$1/docker/cluster/.env.cluster"
fi

pbs_select="${CLUSTER_PBS_SELECT:-1:ncpus=8:mpiprocs=1:ngpus=1}"
pbs_walltime="${CLUSTER_PBS_WALLTIME:-24:00:00}"
pbs_queue="${CLUSTER_PBS_QUEUE:-gpu}"
pbs_job_name="${CLUSTER_PBS_JOB_NAME:-isaaclab}"
pbs_mail_events="${CLUSTER_PBS_MAIL_EVENTS:-}"
pbs_mail_user="${CLUSTER_PBS_MAIL_USER:-}"

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#PBS -l select=${pbs_select}
#PBS -l walltime=${pbs_walltime}
#PBS -j oe
#PBS -q ${pbs_queue}
#PBS -N ${pbs_job_name}
$( [ -n "$pbs_mail_events" ] && [ -n "$pbs_mail_user" ] && echo "#PBS -m ${pbs_mail_events} -M \"${pbs_mail_user}\"" )

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

qsub job.sh
rm job.sh
