#!/usr/bin/env bash

# Load cluster config from synced workspace if available.
if [ -f "$1/docker/cluster/.env.cluster" ]; then
	# shellcheck disable=SC1090
	source "$1/docker/cluster/.env.cluster"
fi

slurm_gpus_per_node="${CLUSTER_SLURM_GPUS_PER_NODE:-rtx_6000:1}"
slurm_nodes="${CLUSTER_SLURM_NODES:-1}"
slurm_cpus_per_task="${CLUSTER_SLURM_CPUS_PER_TASK:-8}"
slurm_mem_per_gpu="${CLUSTER_SLURM_MEM_PER_GPU:-24G}"
slurm_time="${CLUSTER_SLURM_TIME:-8:00:00}"
slurm_partition="${CLUSTER_SLURM_PARTITION:-}"
slurm_account="${CLUSTER_SLURM_ACCOUNT:-}"
slurm_mail_type="${CLUSTER_SLURM_MAIL_TYPE:-}"
slurm_mail_user="${CLUSTER_SLURM_MAIL_USER:-}"
slurm_email_logs_on="${CLUSTER_SLURM_EMAIL_LOGS_ON:-}"
slurm_email_log_tail_lines="${CLUSTER_SLURM_EMAIL_LOG_TAIL_LINES:-200}"

cat <<EOT > job.sh
#!/bin/bash

#SBATCH --gpus-per-node=${slurm_gpus_per_node}
#SBATCH -N${slurm_nodes}
#SBATCH --cpus-per-task=${slurm_cpus_per_task}
#SBATCH --mem-per-gpu=${slurm_mem_per_gpu}
#SBATCH --time=${slurm_time}
#SBATCH --job-name="training-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --output="output_%j.log"
#SBATCH --error="error_%j.log"
$( [ -n "$slurm_partition" ] && echo "#SBATCH --partition=${slurm_partition}" )
$( [ -n "$slurm_account" ] && echo "#SBATCH --account=${slurm_account}" )
$( [ -n "$slurm_mail_type" ] && [ -n "$slurm_mail_user" ] && echo "#SBATCH --mail-type=${slurm_mail_type}" )
$( [ -n "$slurm_mail_type" ] && [ -n "$slurm_mail_user" ] && echo "#SBATCH --mail-user=${slurm_mail_user}" )

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
job_exit_code=0
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}" || job_exit_code=\$?

email_logs_on="${slurm_email_logs_on}"
email_log_tail_lines="${slurm_email_log_tail_lines}"
mail_user="${slurm_mail_user}"

if [ -n "\$mail_user" ] && [ -n "\$email_logs_on" ]; then
	should_email_logs=0
	case "\$email_logs_on" in
		ALWAYS|always)
			should_email_logs=1
			;;
		FAIL|fail)
			if [ "\$job_exit_code" -ne 0 ]; then
				should_email_logs=1
			fi
			;;
		END|end)
			if [ "\$job_exit_code" -eq 0 ]; then
				should_email_logs=1
			fi
			;;
		*)
			should_email_logs=0
			;;
	esac

	if [ "\$should_email_logs" -eq 1 ] && command -v mail >/dev/null 2>&1; then
		err_log="\${SLURM_SUBMIT_DIR:-\$PWD}/error_\${SLURM_JOB_ID:-unknown}.log"
		status_label="SUCCESS"
		if [ "\$job_exit_code" -ne 0 ]; then
			status_label="FAIL"
		fi

		tmp_mail_body="\$(mktemp)"
		{
			echo "SLURM job summary"
			echo "job_id=\${SLURM_JOB_ID:-unknown}"
			echo "job_name=\${SLURM_JOB_NAME:-unknown}"
			echo "state=\${status_label}"
			echo "exit_code=\${job_exit_code}"
			echo "host=\$(hostname)"
			echo
			echo "===== tail -n \${email_log_tail_lines} \${err_log} ====="
			if [ -f "\$err_log" ]; then
				tail -n "\$email_log_tail_lines" "\$err_log"
			else
				echo "(missing)"
			fi
		} > "\$tmp_mail_body"

		mail -s "[SLURM][\${SLURM_JOB_ID:-unknown}] \${status_label} error log tail" "\$mail_user" < "\$tmp_mail_body" || true
		rm -f "\$tmp_mail_body"
	fi
fi

exit "\$job_exit_code"
EOT
sbatch < job.sh
rm job.sh