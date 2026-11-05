#!/usr/bin/env bash
# OpenMPI launches one wrapper per rank. Redirect locally to avoid relying on
# OpenMPI 5's --output-filename forwarding (seen to lose output on the laptop).
set -u
: "${ROSIE_RANK_LOG_DIR:?missing rank log directory}"
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${SLURM_PROCID:-}}}}
[[ $rank =~ ^[0-9]+$ ]] || { echo "cannot determine MPI rank" >&2; exit 95; }
exec > "$ROSIE_RANK_LOG_DIR/rank.$rank.log" 2>&1
printf 'rank=%s host=%s pid=%s gpuMap=%s comm=%s\n' "$rank" "$(hostname)" "$$" \
    "${BRAE_GPU_MAP:-rank}" "${BRAE_COMM:-unset}"
exec "$@"
