#!/bin/bash
set -o pipefail
JOB="$1"; mkdir -p "$JOB"; cd /opt/src/vllm-ascend
VENV_SH=/opt/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
{
  echo "start=$(date -u +%FT%TZ)"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source $VENV_SH || echo "VENDORS set_env MISSING"
  export PYTHONPATH=/opt/src/vllm-ascend:${PYTHONPATH:-}
  # no ASCEND_LAUNCH_BLOCKING: wall-clock throughput bench
  timeout -s ABRT 3600 env python3 -u csrc/attention/glm5_kpool_indexer/serving_shape_bench.py
  echo "rc=$?"
  echo "end=$(date -u +%FT%TZ)"
} > "$JOB/bench.log" 2>&1
