#!/bin/bash
set -uo pipefail
JOB="$1"; mkdir -p "$JOB"; cd /opt/src/vllm-ascend
{
  echo "start=$(date -u +%FT%TZ)"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export SOC_VERSION=ascend910_9391
  bash csrc/build_aclnn.sh /opt/src/vllm-ascend ascend910_9391
  echo "build_aclnn rc=$?"
  echo "end=$(date -u +%FT%TZ)"
} > "$JOB/build.log" 2>&1
