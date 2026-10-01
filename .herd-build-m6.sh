#!/bin/bash
# M6 build driver: rebuild the operator after the SortAll hierarchical-merge fix.
set -uo pipefail
JOB="$1"
mkdir -p "$JOB"
cd /opt/src/vllm-ascend
{
  echo "start=$(date -u +%FT%TZ)"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export SOC_VERSION=ascend910_9391
  bash csrc/build_aclnn.sh /opt/src/vllm-ascend ascend910_9391
  rc1=$?
  echo "build_aclnn rc=$rc1"
  if [ $rc1 -eq 0 ]; then
    python3 setup.py build_ext --inplace
    echo "build_ext rc=$?"
  fi
  echo "end=$(date -u +%FT%TZ)"
} > "$JOB/build.log" 2>&1
