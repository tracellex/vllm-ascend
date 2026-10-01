#!/bin/bash
set -uo pipefail
JOB="$1"
mkdir -p "$JOB"
cd /opt/src/vllm-ascend
{
  echo "start=$(date -u +%FT%TZ)"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export SOC_VERSION=ascend910_9391
  rm -rf build/temp.linux-aarch64-cpython-312
  python3 setup.py build_ext --inplace
  echo "build_ext rc=$?"
  echo "end=$(date -u +%FT%TZ)"
} > "$JOB/build.log" 2>&1
