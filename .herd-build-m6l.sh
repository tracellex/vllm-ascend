#!/bin/bash
set -uo pipefail
JOB="$1"; mkdir -p "$JOB"; cd /opt/src/vllm-ascend
OP=csrc/attention/glm5_kpool_indexer
BIN=csrc/build/binary/ascend910_93/bin/glm5_kpool_indexer
INST=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/opp/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/glm5_kpool_indexer
{
  echo "start=$(date -u +%FT%TZ)"
  echo "== before =="
  sha256sum $BIN/*.o $INST/*.o 2>/dev/null || true
  rm -rf csrc/build
  echo "csrc/build removed rc=$?"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export SOC_VERSION=ascend910_9391
  bash csrc/build_aclnn.sh /opt/src/vllm-ascend ascend910_9391
  echo "build_aclnn rc=$?"
  echo "== after build =="
  sha256sum $BIN/*.o 2>/dev/null || echo "BIN MISSING"
  echo "== after install =="
  sha256sum $INST/*.o 2>/dev/null || echo "INST MISSING"
  echo "end=$(date -u +%FT%TZ)"
} > "$JOB/build.log" 2>&1
