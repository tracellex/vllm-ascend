#!/bin/bash
set -o pipefail
JOB="$1"; mkdir -p "$JOB"; cd /opt/src/vllm-ascend
VENV_SH=/opt/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
run() { echo "== $* =="; "$@"; echo "rc=$?"; }
{
  echo "start=$(date -u +%FT%TZ)"
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source $VENV_SH || echo "VENDORS set_env MISSING"
  export ASCEND_LAUNCH_BLOCKING=1
  export PYTHONPATH=/opt/src/vllm-ascend:${PYTHONPATH:-}
  run env python3 -u csrc/attention/glm5_kpool_indexer/probe_block_table.py
  run env python3 -u csrc/attention/glm5_kpool_indexer/smoke_compare.py --cases tiny single96 packed2 packed3 widescreen big contend2k contend4k contend-pack group-tail --impl ascendc_group_topk
  run env python3 -u csrc/attention/glm5_kpool_indexer/probe_qbar_precision.py
  unset ASCEND_LAUNCH_BLOCKING
  PIDS=$(cd /data1/ascend/myascend/run/stacks/glm53flash-029-pd/tmp/glmk-replay && ls -d pid* | tr "\n" " ")
  run env python3 -u csrc/attention/glm5_kpool_indexer/replay_parity.py --dump /data1/ascend/myascend/run/stacks/glm53flash-029-pd/tmp/glmk-replay --max-calls 414 --pids $PIDS
  run timeout -s ABRT 900 python3 -u -X faulthandler csrc/attention/glm5_kpool_indexer/acceptance_30r.py
  echo "end=$(date -u +%FT%TZ)"
} > "$JOB/test.log" 2>&1
