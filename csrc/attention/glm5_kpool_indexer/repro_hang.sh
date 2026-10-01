#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Hang reproduction package for the CANN 9.1.0 task-dispatch defect.
#
# Environment: Ascend910 (910_9391 / A3), driver 26.0.rc2, CANN 9.1.0
# toolkit, image quay.io/ascend/vllm-ascend:glm53-a3-41run. Reproduced on
# two hosts (192.168.32.39 / .40) and multiple cards with identical rates.
#
# Setup: extract glmk-run-pkg.tgz (vllm_ascend python package with the
# vendored custom ops + the smoke script) and run inside the container.
#
#   for i in $(seq 1 8); do
#     s=$(date +%s)
#     timeout 90 python3 csrc/attention/glm5_kpool_indexer/smoke_compare.py \
#         --cases single96 > /tmp/hang-$i.log 2>&1
#     rc=$?; e=$(date +%s)
#     echo "run$i rc=$rc t=$((e-s))s"
#   done
#
# Expected: ~2/8 runs return rc=124 (timeout kill). The hung process sits in
# torch.npu.synchronize() (faulthandler proof); no runtime error is printed
# (device-level sync has no timeout). Occasionally the stream-level form
# appears instead: AclrtSynchronizeStreamWithTimeout error 507015/507014
# "aicore execution times out" after ~100s.
#
# Facts established by bisection (all at the same ~25-37% hang rate, 8 runs
# each; see the git history of this worktree for details):
#   - empty-kernel build (AIC skips matmul, AIV skips fold/emit) hangs the
#     same -> not kernel workload
#   - inter-core signalling replaced by deterministic GM progress counters
#     -> still hangs -> not the FIA flag handshake
#   - MIX_AIC_1_1 (1:1 core ratio, blockDim formula adjusted) -> hangs
#   - second card and a second host -> same rate -> not hardware/host-local
#   - ASCEND_LAUNCH_BLOCKING=1 -> hangs (2/8)
#   - any msprof invocation lowers the rate dramatically on small shapes
#     (18/18 clean) but a large shape hung on the 5th consecutive launch
#   - triton reference path over identical inputs: 16/16 clean
#   - all completed parity runs are exact (rows_mismatch=0)
# Kernel-side duration for the passing runs: ~362us (T=1024, 8192 pools).
set -euo pipefail
for i in $(seq 1 8); do
  s=$(date +%s)
  timeout 90 python3 csrc/attention/glm5_kpool_indexer/smoke_compare.py \
      --cases single96 > /tmp/hang-$i.log 2>&1 || true
  rc=$?
  e=$(date +%s)
  echo "run$i rc=$rc t=$((e-s))s $(grep -oE 'rows_mismatch=[0-9]+' /tmp/hang-$i.log | head -1)"
done
