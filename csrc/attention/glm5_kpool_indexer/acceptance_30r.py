# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process acceptance: 6 cases x 30 rounds in ONE python process.

Reproduces the intended glmk-m-event-full6-30 trial structure (repeated
invocations sharing one process) with unbuffered output and a timeout budget
that fits the workload (~5-8s per round). Run from the worktree root inside
the build container:

    timeout -s ABRT 900 python3 -u -X faulthandler \
        csrc/attention/glm5_kpool_indexer/acceptance_30r.py
"""

import sys
import argparse

sys.path.insert(0, "csrc/attention/glm5_kpool_indexer")
import smoke_compare as S

names = ["tiny", "single96", "packed2", "packed3", "widescreen", "big",
         "contend2k", "contend4k", "contend-pack", "group-tail"]
parser = argparse.ArgumentParser()
parser.add_argument("--impl", choices=("ascendc_group_topk", "ascendc_group_topk_m64",
                                       "ascendc_group_topk_split"),
                    default="ascendc_group_topk")
impl = parser.parse_args().impl
ok = True
for r in range(1, 31):
    for n in names:
        inputs = S.CASES[n]()
        ref, test = S.run_impl(inputs, impl=impl)
        ok &= S.compare(ref, test, n, inputs=inputs)
        if not ok:
            print("FAIL at", n, "round", r, flush=True)
            sys.exit(1)
    print("round", r, "done", flush=True)
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
