# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-activation parity replay: feeds captured serving inputs through both
the Triton reference chain and the AscendC operator and compares rows exactly
like smoke_compare does.

Capture side: the sentinel-dir dump hook in vllm_ascend/ops/
glm5_kpool_indexer.py (see model-profile 014) writes one .pt per dispatcher
call with every input tensor plus the kwargs. Replay side (build container):

    python3 csrc/attention/glm5_kpool_indexer/replay_parity.py \
        --dump /data1/ascend/myascend/run/stacks/glm53flash-029-pd/tmp/glmk-replay \
        --max-calls 60
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import smoke_compare as S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="sentinel dump root")
    p.add_argument("--max-calls", type=int, default=60)
    p.add_argument("--impl", default="ascendc_group_topk",
                   choices=("ascendc_group_topk", "ascendc_group_topk_m64",
                            "ascendc_group_topk_split"))
    p.add_argument("--pids", nargs="*", default=None,
                   help="restrict to these pid dirs (default: one dir, the largest)")
    args = p.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.dump, "pid*")))
    if not dirs:
        print("FATAL: no pid* dirs under", args.dump)
        sys.exit(2)
    if args.pids:
        dirs = [d for d in dirs if os.path.basename(d) in args.pids]
    else:
        # default: the pid dir with the most captures (fullest core timeline)
        dirs = [max(dirs, key=lambda d: len(glob.glob(os.path.join(d, "call-*.pt"))))]

    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, "call-*.pt"))))
    files = files[: args.max_calls]

    ok = True
    prefill_rows = decode_rows = 0
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        dev = "npu:0"
        inputs = dict(
            query=d["query"].to(dev),
            indexer_cache=d["indexer_cache"].to(dev),
            weights=d["weights"].to(dev),
            cum_query_lens=d["cum_query_lens"].to(dev),
            indexer_seq_lens=d["indexer_seq_lens"].to(dev),
            indexer_block_table=d["indexer_block_table"].to(dev),
            positions=d["positions"].to(dev),
            max_pool=d["max_pool_seq_len"],
        )
        pools = int(d["indexer_seq_lens"].max())
        tag = f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)} T={d['query'].shape[0]} pools={pools}"
        ref, test = S.run_impl(inputs, impl=args.impl)
        ok &= S.compare(ref, test, tag, inputs=inputs)
        if d["query"].shape[0] > 1024:
            prefill_rows += ref.shape[0]
        else:
            decode_rows += ref.shape[0]
        if not ok:
            print("FAIL at", tag, flush=True)
            sys.exit(1)
    print(f"replayed {len(files)} calls: prefill_rows={prefill_rows} decode_rows={decode_rows}")
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import torch  # noqa: E402  (after path insert, mirrors smoke_compare import order)
    main()
