# SPDX-License-Identifier: Apache-2.0
"""Replay one dump N times: determinism of the residual mismatch rows."""
import sys

sys.path.insert(0, "csrc/attention/glm5_kpool_indexer")
import torch
import smoke_compare as S


def main():
    f, n = sys.argv[1], int(sys.argv[2])
    d = torch.load(f, map_location="cpu", weights_only=False)
    dev = "npu:0"
    inputs = dict(
        query=d["query"].to(dev), indexer_cache=d["indexer_cache"].to(dev),
        weights=d["weights"].to(dev), cum_query_lens=d["cum_query_lens"].to(dev),
        indexer_seq_lens=d["indexer_seq_lens"].to(dev),
        indexer_block_table=d["indexer_block_table"].to(dev),
        positions=d["positions"].to(dev), max_pool=d["max_pool_seq_len"])
    for i in range(n):
        ref, test = S.run_impl(inputs)
        ok = S.compare(ref, test, f"attempt{i}", inputs=inputs)
        print(f"attempt {i}: {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
