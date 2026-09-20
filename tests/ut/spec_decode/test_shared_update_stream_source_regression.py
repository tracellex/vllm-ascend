# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for MRV1 main/draft ACL graph stream sharing."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return item
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def test_mrv1_main_and_draft_full_graph_share_update_stream() -> None:
    proposer_graph_setup = ast.unparse(
        _method(
            ROOT / "vllm_ascend/spec_decode/llm_base_proposer.py",
            "AscendSpecDecodeBaseProposer",
            "_maybe_share_lm_head",
        )
    )
    runner_load = ast.unparse(
        _method(
            ROOT / "vllm_ascend/worker/model_runner_v1.py",
            "NPUModelRunner",
            "load_model",
        )
    )

    assert "self.update_stream = None" in proposer_graph_setup
    assert "self.update_stream: torch.npu.Stream = torch.npu.Stream()" in runner_load
    assert "self.drafter.update_stream = self.update_stream" in runner_load


def test_dummy_slots_are_invalidated_before_attention_metadata_build() -> None:
    dummy_run = ast.unparse(
        _method(
            ROOT / "vllm_ascend/worker/model_runner_v1.py",
            "NPUModelRunner",
            "_dummy_run",
        )
    )

    invalidate = "blk_table.slot_mapping.gpu.fill_(-1)"
    build_metadata = "self._build_attention_metadata("
    assert invalidate in dummy_run
    assert dummy_run.index(invalidate) < dummy_run.index(build_metadata)
