# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.distributed.device_communicators.cuda_communicator as cuda_comm_mod

from vllm_ascend.worker.v2 import utils as v2_utils


def test_v2_utils_context_managers_switch_and_restore():
    fake_cuda = MagicMock()
    fake_npu = MagicMock()
    fake_npu.graph.return_value = nullcontext()
    original_comm = cuda_comm_mod.CudaCommunicator
    npu_cls = object()

    with (
        patch.object(v2_utils, "torch", SimpleNamespace(cuda=fake_cuda, npu=fake_npu)),
        patch.object(v2_utils, "breakable_cudagraph", MagicMock()),
        patch.object(v2_utils, "weak_ref_workspaces") as weak_ref,
        patch.object(v2_utils, "get_graph_params", return_value="graph"),
        patch.object(v2_utils, "get_draft_graph_params", return_value="draft"),
        patch.object(
            v2_utils,
            "get_ascend_config",
            return_value=SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_super_kernel=False)),
        ),
        patch.object(v2_utils.logger, "info_once", create=True),
        patch.object(v2_utils.logger, "debug"),
        patch(
            "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator",
            npu_cls,
        ),
    ):
        with v2_utils.torch_cuda_wrapper():
            assert fake_cuda.Event is fake_npu.Event
            assert fake_cuda.graph is v2_utils.torch_npu_graph_wrapper
            assert v2_utils.breakable_cudagraph.weak_ref_tensor is v2_utils.weak_ref_tensor

        with v2_utils.communicator_switch():
            assert cuda_comm_mod.CudaCommunicator is npu_cls
        assert cuda_comm_mod.CudaCommunicator is original_comm

        with v2_utils.torch_npu_graph_wrapper("capture"):
            pass
        weak_ref.assert_any_call("graph")
        weak_ref.assert_any_call("draft")
        assert weak_ref.call_count == 2


@pytest.mark.parametrize("invalid_only", [False, True])
def test_zeroer_filters_invalid_components_and_preserves_static_scale(monkeypatch, invalid_only):
    """执行真实适配类，只替换上游设备 zeroer；覆盖清零和预热委托。"""

    @dataclass
    class Group:
        layer_names: list[str]

    tensors = [torch.ones(3, 2) for _ in range(4)]
    static_scale = torch.full((2,), 7.0)
    context = {
        "invalid": SimpleNamespace(kv_cache=([tensors[0]], (), None, torch.empty(0))),
    }
    if not invalid_only:
        context["mixed"] = SimpleNamespace(kv_cache=([tensors[0]], torch.empty(0), tensors[1]))
        context["mxfp8"] = SimpleNamespace(kv_cache=(*tensors[1:], static_scale))
        context["runner"] = SimpleNamespace(kv_cache=tensors[0])
    delegates = []
    contexts = []

    def delegate(device, **kwargs):
        assert device == torch.device("cpu")
        contexts.append(kwargs["static_forward_context"])
        result = MagicMock()
        delegates.append(result)
        return result

    factory = MagicMock(side_effect=delegate)
    monkeypatch.setattr(v2_utils, "KVBlockZeroer", factory)
    zeroer = v2_utils.AscendV2KVBlockZeroer(
        torch.device("cpu"),
        [Group(list(context))],
        [2],
        context,
        3,
        {"runner"},
        cache_dtype="mxfp8",
    )
    zeroer.zero_block_ids([0, 2])
    zeroer.warmup(3)
    assert len(delegates) == (0 if invalid_only else 3)
    assert all("invalid" not in item and "runner" not in item for item in contexts)
    if not invalid_only:
        assert contexts[2]["mixed"].kv_cache is tensors[1]
        assert [item["mxfp8"].kv_cache for item in contexts] == tensors[1:]
    for item in contexts:
        assert all(isinstance(layer.kv_cache, torch.Tensor) and layer.kv_cache.numel() > 0 for layer in item.values())
    for item in delegates:
        item.zero_block_ids.assert_called_once_with([0, 2])
        item.warmup.assert_called_once_with(3)
    torch.testing.assert_close(static_scale, torch.full((2,), 7.0))
