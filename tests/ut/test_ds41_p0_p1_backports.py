# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""回补的轻依赖验证：执行真实源码节点，仅替换设备和上游委托边界。"""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def source_node(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next((item for item in tree.body if getattr(item, "name", None) == name), None)
    assert node is not None, f"缺少回补能力：{name}"
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(ROOT / path), "exec", flags=0x1000000), namespace)
    return namespace[name]


@dataclass
class Group:
    layer_names: list[str]


def test_zeroer_excludes_nested_empty_and_static_scale_components():
    calls = []

    class Delegate:
        def __init__(self, device, **kwargs):
            context = kwargs["static_forward_context"]
            for value in context.values():
                assert isinstance(value.kv_cache, torch.Tensor)
                assert value.kv_cache.numel() > 0
            calls.append(context)

    cls = source_node(
        "vllm_ascend/worker/v2/utils.py",
        "AscendV2KVBlockZeroer",
        {"KVBlockZeroer": Delegate, "torch": torch, "replace": replace, "SimpleNamespace": SimpleNamespace},
    )
    tensor = torch.ones(2, 3)
    context = {
        "plain": SimpleNamespace(kv_cache=tensor),
        "nested": SimpleNamespace(kv_cache=([tensor], (), None, torch.empty(0), tensor)),
        "mxfp8": SimpleNamespace(kv_cache=(tensor, tensor, tensor, torch.ones(7))),
        "runner": SimpleNamespace(kv_cache=tensor),
    }
    cls(torch.device("cpu"), [Group(list(context))], [1], context, 2, {"runner"}, cache_dtype="mxfp8")
    assert len(calls) == 4
    assert all("runner" not in call for call in calls)
    assert sum("mxfp8" in call for call in calls) == 3
    assert sum("nested" in call for call in calls) == 1


@pytest.mark.parametrize("overlap", [False, True, None])
def test_moe_event_gate_preserves_enabled_and_uninitialized_behavior(overlap):
    stream = MagicMock()

    def config():
        if overlap is None:
            raise RuntimeError("配置未初始化")
        return SimpleNamespace(multistream_overlap_shared_expert=overlap)

    fn = source_node(
        "vllm_ascend/ops/fused_moe/moe_mlp.py",
        "maybe_record_event",
        {"torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream)), "get_ascend_config": config},
    )
    event = fn()
    if overlap is False:
        assert event is None
        stream.record_event.assert_not_called()
    else:
        assert event is stream.record_event.return_value
        stream.record_event.assert_called_once()


def test_otp_pool_reuses_storage_and_isolates_model_dtype_and_shape():
    cls = source_node("vllm_ascend/attention/dsa_v1.py", "OProjBufferPool", {"torch": torch})
    pool = cls()
    signature = ((2, 8, 1, 4), torch.float32, torch.device("cpu"))
    first = pool.get_exchange(*signature)
    assert all(a is b for a, b in zip(first, pool.get_exchange(*signature)))
    assert torch.count_nonzero(first[0]) == 0
    assert pool.get_exchange(signature[0], torch.float16, signature[2])[0] is not first[0]
    assert pool.get_exchange((2, 16, 1, 4), *signature[1:])[0] is not first[0]
    assert cls().get_exchange(*signature)[0] is not first[0]
    rs = pool.get_reduce_scatter((8, 4), *signature[1:])
    assert pool.get_reduce_scatter((8, 4), *signature[1:]) is rs
    assert rs.data_ptr() != first[0].data_ptr()


def test_v41_compile_declares_all_variable_token_dimensions():
    tree = ast.parse((ROOT / "vllm_ascend/models/deepseek_v41/model.py").read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model")
    decorators = [
        node
        for node in model.decorator_list
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "support_torch_compile"
    ]
    assert len(decorators) == 1
    mapping = ast.literal_eval(next(arg.value for arg in decorators[0].keywords if arg.arg == "dynamic_arg_dims"))
    assert mapping == {"input_ids": 0, "positions": 0, "intermediate_tensors": 0, "inputs_embeds": 0}


@pytest.mark.parametrize("computed,num_reqs", [(0, 1), (1, 1), (2, 1), (4, 1), (0, 0)])
def test_engram_masked_lanes_never_form_negative_history_addresses(computed, num_reqs):
    class Pointer:
        def __init__(self, values, offsets=0):
            self.values = values
            self.offsets = np.asarray(offsets)

        def __add__(self, offsets):
            return Pointer(self.values, self.offsets + offsets)

    class Language:
        constexpr = object()

        @staticmethod
        def program_id(axis):
            return 0

        @staticmethod
        def arange(start, end):
            return np.arange(start, end)

        maximum = staticmethod(np.maximum)

        @staticmethod
        def load(pointer, mask=True, other=0):
            # 昇腾 masked lane 也不能形成负 GM 地址；单独验证地址与可见性。
            assert np.all(pointer.offsets >= 0), "masked lane 形成了负历史地址"
            offsets, valid = np.broadcast_arrays(pointer.offsets, mask)
            result = np.full(offsets.shape, other, dtype=np.int64)
            result[valid] = pointer.values[offsets[valid]]
            return result

        @staticmethod
        def store(pointer, values, mask=True):
            pointer.values[pointer.offsets[mask]] = values[mask]

    result = np.full(3, 99, dtype=np.int64)
    fn = source_node("vllm_ascend/ops/triton/engram_lookback.py", "_gather_lookback_kernel", {"tl": Language})
    fn(
        Pointer(result),
        Pointer(np.array([0])),
        Pointer(np.array([computed])),
        Pointer(np.arange(10, 14)),
        4,
        num_reqs,
        3,
        4,
    )
    expected = [10 + computed - 1 - offset if num_reqs and computed > offset else -1 for offset in range(3)]
    np.testing.assert_array_equal(result, expected)
