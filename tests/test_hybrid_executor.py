from __future__ import annotations

import math

import pytest
import torch
import torch.fx as fx
from transformers import GPT2Config, GPT2LMHeadModel

from src.core.decoder_export_backend import try_export_decoder_only_trace
from src.ilp.advanced_terms import ActivationStrategy
from src.runtime.device_plan import DevicePlan, plan_requests_gpu
from src.runtime.hybrid_executor import _DeviceAwareFXInterpreter, HybridExecutionUnsupportedError, run_hybrid_training
from validation import run_hybrid_execution as hybrid_cli


class ToyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Linear(4, 3)
        self.right = torch.nn.Linear(4, 3)
        self.out = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.left(x)
        r = self.right(x)
        return self.out(torch.relu(l + r))


class DictInputModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(16, 8)
        self.proj = torch.nn.Linear(8, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.emb(input_ids).mean(dim=1)
        return self.proj(hidden)


def _tiny_gpt2() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.config.pad_token_id = 0
    model.config.use_cache = False
    model.train()
    return model


def test_hybrid_executor_runs_cpu_plan() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=2,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.steps == 2
    assert len(result.per_step) == 2
    assert result.total_transfer_events == 0
    assert result.avg_power_w >= 0.0
    assert result.total_energy_j >= 0.0
    assert result.recompute_layers == []
    assert result.checkpoint_layers == []
    assert result.backward_relocation_layers == []
    assert result.prefetch_layers == []
    assert result.total_prefetch_events == 0
    assert result.total_prefetch_mb == 0.0
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert math.isfinite(result.min_loss)
    assert all(math.isfinite(step.loss_value) for step in result.per_step)
    assert all(step.recompute_count == 0 for step in result.per_step)
    assert all(step.checkpoint_count == 0 for step in result.per_step)
    assert all(step.backward_relocation_count == 0 for step in result.per_step)
    assert all(step.prefetch_count == 0 for step in result.per_step)
    assert all(step.prefetch_total_mb == 0.0 for step in result.per_step)


def test_hybrid_executor_reports_supervised_accuracy_when_target_provided() -> None:
    model = ToyMLP()
    inp = torch.randn(4, 4)
    target = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        target_data=target,
        plan=plan,
        steps=2,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.quality_metric_name == "accuracy"
    assert result.final_quality_metric is not None
    assert 0.0 <= result.final_quality_metric <= 1.0
    assert all(step.task_metric_value is not None for step in result.per_step)


def test_hybrid_executor_warns_when_gpu_unavailable() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.energy_source in {"nvml", "rapl"}
    if not torch.cuda.is_available():
        assert any("CUDA is unavailable" in w for w in result.warnings)


def test_plan_requests_gpu_detection() -> None:
    cpu_plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )
    gpu_plan = DevicePlan(
        assignment_forward={"net.0": "GPU", "net.1": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[("net.0", "net.0")],
    )

    assert plan_requests_gpu(cpu_plan) is False
    assert plan_requests_gpu(gpu_plan) is True


def test_hybrid_executor_supports_recompute_strategy() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        activation_strategies={"net.0": ActivationStrategy("net.0", recompute=True)},
        steps=2,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.recompute_layers == ["net.0"]
    assert all(step.recompute_count == 1 for step in result.per_step)


def test_hybrid_executor_supports_checkpoint_strategy() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        activation_strategies={"net.1": "checkpoint"},
        steps=1,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    assert result.checkpoint_layers == ["net.1"]
    assert result.unsupported_checkpoint_layers == []
    assert all(step.checkpoint_count == 1 for step in result.per_step)
    assert not any("runtime currently supports only recompute" in warning for warning in result.warnings)


def test_hybrid_executor_warns_when_backward_assignment_differs() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[("net.0", "net.0")],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
    )

    assert result.status == "ok"
    if torch.cuda.is_available():
        assert result.backward_relocation_layers == ["net.0"]
        assert all(step.backward_relocation_count == 1 for step in result.per_step)
    else:
        assert result.backward_relocation_layers == []
        assert any("cannot be materialized" in warning for warning in result.warnings)


def test_hybrid_executor_prefetch_policy_activation() -> None:
    model = ToyMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "GPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[("net.0", "net.1")],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
        enable_async_transfer=True,
        enable_prefetch=True,
    )

    assert result.status == "ok"
    if torch.cuda.is_available():
        assert result.total_prefetch_events >= 1
        assert result.total_prefetch_mb >= 0.0
        assert len(result.prefetch_layers) >= 1
        assert any(step.prefetch_count >= 1 for step in result.per_step)
    else:
        assert result.total_prefetch_events == 0
        assert result.prefetch_layers == []


def test_hybrid_executor_dag_mode_supports_branching_model() -> None:
    model = BranchMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"left": "CPU", "right": "CPU", "out": "CPU"},
        assignment_backward={"left": "CPU", "right": "CPU", "out": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=2,
        lr=0.01,
        strict_plan=True,
        execution_mode="dag",
    )

    assert result.status == "ok"
    assert result.steps == 2
    assert len(result.per_step) == 2


def test_hybrid_executor_dag_mode_applies_activation_strategies_when_supported() -> None:
    model = BranchMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"left": "CPU", "right": "CPU", "out": "CPU"},
        assignment_backward={"left": "CPU", "right": "CPU", "out": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        activation_strategies={"left": "recompute", "right": "checkpoint"},
        steps=1,
        lr=0.01,
        strict_plan=True,
        execution_mode="dag",
    )

    assert result.status == "ok"
    assert "left" in result.recompute_layers
    assert "right" in result.checkpoint_layers
    assert result.unsupported_checkpoint_layers == []
    assert any(step.recompute_count >= 1 for step in result.per_step)
    assert any(step.checkpoint_count >= 1 for step in result.per_step)


def test_hybrid_executor_auto_supports_prefetch_features_without_forcing_linear() -> None:
    model = BranchMLP()
    inp = torch.randn(3, 4)
    plan = DevicePlan(
        assignment_forward={"left": "CPU", "right": "CPU", "out": "CPU"},
        assignment_backward={"left": "CPU", "right": "CPU", "out": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
        enable_prefetch=True,
        execution_mode="auto",
    )

    assert result.status == "ok"
    assert not any("selected linear runtime" in warning for warning in result.warnings)


def test_hybrid_executor_auto_uses_hf_trace_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    model = BranchMLP()
    inp = torch.randn(3, 4)
    original_symbolic_trace = torch.fx.symbolic_trace
    plan = DevicePlan(
        assignment_forward={"left": "CPU", "right": "CPU", "out": "CPU"},
        assignment_backward={"left": "CPU", "right": "CPU", "out": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    monkeypatch.setattr("src.runtime.hybrid_executor.fx.symbolic_trace", lambda _model: (_ for _ in ()).throw(RuntimeError("fx failed")))
    monkeypatch.setattr("src.runtime.hybrid_executor._try_huggingface_symbolic_trace", lambda current_model: original_symbolic_trace(current_model))

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
        execution_mode="auto",
    )

    assert result.status == "ok"
    assert not any("fell back to linear runtime" in warning for warning in result.warnings)


def test_hybrid_executor_auto_rejects_structured_input_without_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DictInputModel()
    inp = {"input_ids": torch.randint(0, 16, (2, 5), dtype=torch.long)}
    target = torch.tensor([0, 1], dtype=torch.long)
    plan = DevicePlan(
        assignment_forward={"emb": "CPU", "proj": "CPU"},
        assignment_backward={"emb": "CPU", "proj": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    monkeypatch.setattr("src.runtime.hybrid_executor.fx.symbolic_trace", lambda _model: (_ for _ in ()).throw(RuntimeError("fx failed")))
    monkeypatch.setattr("src.runtime.hybrid_executor._try_huggingface_symbolic_trace", lambda _model: None)

    with pytest.raises(HybridExecutionUnsupportedError, match="structured inputs"):
        run_hybrid_training(
            model=model,
            input_data=inp,
            target_data=target,
            plan=plan,
            steps=1,
            lr=0.01,
            strict_plan=True,
            execution_mode="auto",
        )


def test_hybrid_executor_dag_mode_supports_structured_dict_input() -> None:
    model = DictInputModel()
    inp = {"input_ids": torch.randint(0, 16, (2, 5), dtype=torch.long)}
    target = torch.tensor([0, 1], dtype=torch.long)
    plan = DevicePlan(
        assignment_forward={"emb": "CPU", "proj": "CPU"},
        assignment_backward={"emb": "CPU", "proj": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        target_data=target,
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
        execution_mode="dag",
    )

    assert result.status == "ok"
    assert result.steps == 1
    assert len(result.per_step) == 1


def test_hybrid_executor_dag_mode_supports_decoder_only_export_backend() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8), dtype=torch.long)
    inp = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    export_trace_ctx = try_export_decoder_only_trace(model, inp)

    assert export_trace_ctx is not None

    plan_layers = {name: "CPU" for name in set(export_trace_ctx.node_layer_names.values())}
    plan = DevicePlan(
        assignment_forward=dict(plan_layers),
        assignment_backward=dict(plan_layers),
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        target_data=input_ids.clone(),
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
        execution_mode="dag",
    )

    assert result.status == "ok"
    assert result.steps == 1
    assert len(result.per_step) == 1
    assert result.quality_metric_name == "token_accuracy"
    assert result.final_quality_metric is not None


def test_hybrid_executor_export_backend_ignores_nonmaterial_unplanned_leaves() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8), dtype=torch.long)
    inp = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    export_trace_ctx = try_export_decoder_only_trace(model, inp)

    assert export_trace_ctx is not None

    plan_layers = {name: "CPU" for name in set(export_trace_ctx.node_layer_names.values())}
    plan = DevicePlan(
        assignment_forward=dict(plan_layers),
        assignment_backward=dict(plan_layers),
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    result = run_hybrid_training(
        model=model,
        input_data=inp,
        target_data=input_ids.clone(),
        plan=plan,
        steps=1,
        lr=0.01,
        strict_plan=True,
        execution_mode="dag",
    )

    assert result.status == "ok"
    assert not any("not explicitly assigned and default to CPU" in warning for warning in result.warnings)


def test_decoder_only_export_interpreter_skips_tensor_metadata_assertion() -> None:
    model = ToyMLP()
    traced = fx.symbolic_trace(model)
    plan = DevicePlan(
        assignment_forward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        assignment_backward={"net.0": "CPU", "net.1": "CPU", "net.2": "CPU"},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )
    interp = _DeviceAwareFXInterpreter(
        traced,
        plan,
        gpu_id=0,
        transfer_metrics={"bytes": 0.0, "events": 0.0},
        activation_strategies=None,
        warnings=[],
        runtime_features={
            "recompute_layers": set(),
            "checkpoint_layers": set(),
            "backward_relocation_layers": set(),
            "prefetch_layers": set(),
            "unsupported_activation_layers": set(),
            "recompute_count": 0,
            "checkpoint_count": 0,
            "backward_relocation_count": 0,
            "prefetch_count": 0,
            "prefetch_bytes": 0.0,
        },
        enable_async_transfer=False,
        enable_prefetch=False,
        transfer_stream=None,
        supports_activation_strategies=False,
    )

    result = interp.call_function(
        torch.ops.aten._assert_tensor_metadata.default,
        args=(torch.randn(2, 3),),
        kwargs={"dtype": torch.float32, "device": torch.device("cpu"), "layout": torch.strided},
    )

    assert result is None


def test_run_hybrid_execution_safe_wraps_unsupported_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    args = object()
    plan = DevicePlan(
        assignment_forward={},
        assignment_backward={},
        cut_edges_forward=[],
        cut_edges_backward=[],
        cross_phase_edges=[],
    )

    def _raise_unsupported(*_args: object, **_kwargs: object) -> None:
        raise hybrid_cli.HybridExecutionUnsupportedError("trace unsupported")

    monkeypatch.setattr(hybrid_cli, "_run_single", _raise_unsupported)

    payload = hybrid_cli._run_single_safe(args, plan, "ilp_plan")

    assert payload["result"].status == "unsupported"
    assert any("trace unsupported" in warning for warning in payload["result"].warnings)
