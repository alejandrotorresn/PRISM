import pytest
import torch

from src.core.decoder_export_backend import is_decoder_only_export_candidate, try_export_decoder_only_trace
from src.core.graph_extractor import export_graph_artifacts
from src.runtime.device_plan import DevicePlan
from src.runtime.hybrid_executor import run_hybrid_training

transformers = pytest.importorskip("transformers")


def _tiny_bert():
    cfg = transformers.BertConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        vocab_size=100, num_labels=3,
    )
    return transformers.BertForSequenceClassification(cfg)


def test_bert_is_an_export_candidate_for_dict_and_tensor_inputs():
    m = _tiny_bert()
    ids = torch.randint(0, 100, (2, 8))
    assert is_decoder_only_export_candidate(m, {"input_ids": ids})
    assert is_decoder_only_export_candidate(m, ids)
    assert not is_decoder_only_export_candidate(torch.nn.Linear(2, 2), ids)


def test_encoder_decoder_models_stay_excluded():
    cfg = transformers.T5Config(d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8, vocab_size=50)
    m = transformers.T5ForConditionalGeneration(cfg)
    assert not is_decoder_only_export_candidate(m, {"input_ids": torch.randint(0, 50, (2, 4))})


def test_bert_graph_is_extracted_with_export_backend(tmp_path):
    m = _tiny_bert().eval()
    ids = torch.randint(0, 100, (2, 8))
    for inp in ({"input_ids": ids}, ids):
        res = export_graph_artifacts(m, "bert", str(tmp_path), inp, include_records=True)
        assert res["trace_source"] == "torch_export_decoder_only"
        names = {n["node_name"] for n in res["nodes"]}
        assert any("attention" in n for n in names)
        assert res["edges_count"] > 0


def test_bert_trains_end_to_end_in_dag_mode():
    m = _tiny_bert()
    ids = torch.randint(0, 100, (2, 8))
    ctx = try_export_decoder_only_trace(m, {"input_ids": ids})
    assert ctx is not None
    layers = {n: "CPU" for n in set(ctx.node_layer_names.values())}
    plan = DevicePlan(dict(layers), dict(layers), [], [], [])
    res = run_hybrid_training(
        model=m, input_data={"input_ids": ids}, target_data=torch.randint(0, 3, (2,)),
        plan=plan, steps=1, strict_plan=False, execution_mode="dag",
    )
    assert res.status == "ok"
