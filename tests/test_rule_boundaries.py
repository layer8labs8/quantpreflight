"""Both-sides boundary tests: each rule must fire on the bad input and stay quiet on the good one."""

from __future__ import annotations

from typing import Any

import pytest

from quantpreflight.formatters import _fmt_bytes, format_json_report, format_sarif_report, format_text_report
from quantpreflight.models import CheckpointMeta, QuantSpec, Severity, TargetSpec, TensorSpec, TokenizerMeta
from quantpreflight.profiles import load_advisories
from quantpreflight.rules import analyze_checkpoint, context_limit, estimate_per_gpu_vram_gb

LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
AWQ4 = QuantSpec(method="awq", bits=4.0)


def _tensor(
    name: str,
    shape: tuple[int, ...] = (8, 8),
    dtype: str = "F16",
    bits: float = 16.0,
    *,
    logical: tuple[int, ...] | None = None,
    offsets: tuple[int, int] | None = None,
) -> TensorSpec:
    numel = 1
    for d in shape:
        numel *= d
    return TensorSpec(
        name=name,
        storage_dtype=dtype,
        shape=shape,
        shard="model.safetensors",
        byte_size=int(numel * bits / 8),
        effective_bits=bits,
        logical_shape=logical,
        data_offsets=offsets,
    )


def _meta(
    tensors: tuple[TensorSpec, ...] = (),
    *,
    config: dict[str, Any] | None = None,
    quant: QuantSpec | None = None,
    container: str = "safetensors",
    **kwargs: Any,
) -> CheckpointMeta:
    return CheckpointMeta(
        source="mock",
        container_format=container,
        config=LLAMA if config is None else config,
        tensors={t.name: t for t in tensors},
        quant=quant or QuantSpec(),
        **kwargs,
    )


def _titles(meta: CheckpointMeta, rule: str, **target: Any) -> list[str]:
    report = analyze_checkpoint(meta, TargetSpec(**target))
    return [f.title for f in report.findings if f.rule_id == rule]


# QPF001


@pytest.mark.parametrize(("bits", "flagged"), [(4.0, False), (3.9, True)])
def test_qpf001_threshold_is_exclusive(bits: float, flagged: bool) -> None:
    meta = _meta((_tensor("lm_head.weight", bits=bits),), quant=AWQ4)
    assert bool(_titles(meta, "QPF001")) is flagged


def test_qpf001_strictest_matching_rule_wins() -> None:
    meta = _meta((_tensor("vision_tower.blocks.0.norm1.weight", bits=8.0),), quant=QuantSpec(method="awq", bits=8.0))
    assert _titles(meta, "QPF001") == ["Bit-starved sensitive layer (llama_norms)"]


# QPF002


@pytest.mark.parametrize(("token_id", "flagged"), [(99, False), (100, True)])
def test_qpf002_token_id_boundary(token_id: int, flagged: bool) -> None:
    meta = _meta(
        (_tensor("model.embed_tokens.weight", (100, 8)), _tensor("lm_head.weight", (500, 8))),
        tokenizer_meta=TokenizerMeta(max_token_id=token_id, max_token_str="<x>"),
    )
    assert bool(_titles(meta, "QPF002")) is flagged


@pytest.mark.parametrize(("vocab", "flagged"), [(100, False), (101, True)])
def test_qpf002_config_vocab_boundary(vocab: int, flagged: bool) -> None:
    meta = _meta((_tensor("model.embed_tokens.weight", (100, 8)),), config={**LLAMA, "vocab_size": vocab})
    assert bool(_titles(meta, "QPF002")) is flagged


def test_qpf002_uses_first_embedding_and_head_matched_by_suffix() -> None:
    meta = _meta(
        (
            _tensor("other.weight", (5, 8)),
            _tensor("text_model.model.embed_tokens.weight", (200, 8)),
            _tensor("vision.embed_tokens.weight", (5, 8)),
            _tensor("text_model.lm_head.weight", (100, 8)),
            _tensor("aux.lm_head.weight", (500, 8)),
        ),
        tokenizer_meta=TokenizerMeta(max_token_id=150, max_token_str="<x>"),
    )
    assert _titles(meta, "QPF002") == ["Tokenizer ID exceeds lm_head output vocabulary rows"]


# QPF003


@pytest.mark.parametrize(
    ("cfg", "gen", "tok", "title"),
    [
        (2, None, (2,), None),
        (2, None, (3,), "Chat stop token missing from generation eos_token_id"),
        (None, [2, 3], (3,), None),
        (None, None, (3,), None),
        (2, [5], (), "Disjoint eos_token_id sets between config.json and generation_config.json"),
        (2, [2, 5], (), None),
        (None, [5], (), None),
        (2, None, (), None),
    ],
)
def test_qpf003_stop_token_matrix(cfg: Any, gen: Any, tok: tuple[int, ...], title: str | None) -> None:
    config = {**LLAMA, **({} if cfg is None else {"eos_token_id": cfg})}
    meta = _meta(
        config=config,
        generation_config={} if gen is None else {"eos_token_id": gen},
        tokenizer_meta=TokenizerMeta(eos_token="<e>", eos_token_ids=tok),
    )
    assert _titles(meta, "QPF003") == ([] if title is None else [title])


# QPF004

_BF16 = (_tensor("a.weight", dtype="BF16"),)
_F16 = (_tensor("a.weight"),)
_FP8 = QuantSpec(method="fp8", bits=8.0)
_FP8_ACT = QuantSpec(method="fp8", bits=8.0, activation_dtype="fp8")
_FP8_BLOCK = QuantSpec(method="fp8", bits=8.0, weight_block_size=(128, 128))
_AWQ_BLOCK = QuantSpec(method="awq", bits=4.0, weight_block_size=(128, 128))
_MARLIN = QuantSpec(method="gptq", bits=4.0, format_version="marlin")
_BF16_CFG = {**LLAMA, "torch_dtype": "bfloat16"}


@pytest.mark.parametrize(
    ("tensors", "quant", "config", "gpu", "prefix"),
    [
        (_BF16, None, None, "T4", "BF16 runs as FP16"),
        (_BF16, None, None, "A100", None),
        (_F16, None, None, "T4", None),
        (_F16, AWQ4, _BF16_CFG, "T4", "BF16 runs as FP16"),
        (_F16, AWQ4, None, "T4", None),
        (_BF16, AWQ4, None, "T4", None),
        (_F16, QuantSpec(method="nvfp4", bits=4.0), None, "H100", "NVFP4 requires"),
        (_F16, QuantSpec(method="nvfp4", bits=4.0), None, "B200", None),
        (_F16, AWQ4, None, "H100", None),
        (_F16, _FP8_BLOCK, None, "L4", "Block-scaled FP8"),
        (_F16, _FP8, None, "L4", None),
        (_F16, _FP8_BLOCK, None, "H100", None),
        (_F16, _AWQ_BLOCK, None, "L4", None),
        (_F16, _FP8, None, "H100", None),
        (_F16, _FP8, None, "A100", None),
        (_F16, _FP8, None, "T4", "No FP8 cores"),
        (_F16, _FP8_ACT, None, "A100", "No FP8 cores"),
        (_F16, _FP8_BLOCK, None, "A100", "No FP8 cores"),
        (_F16, _MARLIN, None, "MI300X", "Pre-packed Marlin"),
        (_F16, _MARLIN, None, "A100", None),
        (_F16, QuantSpec(method="gptq", bits=4.0), None, "MI300X", None),
    ],
)
def test_qpf004_gpu_matrix(
    tensors: tuple[TensorSpec, ...],
    quant: QuantSpec | None,
    config: dict[str, Any] | None,
    gpu: str,
    prefix: str | None,
) -> None:
    titles = _titles(_meta(tensors, config=config, quant=quant), "QPF004", gpu=gpu)
    if prefix is None:
        assert titles == []
    else:
        assert len(titles) == 1 and titles[0].startswith(prefix)


# QPF005


@pytest.mark.parametrize(
    ("engine", "method", "flagged"),
    [("vllm", "autoround", True), ("vllm", "compressed-tensors", True), ("vllm", "awq", False), ("sglang", "autoround", False)],
)
def test_qpf005_int8_marlin_flag(engine: str, method: str, flagged: bool) -> None:
    meta = _meta(quant=QuantSpec(method=method, bits=8.0))
    assert bool(_titles(meta, "QPF005", engine=engine, marlin_input_dtype="int8")) is flagged


# QPF006

_NOT_HEADS = "num_attention_heads not divisible by TP degree"
_NOT_KV = "GQA num_key_value_heads not divisible by TP degree"
_NOT_MULT = "TP degree is not a multiple of num_key_value_heads"
_REPLICA = "KV cache heads will be replicated across TP ranks"
_NOT_EXP = "MoE expert count not divisible by TP degree"


@pytest.mark.parametrize(
    ("extra", "tp", "expected"),
    [
        ({"num_attention_heads": 8}, 2, []),
        ({"num_attention_heads": 9}, 2, [_NOT_KV, _NOT_HEADS]),
        ({"num_attention_heads": 8, "num_key_value_heads": 4}, 4, []),
        ({"num_attention_heads": 12, "num_key_value_heads": 6}, 4, [_NOT_KV]),
        ({"num_attention_heads": 8, "num_key_value_heads": 2}, 4, [_REPLICA]),
        ({"num_attention_heads": 6, "num_key_value_heads": 2}, 3, [_NOT_MULT]),
        ({"num_local_experts": 8}, 2, []),
        ({"num_local_experts": 9}, 2, [_NOT_EXP]),
        ({"n_routed_experts": 1}, 2, []),
    ],
)
def test_qpf006_tp_matrix(extra: dict[str, int], tp: int, expected: list[str]) -> None:
    assert _titles(_meta(config={**LLAMA, **extra}), "QPF006", tp=tp) == expected


# QPF007

_COL = "model.layers.0.self_attn.q_proj.qweight"
_ROW = "model.layers.0.self_attn.o_proj.qweight"
_TP_ERR = "Quantized projection dimension not divisible by TP degree"
_GROUP_ERR = "Row-parallel sharded input dimension violates quantization group_size"
_MARLIN_ERR = "Sharded quantized dimensions violate Marlin kernel tile alignment"
_BLOCK_ERR = "Sharded FP8 tensor dimensions violate weight_block_size"
_G128 = QuantSpec(method="gptq", bits=4.0, group_size=128)
_G100 = QuantSpec(method="gptq", bits=4.0, group_size=100)
_G100_DESC = QuantSpec(method="gptq", bits=4.0, group_size=100, desc_act=True)
_BLOCK = QuantSpec(method="fp8", bits=8.0, weight_block_size=(128, 128))


@pytest.mark.parametrize(
    ("name", "shape", "quant", "bits", "tp", "expected"),
    [
        (_COL, (128, 256), AWQ4, 4.0, 2, []),
        (_COL, (129, 256), AWQ4, 4.0, 2, [_TP_ERR]),
        (_ROW, (128, 257), AWQ4, 4.0, 2, [_TP_ERR]),
        (_ROW, (64, 256), _G128, 4.0, 1, []),
        (_ROW, (64, 256), _G100, 4.0, 1, [_GROUP_ERR]),
        (_ROW, (64, 256), _G100_DESC, 4.0, 1, []),
        (_COL, (64, 256), _G100, 4.0, 1, []),
        (_ROW, (64, 256), _G100, 16.0, 1, []),
        (_ROW, (64, 256), _G100, 8.0, 1, [_GROUP_ERR]),
        (_ROW, (64, 256), _G100, 8.5, 1, []),
        (_ROW, (256,), _G100, 4.0, 1, []),
        (_COL, (64, 100), AWQ4, 4.0, 1, [_MARLIN_ERR]),
        (_COL, (100, 128), AWQ4, 4.0, 1, [_MARLIN_ERR]),
        (_COL, (128, 128), _BLOCK, 8.0, 1, []),
        (_COL, (130, 128), _BLOCK, 8.0, 1, [_BLOCK_ERR]),
        (_COL, (128, 130), _BLOCK, 8.0, 1, [_BLOCK_ERR]),
    ],
)
def test_qpf007_alignment_matrix(
    name: str, shape: tuple[int, ...], quant: QuantSpec, bits: float, tp: int, expected: list[str]
) -> None:
    meta = _meta((_tensor(name, shape, bits=bits),), quant=quant)
    assert _titles(meta, "QPF007", gpu="H100", tp=tp) == expected


# QPF008


@pytest.mark.parametrize(
    ("engine", "method", "desc_act", "tp", "prefix"),
    [
        ("vllm", "gptq", True, 2, "vLLM rejects"),
        ("vllm", "gptq", True, 1, "vLLM rejects"),
        ("sglang", "gptq", True, 2, "GPTQ desc_act=true incurs"),
        ("sglang", "gptq", True, 1, None),
        ("vllm", "gptq", False, 2, None),
        ("vllm", "awq", True, 2, None),
    ],
)
def test_qpf008_desc_act_matrix(engine: str, method: str, desc_act: bool, tp: int, prefix: str | None) -> None:
    meta = _meta(quant=QuantSpec(method=method, bits=4.0, desc_act=desc_act))
    titles = _titles(meta, "QPF008", tp=tp, engine=engine)
    assert titles[0].startswith(prefix) if prefix else titles == []


# QPF009


@pytest.mark.parametrize(
    ("name", "shape", "hidden", "flagged"),
    [
        ("mtp.pre_projection.weight", (64, 10), 64, False),
        ("mtp.pre_projection.weight", (128, 10), 64, False),
        ("mtp.pre_projection.weight", (10, 10), 64, True),
        ("eagle.pre_projection.weight", (10, 10), 64, True),
        ("mtp.other.weight", (10, 10), 64, False),
        ("model.pre_projection.weight", (10, 10), 64, False),
        ("mtp.pre_projection.weight", (10,), 64, False),
        ("mtp.pre_projection.weight", (10, 10), 0, False),
    ],
)
def test_qpf009_mtp_projection_matrix(name: str, shape: tuple[int, ...], hidden: int, flagged: bool) -> None:
    meta = _meta((_tensor(name, shape),), config={**LLAMA, "hidden_size": hidden})
    assert bool(_titles(meta, "QPF009")) is flagged


# QPF010

_NO_QCFG = "Quantized tensors present without quantization_config in config.json"
_STALE_QCFG = "config.json declares quantization, but all tensors are >=16-bit"


@pytest.mark.parametrize(
    ("tensors", "quant", "kwargs", "expected"),
    [
        ((_tensor("a.qweight", bits=4.0),), None, {}, [_NO_QCFG]),
        ((_tensor("a.weight", dtype="F8_E4M3", bits=8.0),), None, {}, [_NO_QCFG]),
        ((_tensor("a.weight"),), None, {}, []),
        ((_tensor("a.qweight", bits=4.0),), None, {"config": {}}, []),
        ((_tensor("a.qweight", bits=4.0),), AWQ4, {}, []),
        ((_tensor("a.qweight", bits=4.0),), None, {"container": "gguf"}, []),
        ((_tensor("a.weight"),), AWQ4, {}, [_STALE_QCFG]),
        ((_tensor("a.weight"),), QuantSpec(method="bitsandbytes", bits=4.0), {}, []),
        ((_tensor("a.weight"),), AWQ4, {"missing_shards": ("m.safetensors",)}, []),
        ((), AWQ4, {}, []),
    ],
)
def test_qpf010_contradiction_matrix(
    tensors: tuple[TensorSpec, ...], quant: QuantSpec | None, kwargs: dict[str, Any], expected: list[str]
) -> None:
    assert _titles(_meta(tensors, quant=quant, **kwargs), "QPF010", engine="vllm") == expected


# QPF011


def test_qpf011_remediation_matches_container() -> None:
    gguf = analyze_checkpoint(_meta(container="gguf"), TargetSpec(engine="trtllm"))
    assert "--engine llamacpp" in next(f for f in gguf.findings if f.rule_id == "QPF011").remediation
    st = analyze_checkpoint(_meta(), TargetSpec(engine="llamacpp"))
    assert "convert_hf_to_gguf" in next(f for f in st.findings if f.rule_id == "QPF011").remediation


@pytest.mark.parametrize(
    ("container", "engine", "flagged"),
    [("gguf", "vllm", True), ("gguf", "sglang", True), ("safetensors", "vllm", False)],
)
def test_qpf011_iquant_only_on_gguf_with_vllm_family(container: str, engine: str, flagged: bool) -> None:
    quant = QuantSpec(method="gguf", bits=2.0)
    meta = _meta((_tensor("a.weight", dtype="IQ2_XXS", bits=2.0),), quant=quant, container=container)
    titles = [t for t in _titles(meta, "QPF011", engine=engine) if t.startswith("Sub-3-bit")]
    assert bool(titles) is flagged


# QPF012


_GIB_META = {"total_weight_bytes": 1024**3}


@pytest.mark.parametrize(("vram", "flagged"), [(1.83, False), (1.82, True)])
def test_qpf012_budget_boundary_is_exclusive(vram: float, flagged: bool) -> None:
    meta = _meta(config={}, **_GIB_META)
    target = {"vram_gb_override": vram, "gpu_memory_utilization": 1.0}
    assert bool(_titles(meta, "QPF012", **target)) is flagged


def test_qpf012_skips_checkpoints_without_weights() -> None:
    report = analyze_checkpoint(_meta(config={}), TargetSpec(vram_gb_override=1.0))
    assert report.estimated_vram_per_gpu_gb is None


_KV_BASE = {"num_hidden_layers": 8, "num_attention_heads": 8, "hidden_size": 800, "max_position_embeddings": 4096}


def _kv_gib(**extra: Any) -> float:
    meta = _meta(config={}, total_weight_bytes=0)
    target = TargetSpec(context_len=1_000_000)
    return estimate_per_gpu_vram_gb(meta, {**_KV_BASE, **extra}, target)[2]


def test_kv_estimate_attention_layer_accounting() -> None:
    base = _kv_gib()
    assert base > 0
    assert _kv_gib(attn_layer_period=2) == pytest.approx(base / 2, abs=0.02)
    assert _kv_gib(model_type="jamba") == pytest.approx(base / 8, abs=0.02)
    assert _kv_gib(layers_block_type=[], attn_layer_period=2) == pytest.approx(base / 2, abs=0.02)
    assert _kv_gib(layers_block_type=["attention", "mamba"] * 4) == pytest.approx(base / 2, abs=0.02)
    assert _kv_gib(layers_block_type=["mamba"] * 8) == pytest.approx(base, abs=0.02)


def test_kv_estimate_degenerate_config_values() -> None:
    assert _kv_gib(num_attention_heads=0) == 0.0
    assert _kv_gib(kv_lora_rank=0) == _kv_gib()
    assert _kv_gib(kv_lora_rank=512, num_hidden_layers=0) == 0.0


# QPF013


@pytest.mark.parametrize(
    ("offsets", "expected_prefix"),
    [
        ((0, 128), None),
        ((100, 50), "Inverted or negative"),
        ((-8, 120), "Inverted or negative"),
        ((0, 64), "Tensor span length mismatch"),
    ],
)
def test_qpf013_single_span_matrix(offsets: tuple[int, int], expected_prefix: str | None) -> None:
    titles = _titles(_meta((_tensor("t.weight", offsets=offsets),)), "QPF013")
    if expected_prefix is None:
        assert titles == []
    else:
        assert len(titles) == 1 and titles[0].startswith(expected_prefix)


def test_qpf013_packed_tensor_is_exempt_from_span_check() -> None:
    packed = _tensor("t.qweight", (8, 8), bits=4.0, logical=(4, 4), offsets=(0, 3))
    assert _titles(_meta((packed,), quant=AWQ4), "QPF013") == []


def test_qpf013_adjacent_spans_do_not_overlap() -> None:
    meta = _meta((_tensor("a.weight", offsets=(0, 128)), _tensor("b.weight", offsets=(128, 256))))
    assert _titles(meta, "QPF013") == []


def test_qpf013_empty_tensor_span_is_valid() -> None:
    assert _titles(_meta((_tensor("e.weight", (0, 8), offsets=(0, 0)),)), "QPF013") == []


@pytest.mark.parametrize(("file_size", "flagged"), [(100, False), (101, True), (228, False)])
def test_qpf013_eof_ignores_header_only_files(file_size: int, flagged: bool) -> None:
    meta = _meta(
        (_tensor("t.weight", offsets=(0, 128)),),
        shard_file_sizes={"model.safetensors": file_size},
        shard_header_lengths={"model.safetensors": 100},
    )
    assert bool(_titles(meta, "QPF013")) is flagged


# QPF014

_PICKLE = "Insecure legacy PyTorch pickle checkpoint format (arbitrary code execution hazard)"
_LINGER = "Redundant legacy pickle weight files lingering alongside Safetensors"


@pytest.mark.parametrize(
    ("container", "tensors", "expected"),
    [
        ("pytorch_bin", (_tensor("a.weight"),), _PICKLE),
        ("safetensors", (), _PICKLE),
        ("safetensors", (_tensor("a.weight"),), _LINGER),
    ],
)
def test_qpf014_pickle_severity_split(container: str, tensors: tuple[TensorSpec, ...], expected: str) -> None:
    meta = _meta(tensors, container=container, legacy_weight_files=("pytorch_model.bin",))
    assert expected in _titles(meta, "QPF014")


@pytest.mark.parametrize(("count", "suffix_present"), [(3, False), (4, True)])
def test_qpf014_file_preview_truncation(count: int, suffix_present: bool) -> None:
    files = tuple(f"w{i}.bin" for i in range(count))
    report = analyze_checkpoint(_meta(container="pytorch_bin", legacy_weight_files=files))
    message = next(f for f in report.findings if f.rule_id == "QPF014").message
    assert (" more)" in message) is suffix_present


@pytest.mark.parametrize(
    ("auto_map", "code", "flagged"),
    [
        ({"AutoModel": "m.X"}, ("m.py",), True),
        ({"AutoModel": "m.X"}, (), False),
        ({}, ("m.py",), False),
        (None, ("m.py",), False),
    ],
)
def test_qpf014_remote_code_needs_map_and_files(auto_map: Any, code: tuple[str, ...], flagged: bool) -> None:
    config = {**LLAMA, **({} if auto_map is None else {"auto_map": auto_map})}
    assert bool(_titles(_meta(config=config, custom_code_files=code), "QPF014")) is flagged


# Report labels and formatters


def test_report_quant_summary_and_architecture_label() -> None:
    plain = analyze_checkpoint(_meta((_tensor("a.weight"),)))
    assert plain.quant_summary == "unquantized (F16)"
    assert plain.architecture == "LlamaForCausalLM"

    full = QuantSpec(method="gptq", bits=4.0, group_size=128, format_version="marlin")
    assert analyze_checkpoint(_meta(quant=full)).quant_summary == "gptq 4-bit g128 marlin"
    assert analyze_checkpoint(_meta(quant=QuantSpec(method="awq"))).quant_summary == "awq"

    assert analyze_checkpoint(_meta(config={"architectures": [], "model_type": "x"})).architecture == "x"
    assert analyze_checkpoint(_meta(config={"architectures": "y", "model_type": "x"})).architecture == "x"


@pytest.mark.parametrize(
    ("size", "text"),
    [(512, "512 B"), (2048, "2.0 KiB"), (3 * 1024**2, "3.00 MiB"), (5 * 1024**3, "5.00 GiB")],
)
def test_fmt_bytes_units(size: int, text: str) -> None:
    assert _fmt_bytes(size) == text


def test_formatters_render_findings_and_clean_reports() -> None:
    bad = analyze_checkpoint(_meta(config={**LLAMA, "num_attention_heads": 9}), TargetSpec(gpu="H100", tp=2))
    text = format_text_report(bad)
    assert "QPF006" in text and "gpu=H100" in text and "Fix:" in text
    assert '"rule_id": "QPF006"' in format_json_report(bad)
    assert '"level": "error"' in format_sarif_report(bad)

    assert "Result: PASS (0 findings)" in format_text_report(analyze_checkpoint(_meta()))


@pytest.mark.parametrize(
    ("query", "shown"),
    [("H100", "gpu=H100,"), ("rtx4090", "gpu=RTX4090,"), ("sm_89", "gpu=L40S (from sm_89),"), ("Voodoo3", "gpu=Voodoo3,")],
)
def test_text_report_names_the_resolved_gpu(query: str, shown: str) -> None:
    report = analyze_checkpoint(_meta())
    report.target = TargetSpec(gpu=query)
    assert shown in format_text_report(report)


def test_severity_threshold_gates_pass_result() -> None:
    meta = _meta(config={**LLAMA, "num_attention_heads": 8, "num_key_value_heads": 2})
    report = analyze_checkpoint(meta, TargetSpec(tp=4))
    assert report.highest_severity() == Severity.MEDIUM
    assert report.passed(Severity.HIGH) and not report.passed(Severity.MEDIUM) and report.passed(None)


# Remaining edges


@pytest.mark.parametrize(("token_id", "flagged"), [(99, False), (100, True)])
def test_qpf002_lm_head_boundary(token_id: int, flagged: bool) -> None:
    meta = _meta(
        (_tensor("model.embed_tokens.weight", (500, 8)), _tensor("lm_head.weight", (100, 8))),
        tokenizer_meta=TokenizerMeta(max_token_id=token_id, max_token_str="<x>"),
    )
    assert bool(_titles(meta, "QPF002")) is flagged


def test_qpf002_reads_flat_tensors() -> None:
    meta = _meta(
        (_tensor("model.embed_tokens.weight", (500,)), _tensor("lm_head.weight", (100,))),
        tokenizer_meta=TokenizerMeta(max_token_id=200, max_token_str="<x>"),
    )
    assert _titles(meta, "QPF002") == ["Tokenizer ID exceeds lm_head output vocabulary rows"]


def test_qpf007_handles_a_two_part_tensor_name() -> None:
    meta = _meta((_tensor("q_proj.qweight", (129, 256), bits=4.0),), quant=AWQ4)
    assert _titles(meta, "QPF007", gpu="H100", tp=2) == [_TP_ERR]


def test_vram_values_at_the_integer_limit_still_count() -> None:
    meta = _meta(config={})
    config = {"num_hidden_layers": 1, "num_attention_heads": 1, "hidden_size": 1}
    target = TargetSpec(context_len=10**12)
    assert estimate_per_gpu_vram_gb(meta, config, target)[2] > 0
    assert estimate_per_gpu_vram_gb(meta, {**config, "hidden_size": 10**12}, TargetSpec(context_len=1))[2] > 0
    assert estimate_per_gpu_vram_gb(meta, {**config, "hidden_size": 10**12 + 1}, TargetSpec(context_len=1))[2] == 0


@pytest.mark.parametrize(
    ("size", "text"),
    [(1023, "1023 B"), (1024, "1.0 KiB"), (1024**2, "1.00 MiB"), (1024**3, "1.00 GiB")],
)
def test_fmt_bytes_unit_boundaries(size: int, text: str) -> None:
    assert _fmt_bytes(size) == text


# QPF015


@pytest.fixture
def advisories(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data: dict[str, Any] = {
        "updated": "2026-01-01",
        "engines": {
            "vllm": [
                {"id": "CVE-NEW", "ranges": [{"introduced": "0.10.0", "fixed": "0.20.0", "last_affected": None}]},
                {"id": "CVE-OLD", "ranges": [{"introduced": "0", "fixed": "0.12.1", "last_affected": None}]},
                {"id": "CVE-CAPPED", "ranges": [{"introduced": "0.30.0", "fixed": None, "last_affected": "0.31.0"}]},
                {"id": "CVE-OPEN", "ranges": [{"introduced": "0.40.0", "fixed": None, "last_affected": None}]},
                {
                    "id": "CVE-TWO",
                    "ranges": [
                        {"introduced": "0.1.0", "fixed": "0.2.0", "last_affected": None},
                        {"introduced": "0.5.0", "fixed": "0.6.0", "last_affected": None},
                    ],
                },
            ]
        },
    }
    monkeypatch.setattr("quantpreflight.rules.load_advisories", lambda: data)
    return data


def _advisory_finding(version: str, engine: str = "vllm") -> Any:
    report = analyze_checkpoint(_meta(), TargetSpec(engine=engine, engine_version=version))
    return next((f for f in report.findings if f.rule_id == "QPF015"), None)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.0.1", ["CVE-OLD"]),
        ("0.1.0", ["CVE-OLD", "CVE-TWO"]),
        ("0.2.0", ["CVE-OLD"]),
        ("0.5.5", ["CVE-OLD", "CVE-TWO"]),
        ("0.6.0", ["CVE-OLD"]),
        ("0.9.9", ["CVE-OLD"]),
        ("0.10.0", ["CVE-NEW", "CVE-OLD"]),
        ("0.10", ["CVE-NEW", "CVE-OLD"]),
        ("0.12.1", ["CVE-NEW"]),
        ("0.19.9", ["CVE-NEW"]),
        ("0.20.0", None),
        ("0.29.9", None),
        ("0.30.0", ["CVE-CAPPED"]),
        ("0.31.0", ["CVE-CAPPED"]),
        ("0.31.1", None),
        ("0.40.0", ["CVE-OPEN"]),
        ("1.0.0", ["CVE-OPEN"]),
        ("v0.19.0", ["CVE-NEW"]),
        ("0.19.0rc1", ["CVE-NEW"]),
        (" 0.19.0.post1 ", ["CVE-NEW"]),
    ],
)
def test_qpf015_version_ranges(advisories: dict[str, Any], version: str, expected: list[str] | None) -> None:
    finding = _advisory_finding(version)
    if expected is None:
        assert finding is None
    else:
        assert f"affected range of {len(expected)} advisor" in finding.message
        assert all(item in finding.message for item in expected)


def test_qpf015_recommends_the_newest_fix(advisories: dict[str, Any]) -> None:
    finding = _advisory_finding("0.11.0")
    assert finding.severity is Severity.HIGH
    assert finding.remediation == "Upgrade vLLM to 0.20.0 or newer."
    assert finding.references == ("https://osv.dev/vulnerability/CVE-NEW", "https://osv.dev/vulnerability/CVE-OLD")
    assert "2026-01-01" in finding.message


def test_qpf015_without_a_fix_is_medium(advisories: dict[str, Any]) -> None:
    finding = _advisory_finding("0.40.0")
    assert finding.severity is Severity.MEDIUM
    assert finding.remediation == "No fixed vLLM release is listed yet."


def test_qpf015_mixed_fixed_and_open_advisories(advisories: dict[str, Any]) -> None:
    advisories["engines"]["vllm"][3]["ranges"][0]["introduced"] = "0"
    finding = _advisory_finding("0.11.0")
    assert finding.severity is Severity.HIGH
    assert finding.remediation == "Upgrade vLLM to 0.20.0 or newer. Review the 1 unfixed advisory too."


def test_qpf015_lists_five_advisories_at_most(advisories: dict[str, Any]) -> None:
    advisories["engines"]["vllm"] = [
        {"id": f"CVE-{i}", "ranges": [{"introduced": "0", "fixed": "9", "last_affected": None}]} for i in range(7)
    ]
    finding = _advisory_finding("1.0")
    assert finding.title == "vLLM 1.0 has 7 published advisories"
    assert len(finding.references) == 5
    assert "CVE-5" not in finding.message


def test_qpf015_is_off_without_a_version(advisories: dict[str, Any]) -> None:
    assert _titles(_meta(), "QPF015") == []


def test_qpf015_has_no_data_for_other_engines(advisories: dict[str, Any]) -> None:
    assert _advisory_finding("0.1.0", engine="trtllm") is None


@pytest.mark.parametrize("version", ["x", "", "latest", "v"])
def test_qpf015_rejects_unreadable_versions(advisories: dict[str, Any], version: str) -> None:
    if version:
        with pytest.raises(ValueError, match="Invalid engine version"):
            _advisory_finding(version)
    else:
        assert _advisory_finding(version) is None


def test_bundled_advisories_cover_the_cve_this_project_cites() -> None:
    assert _advisory_finding("0.31.0").severity is Severity.HIGH
    assert "CVE-2026-115916" in {item["id"] for item in load_advisories()["engines"]["vllm"]}
    assert all(item["ranges"] for item in load_advisories()["engines"]["vllm"])


# Corrections from real vLLM runs


def _severity(meta: CheckpointMeta, rule: str, **target: Any) -> list[str]:
    report = analyze_checkpoint(meta, TargetSpec(**target))
    return [f.severity.value for f in report.findings if f.rule_id == rule]


def test_bf16_on_a_gpu_without_it_is_only_medium() -> None:
    assert _severity(_meta(_BF16), "QPF004", gpu="T4") == ["medium"]


def test_fp8_without_cores_is_high_because_real_loads_were_mixed() -> None:
    assert _severity(_meta(_F16, quant=_FP8_BLOCK), "QPF004", gpu="T4") == ["high"]


@pytest.mark.parametrize(
    ("raw", "gpu", "engine", "flagged"),
    [
        ({"quant_method": "modelopt"}, "T4", "vllm", True),
        ({"quant_algo": "FP8"}, "T4", "vllm", True),
        ({"quantization": {"quant_algo": "FP8"}}, "T4", "vllm", True),
        ({"quant_algo": "FP8"}, "A100", "vllm", False),
        ({"quant_algo": "FP8"}, "T4", "sglang", False),
        ({"quant_algo": "FP8"}, "MI250X", "vllm", False),
        ({"quant_method": "fp8"}, "T4", "vllm", False),
    ],
)
def test_modelopt_needs_compute_capability_80(raw: dict[str, Any], gpu: str, engine: str, flagged: bool) -> None:
    meta = _meta(_F16, quant=QuantSpec(method="fp8", bits=8.0, raw_config=raw))
    titles = _titles(meta, "QPF004", gpu=gpu, engine=engine)
    assert any(title.startswith("ModelOpt checkpoints need sm_80") for title in titles) is flagged


@pytest.mark.parametrize(
    ("context", "limit", "engine", "flagged"),
    [
        (4096, 2048, "vllm", True),
        (2048, 2048, "vllm", False),
        (2049, 2048, "vllm", True),
        (None, 2048, "vllm", False),
        (4096, None, "vllm", False),
        (4096, 2048, "sglang", False),
    ],
)
def test_qpf016_context_limit(context: int | None, limit: int | None, engine: str, flagged: bool) -> None:
    config = {**LLAMA, **({} if limit is None else {"max_position_embeddings": limit})}
    assert bool(_titles(_meta(config=config), "QPF016", context_len=context, engine=engine)) is flagged


def test_qpf016_names_a_setting_that_works() -> None:
    report = analyze_checkpoint(_meta(config={**LLAMA, "max_position_embeddings": 2048}), TargetSpec(context_len=4096))
    assert report.findings[0].remediation == "Set --context-len to 2048 or lower."


@pytest.mark.parametrize(
    ("config", "tokenizer", "expected"),
    [
        ({"max_position_embeddings": 8192, "n_positions": 4096}, {}, 4096),
        ({"max_position_embeddings": 8192, "model_max_length": 2048}, {}, 2048),
        ({"max_position_embeddings": 8192}, {"model_max_length": 4096}, 4096),
        ({"max_position_embeddings": 4096, "rope_scaling": {"type": "linear", "factor": 4}}, {}, 16384),
        ({"max_position_embeddings": 4096, "rope_scaling": {"type": "yarn", "factor": 4}}, {}, 4096),
    ],
)
def test_context_limit_matches_vllm_precedence(config: dict[str, Any], tokenizer: dict[str, Any], expected: int) -> None:
    assert context_limit(config, tokenizer) == expected


# Corrections from scanning 500 public models


@pytest.mark.parametrize(
    ("name", "found"),
    [
        ("output.weight", True),
        ("transformer.output.weight", True),
        ("blk.0.attn_output.weight", False),
        ("blk.0.ffn_output.weight", False),
        ("my_lm_head.weight", False),
    ],
)
def test_head_names_match_whole_parts(name: str, found: bool) -> None:
    meta = _meta(
        (_tensor("token_embd.weight", (500, 8)), _tensor(name, (100, 8))),
        tokenizer_meta=TokenizerMeta(max_token_id=200, max_token_str="<x>"),
    )
    assert bool(_titles(meta, "QPF002")) is found


def test_embedding_names_match_whole_parts() -> None:
    meta = _meta((_tensor("detoken_embd.weight", (5, 8)),), config={**LLAMA, "vocab_size": 100})
    assert _titles(meta, "QPF002") == []


def test_stop_token_drift_is_medium_because_vllm_also_uses_the_tokenizer() -> None:
    meta = _meta(tokenizer_meta=TokenizerMeta(eos_token="<e>", eos_token_ids=(3,)), config={**LLAMA, "eos_token_id": 2})
    assert _severity(meta, "QPF003") == ["medium"]
    assert _severity(_meta(config={**LLAMA, "eos_token_id": 2}, generation_config={"eos_token_id": [5]}), "QPF003") == ["medium"]


@pytest.mark.parametrize(
    ("quant", "severity"),
    [
        (QuantSpec(method="compressed-tensors", bits=4.0, format_version="pack-quantized"), "medium"),
        (QuantSpec(method="awq", bits=4.0), "high"),
        (QuantSpec(method="gptq", bits=4.0, format_version="marlin"), "critical"),
    ],
)
def test_marlin_tile_severity(quant: QuantSpec, severity: str) -> None:
    meta = _meta((_tensor(_COL, (64, 100), bits=4.0),), quant=quant)
    assert _severity(meta, "QPF007", gpu="H100") == [severity]


def test_float_compressed_tensors_skip_the_marlin_check() -> None:
    quant = QuantSpec(method="compressed-tensors", bits=8.0, format_version="float-quantized")
    meta = _meta((_tensor(_COL, (64, 100), bits=8.0),), quant=quant)
    assert _titles(meta, "QPF007", gpu="H100") == []


def test_mlx_models_are_named_and_not_called_stale() -> None:
    meta = _meta((_tensor("a.weight", dtype="U32", bits=32.0),), quant=QuantSpec(method="mlx", bits=4.0))
    assert _titles(meta, "QPF010") == []
    assert _titles(meta, "QPF011", engine="vllm") == ["Quantization method 'mlx' unsupported by vLLM"]


def test_pickle_weights_are_a_security_warning_not_an_engine_error() -> None:
    meta = _meta(container="pytorch_bin", legacy_weight_files=("pytorch_model.bin",))
    assert _titles(meta, "QPF011", engine="vllm") == []
    assert _titles(meta, "QPF011", engine="trtllm")
    assert _titles(meta, "QPF014", engine="vllm")


def test_marlin_packed_weights_count_as_quantized() -> None:
    tensor = _tensor("model.layers.0.self_attn.q_proj.marlin_qweight", dtype="I32", bits=32.0)
    assert _titles(_meta((tensor,), quant=QuantSpec(method="gptq", bits=4.0, format_version="marlin")), "QPF010") == []


def test_duplicate_tensors_are_medium() -> None:
    assert _severity(_meta(duplicate_tensors=("a.weight",)), "QPF009") == ["medium"]
