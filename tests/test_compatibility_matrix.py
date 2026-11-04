"""Rule output compared with answers worked out separately from the GPU tables and plain arithmetic."""

from __future__ import annotations

from itertools import product
from typing import Any

import pytest

from quantpreflight.models import CheckpointMeta, QuantSpec, TargetSpec, TensorSpec
from quantpreflight.profiles import load_gpu_profiles
from quantpreflight.rules import analyze_checkpoint, estimate_per_gpu_vram_gb

GPUS = sorted(load_gpu_profiles().values(), key=lambda gpu: gpu.id)
ENGINES = ["vllm", "sglang", "trtllm"]
LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}


def _ids(value: Any) -> str:
    return getattr(value, "id", str(value))


def _meta(config: dict[str, Any], quant: QuantSpec | None = None, **kwargs: Any) -> CheckpointMeta:
    return CheckpointMeta(source="mock", container_format="safetensors", config=config, quant=quant or QuantSpec(), **kwargs)


def _rule(meta: CheckpointMeta, rule: str, **target: Any) -> list[str]:
    return [f.title for f in analyze_checkpoint(meta, TargetSpec(**target)).findings if f.rule_id == rule]


def _has(titles: list[str], prefix: str) -> bool:
    return any(title.startswith(prefix) for title in titles)


# Hardware support follows the capability tables


@pytest.mark.parametrize(("gpu", "engine"), list(product(GPUS, ENGINES)), ids=_ids)
def test_nvfp4_needs_nvfp4_cores(gpu: Any, engine: str) -> None:
    meta = _meta(LLAMA, QuantSpec(method="nvfp4", bits=4.0))
    assert _has(_rule(meta, "QPF004", gpu=gpu.id, engine=engine), "NVFP4 requires") is (not gpu.supports_nvfp4)


@pytest.mark.parametrize(("gpu", "engine"), list(product(GPUS, ENGINES)), ids=_ids)
def test_prepacked_marlin_needs_marlin_support(gpu: Any, engine: str) -> None:
    meta = _meta(LLAMA, QuantSpec(method="gptq", bits=4.0, format_version="marlin"))
    assert _has(_rule(meta, "QPF004", gpu=gpu.id, engine=engine), "Pre-packed Marlin") is (not gpu.supports_marlin)


@pytest.mark.parametrize("gpu", GPUS, ids=_ids)
def test_bf16_weights_need_bf16_support(gpu: Any) -> None:
    tensor = TensorSpec("a.weight", "BF16", (8, 8), "s", 128, 16.0)
    meta = _meta(LLAMA, tensors={"a.weight": tensor})
    assert _has(_rule(meta, "QPF004", gpu=gpu.id), "BF16 runs as FP16") is (not gpu.supports_bf16)


@pytest.mark.parametrize("gpu", GPUS, ids=_ids)
def test_fp8_activations_need_fp8_cores(gpu: Any) -> None:
    meta = _meta(LLAMA, QuantSpec(method="fp8", bits=8.0, activation_dtype="fp8"))
    assert _has(_rule(meta, "QPF004", gpu=gpu.id), "No FP8 cores") is (not gpu.supports_fp8_w8a8)


@pytest.mark.parametrize("gpu", GPUS, ids=_ids)
def test_weight_only_fp8_runs_where_marlin_can_take_over(gpu: Any) -> None:
    meta = _meta(LLAMA, QuantSpec(method="fp8", bits=8.0))
    expected = not gpu.supports_fp8_w8a8 and not gpu.supports_marlin
    assert _has(_rule(meta, "QPF004", gpu=gpu.id), "No FP8 cores") is expected


@pytest.mark.parametrize(("gpu", "engine"), list(product(GPUS, ["vllm", "sglang"])), ids=_ids)
def test_block_fp8_is_restricted_on_ada_only(gpu: Any, engine: str) -> None:
    meta = _meta(LLAMA, QuantSpec(method="fp8", bits=8.0, weight_block_size=(128, 128)))
    titles = _rule(meta, "QPF004", gpu=gpu.id, engine=engine)
    assert _has(titles, "Block-scaled FP8") is (gpu.arch == "sm_89")


@pytest.mark.parametrize("gpu", GPUS, ids=_ids)
def test_tensorrt_has_no_block_fp8_restriction(gpu: Any) -> None:
    meta = _meta(LLAMA, QuantSpec(method="fp8", bits=8.0, weight_block_size=(128, 128)))
    assert not _has(_rule(meta, "QPF004", gpu=gpu.id, engine="trtllm"), "Block-scaled FP8")


# Kernel hazards


@pytest.mark.parametrize(
    ("gpu", "method", "bits"),
    list(product(GPUS, ["awq", "gptq", "compressed-tensors", "autoround"], [4.0, 8.0])),
    ids=_ids,
)
def test_marlin_fp8_corruption_is_limited_to_sm120_and_4_bit(gpu: Any, method: str, bits: float) -> None:
    meta = _meta(LLAMA, QuantSpec(method=method, bits=bits))
    expected = bits == 4.0 and gpu.compute_capability in (120, 121)
    titles = _rule(meta, "QPF005", gpu=gpu.id, marlin_input_dtype="fp8")
    assert _has(titles, "Silent Marlin W4A8-FP8") is expected


@pytest.mark.parametrize("dtype", [None, "fp16", "bf16"])
def test_marlin_fp8_rule_needs_the_flag(dtype: str | None) -> None:
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0))
    assert _rule(meta, "QPF005", gpu="GB10", marlin_input_dtype=dtype) == []


def test_marlin_flag_is_read_in_any_case() -> None:
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0))
    assert _rule(meta, "QPF005", gpu="GB10", marlin_input_dtype=" FP8 ")


@pytest.mark.parametrize(("engine", "flagged"), [("vllm", True), ("sglang", False)])
def test_marlin_rule_is_vllm_only(engine: str, flagged: bool) -> None:
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0))
    assert bool(_rule(meta, "QPF005", gpu="GB10", marlin_input_dtype="fp8", engine=engine)) is flagged


def test_4_bit_moe_on_gb10_gets_a_medium_hint_without_the_flag() -> None:
    tensor = TensorSpec("mlp.experts.0.up_proj.qweight", "I32", (8, 8), "s", 64, 4.0)
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0), tensors={tensor.name: tensor})
    assert _has(_rule(meta, "QPF005", gpu="GB10"), "Avoid VLLM_MARLIN_INPUT_DTYPE")
    assert _rule(_meta(LLAMA, QuantSpec(method="awq", bits=4.0)), "QPF005", gpu="GB10") == []


# Tensor parallelism

HEADS = [1, 2, 3, 4, 6, 7, 8, 9, 12, 16, 24, 32, 64]
TP = [2, 3, 4, 6, 8, 16]


@pytest.mark.parametrize(("heads", "tp"), list(product(HEADS, TP)))
def test_head_divisibility(heads: int, tp: int) -> None:
    config = {**LLAMA, "num_attention_heads": heads}
    titles = _rule(_meta(config), "QPF006", tp=tp)
    assert ("num_attention_heads not divisible by TP degree" in titles) is (heads % tp != 0)


@pytest.mark.parametrize(("kv_heads", "tp"), list(product(HEADS, TP)))
def test_kv_head_outcomes(kv_heads: int, tp: int) -> None:
    config = {**LLAMA, "num_attention_heads": 96, "num_key_value_heads": kv_heads}
    titles = _rule(_meta(config), "QPF006", tp=tp)
    if kv_heads >= tp:
        expected = {"GQA num_key_value_heads not divisible by TP degree"} if kv_heads % tp else set()
    elif tp % kv_heads:
        expected = {"TP degree is not a multiple of num_key_value_heads"}
    else:
        expected = {"KV cache heads will be replicated across TP ranks"}
    assert {t for t in titles if "num_attention_heads" not in t} == expected


@pytest.mark.parametrize(("experts", "tp"), list(product([2, 3, 4, 8, 16, 128], TP)))
@pytest.mark.parametrize("key", ["num_local_experts", "n_routed_experts", "num_experts"])
def test_expert_divisibility(key: str, experts: int, tp: int) -> None:
    titles = _rule(_meta({**LLAMA, key: experts}), "QPF006", tp=tp)
    assert ("MoE expert count not divisible by TP degree" in titles) is (experts % tp != 0)


@pytest.mark.parametrize("heads", HEADS)
def test_a_single_gpu_never_has_a_tp_problem(heads: int) -> None:
    assert _rule(_meta({**LLAMA, "num_attention_heads": heads, "num_local_experts": 7}), "QPF006", tp=1) == []


# Alignment

SIZES = [64, 128, 192, 256, 1024, 4096]


def _linear(name: str, out_f: int, in_f: int) -> TensorSpec:
    return TensorSpec(name, "I32", (out_f, in_f), "s", 64, 4.0)


@pytest.mark.parametrize(("out_f", "in_f", "tp"), list(product(SIZES, SIZES, [1, 2, 4])))
def test_marlin_alignment_of_col_parallel_layers(out_f: int, in_f: int, tp: int) -> None:
    tensor = _linear("model.layers.0.self_attn.q_proj.weight", out_f, in_f)
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0), tensors={tensor.name: tensor})
    expected = (out_f // tp) % 64 != 0 or in_f % 128 != 0
    assert _has(_rule(meta, "QPF007", gpu="H100", tp=tp), "Sharded quantized dimensions") is expected


@pytest.mark.parametrize(("out_f", "in_f", "tp"), list(product(SIZES, SIZES, [1, 2, 4])))
def test_marlin_alignment_of_row_parallel_layers(out_f: int, in_f: int, tp: int) -> None:
    tensor = _linear("model.layers.0.self_attn.o_proj.weight", out_f, in_f)
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0), tensors={tensor.name: tensor})
    expected = out_f % 64 != 0 or (in_f // tp) % 128 != 0
    assert _has(_rule(meta, "QPF007", gpu="H100", tp=tp), "Sharded quantized dimensions") is expected


@pytest.mark.parametrize(("in_f", "group", "tp"), list(product([256, 512, 1024], [64, 100, 128, 256], [1, 2, 4])))
def test_group_size_after_sharding(in_f: int, group: int, tp: int) -> None:
    tensor = _linear("model.layers.0.self_attn.o_proj.weight", 64, in_f)
    meta = _meta(LLAMA, QuantSpec(method="gptq", bits=4.0, group_size=group), tensors={tensor.name: tensor})
    expected = (in_f // tp) % group != 0
    assert _has(_rule(meta, "QPF007", gpu="H100", tp=tp), "Row-parallel sharded input") is expected


@pytest.mark.parametrize(
    "method",
    ["none", "gguf", "bitsandbytes"],
)
def test_alignment_skips_methods_without_fixed_tiles(method: str) -> None:
    tensor = _linear("model.layers.0.self_attn.q_proj.weight", 63, 63)
    meta = _meta(LLAMA, QuantSpec(method=method, bits=4.0), tensors={tensor.name: tensor})
    assert _rule(meta, "QPF007", gpu="H100", tp=2) == []


def test_marlin_alignment_is_not_checked_on_gpus_without_marlin() -> None:
    tensor = _linear("model.layers.0.self_attn.q_proj.weight", 63, 63)
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0), tensors={tensor.name: tensor})
    assert _rule(meta, "QPF007", gpu="MI300X") == []
    assert _rule(meta, "QPF007", gpu="H100")
    assert _rule(meta, "QPF007")


@pytest.mark.parametrize(("bits", "checked"), [(4.0, True), (8.0, True), (2.0, False), (6.0, False)])
def test_marlin_alignment_only_applies_to_4_and_8_bit(bits: float, checked: bool) -> None:
    tensor = _linear("model.layers.0.self_attn.q_proj.weight", 63, 63)
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=bits), tensors={tensor.name: tensor})
    assert _has(_rule(meta, "QPF007", gpu="H100"), "Sharded quantized dimensions") is checked


def test_marlin_error_is_critical_for_prepacked_checkpoints() -> None:
    tensor = _linear("model.layers.0.self_attn.q_proj.weight", 63, 63)
    for fmt, severity in (("marlin", "critical"), ("gptq", "high")):
        meta = _meta(LLAMA, QuantSpec(method="gptq", bits=4.0, format_version=fmt), tensors={tensor.name: tensor})
        finding = next(f for f in analyze_checkpoint(meta, TargetSpec(gpu="H100")).findings if f.rule_id == "QPF007")
        assert finding.severity.value == severity


def test_tensors_outside_linear_layers_are_not_aligned() -> None:
    tensors = [_linear("model.embed_tokens.weight", 63, 63), _linear("lm_head.weight", 63, 63), _linear("single", 63, 63), _linear("model.layers.0.self_attn.q_proj.scales", 63, 63)]
    meta = _meta(LLAMA, QuantSpec(method="awq", bits=4.0), tensors={t.name: t for t in tensors})
    assert _rule(meta, "QPF007", gpu="H100") == []


# Memory


@pytest.mark.parametrize(
    ("weights_gib", "tp", "vram"),
    list(product([1, 2, 8, 20, 40, 80], [1, 2, 4, 8], [8.0, 16.0, 24.0, 48.0, 80.0])),
)
def test_memory_budget_without_kv_cache(weights_gib: int, tp: int, vram: float) -> None:
    meta = _meta({}, total_weight_bytes=weights_gib * 1024**3)
    report = analyze_checkpoint(meta, TargetSpec(tp=tp, vram_gb_override=vram))
    total = round(weights_gib / tp * 1.18 + 0.65, 2)
    assert report.estimated_vram_per_gpu_gb == total
    assert any(f.rule_id == "QPF012" for f in report.findings) is (total > round(vram * 0.9, 2))


@pytest.mark.parametrize("gpu", GPUS, ids=_ids)
def test_memory_budget_uses_the_gpu_table(gpu: Any) -> None:
    fits = _meta({}, total_weight_bytes=int(gpu.vram_gb * 0.6 * 1024**3))
    too_big = _meta({}, total_weight_bytes=int(gpu.vram_gb * 0.9 * 1024**3))
    assert _rule(fits, "QPF012", gpu=gpu.id) == []
    assert _rule(too_big, "QPF012", gpu=gpu.id)


def test_memory_override_wins_over_the_gpu_table() -> None:
    meta = _meta({}, total_weight_bytes=100 * 1024**3)
    assert _rule(meta, "QPF012", gpu="H100")
    assert _rule(meta, "QPF012", gpu="H100", vram_gb_override=1000.0) == []


def test_memory_rule_needs_a_budget() -> None:
    report = analyze_checkpoint(_meta({}, total_weight_bytes=10**15))
    assert report.estimated_vram_per_gpu_gb is not None
    assert not report.findings


def test_memory_utilization_changes_the_budget() -> None:
    meta = _meta({}, total_weight_bytes=8 * 1024**3)
    assert _rule(meta, "QPF012", vram_gb_override=12.0, gpu_memory_utilization=0.5)
    assert _rule(meta, "QPF012", vram_gb_override=12.0, gpu_memory_utilization=0.9) == []


def test_memory_finding_recommends_enough_gpus() -> None:
    meta = _meta({}, total_weight_bytes=40 * 1024**3)
    finding = analyze_checkpoint(meta, TargetSpec(gpu="L4")).findings[0]
    assert "at least 3" in finding.remediation
    assert "L4" in finding.title


def test_memory_label_without_a_named_gpu() -> None:
    meta = _meta({}, total_weight_bytes=40 * 1024**3)
    assert "8 GiB GPU" in analyze_checkpoint(meta, TargetSpec(vram_gb_override=8.0)).findings[0].title


# KV cache arithmetic

STANDARD = {"num_hidden_layers": 8, "num_attention_heads": 8, "hidden_size": 800}
CONTEXT = 1_000_000


def _kv_bytes(config: dict[str, Any], quant: QuantSpec | None = None, tp: int = 1, context: int | None = CONTEXT) -> float:
    meta = _meta({}, quant)
    return estimate_per_gpu_vram_gb(meta, config, TargetSpec(tp=tp, context_len=context))[2]


def _gib(value: float) -> float:
    return round(value / 1024**3, 2)


def test_standard_kv_cache_formula() -> None:
    assert _kv_bytes(STANDARD) == _gib(2 * 8 * 8 * 100 * 2 * CONTEXT)


def test_gqa_uses_key_value_heads() -> None:
    assert _kv_bytes({**STANDARD, "num_key_value_heads": 2}) == _gib(2 * 8 * 2 * 100 * 2 * CONTEXT)


def test_head_dim_overrides_the_division() -> None:
    assert _kv_bytes({**STANDARD, "head_dim": 50}) == _gib(2 * 8 * 8 * 50 * 2 * CONTEXT)


def test_fp8_activations_halve_the_cache() -> None:
    assert _kv_bytes(STANDARD, QuantSpec(method="fp8", activation_dtype="fp8")) == _gib(2 * 8 * 8 * 100 * CONTEXT)


@pytest.mark.parametrize(("tp", "heads"), [(2, 4), (4, 2), (3, 3), (16, 1)])
def test_kv_heads_split_across_ranks(tp: int, heads: int) -> None:
    assert _kv_bytes(STANDARD, tp=tp) == _gib(2 * 8 * heads * 100 * 2 * CONTEXT)


def test_context_defaults_to_the_model_limit_capped_at_4096() -> None:
    assert _kv_bytes({**STANDARD, "max_position_embeddings": 1000}, context=None) == _gib(2 * 8 * 8 * 100 * 2 * 1000)
    assert _kv_bytes({**STANDARD, "max_position_embeddings": 100000}, context=None) == _gib(2 * 8 * 8 * 100 * 2 * 4096)
    assert _kv_bytes(STANDARD, context=None) == _gib(2 * 8 * 8 * 100 * 2 * 4096)


def test_latent_attention_formula() -> None:
    config = {**STANDARD, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    assert _kv_bytes(config) == _gib(8 * 576 * 2 * CONTEXT)
    assert _kv_bytes({**STANDARD, "kv_lora_rank": 512}) == _gib(8 * 576 * 2 * CONTEXT)


def test_mamba_state_formula() -> None:
    assert _kv_bytes({**STANDARD, "model_type": "mamba", "state_size": 16, "intermediate_size": 1000}) == _gib(8 * 1000 * 16 * 2)
    assert _kv_bytes({"model_type": "mamba2", "d_state": 8, "hidden_size": 500, "num_hidden_layers": 2}) == _gib(2 * 1000 * 8 * 2)
    assert _kv_bytes({"model_type": "mamba"}) == _gib(1 * 4096 * 16 * 2)


@pytest.mark.parametrize("key", ["sliding_window", "sliding_window_size"])
def test_sliding_window_bounds_the_cache(key: str) -> None:
    assert _kv_bytes({**STANDARD, key: 1000}) == _gib(2 * 8 * 8 * 100 * 2 * 1000)


def test_sliding_window_larger_than_the_context_changes_nothing() -> None:
    assert _kv_bytes({**STANDARD, "sliding_window": CONTEXT * 2}) == _kv_bytes(STANDARD)


def test_gemma2_alternates_sliding_and_global_layers() -> None:
    config = {**STANDARD, "model_type": "gemma2", "sliding_window": 1000}
    assert _kv_bytes(config) == _gib(2 * 8 * 8 * 100 * 2 * (1000 + CONTEXT) / 2)


# Geometry from https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json.
GPT_OSS = {
    "model_type": "gpt_oss",
    "num_hidden_layers": 24,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "hidden_size": 2880,
    "max_position_embeddings": 131072,
    "sliding_window": 128,
    "layer_types": ["sliding_attention", "full_attention"] * 12,
}


def test_gpt_oss_counts_the_cache_of_its_twelve_global_layers() -> None:
    expected = 2 * 8 * 64 * 2 * (12 * 128 + 12 * 131072)
    assert _kv_bytes(GPT_OSS, context=131072) == _gib(expected) == 3.0


@pytest.mark.parametrize("global_type", ["full_attention", "global_attention", "FULL_ATTENTION"])
def test_global_layer_types_keep_the_full_context(global_type: str) -> None:
    config = {**STANDARD, "sliding_window": 1000, "layer_types": [global_type, "sliding_attention"] * 4}
    assert _kv_bytes(config) == _gib(2 * 8 * 100 * 2 * (4 * CONTEXT + 4 * 1000))


@pytest.mark.parametrize(("pattern", "global_layers"), [(1, 8), (3, 2), (6, 1), (9, 0)])
def test_gemma3_counts_only_each_complete_window_pattern(pattern: int, global_layers: int) -> None:
    config = {**STANDARD, "model_type": "gemma3_text", "sliding_window": 1000, "sliding_window_pattern": pattern}
    expected = 2 * 8 * 100 * 2 * (global_layers * CONTEXT + (8 - global_layers) * 1000)
    assert _kv_bytes(config) == _gib(expected)


def test_global_layer_types_take_precedence_over_the_window_pattern() -> None:
    config = {
        **STANDARD,
        "sliding_window": 1000,
        "layer_types": ["full_attention"] + ["sliding_attention"] * 7,
        "sliding_window_pattern": 1,
    }
    assert _kv_bytes(config) == _gib(2 * 8 * 100 * 2 * (CONTEXT + 7 * 1000))


def test_gemma2_explicit_global_layer_types_override_its_default_alternation() -> None:
    config = {**STANDARD, "model_type": "gemma2", "sliding_window": 1000, "layer_types": ["sliding_attention"] * 8}
    assert _kv_bytes(config) == _gib(2 * 8 * 8 * 100 * 2 * 1000)


def test_mistral_without_sliding_metadata_keeps_every_layer_windowed() -> None:
    config = {**STANDARD, "model_type": "mistral", "sliding_window": 1000}
    assert _kv_bytes(config) == _gib(2 * 8 * 8 * 100 * 2 * 1000)


def test_global_layer_types_do_not_inflate_contexts_shorter_than_the_window() -> None:
    assert _kv_bytes(GPT_OSS, context=64) == _gib(2 * 24 * 8 * 64 * 2 * 64)


@pytest.mark.parametrize(("tp", "concurrency", "dtype"), [(1, 1, None), (2, 4, None), (16, 2, "fp8")])
def test_global_layers_keep_parallelism_precision_and_concurrency_scaling(tp: int, concurrency: int, dtype: str | None) -> None:
    meta = _meta(GPT_OSS, QuantSpec(activation_dtype=dtype))
    target = TargetSpec(tp=tp, context_len=131072, concurrency=concurrency)
    kv_heads = {1: 8, 2: 4, 16: 1}[tp]
    element_bytes = 1 if dtype == "fp8" else 2
    expected = 2 * kv_heads * 64 * element_bytes * (12 * 128 + 12 * 131072) * concurrency
    assert estimate_per_gpu_vram_gb(meta, GPT_OSS, target)[2] == _gib(expected)


def test_global_layers_reach_the_vram_budget_finding() -> None:
    meta = _meta(GPT_OSS, total_weight_bytes=16 * 1024**3)
    report = analyze_checkpoint(meta, TargetSpec(gpu="L4", context_len=131072))
    assert report.estimated_vram_per_gpu_gb == 22.53
    assert any(f.rule_id == "QPF012" for f in report.findings)


def test_gemma3_window_pattern_is_read_from_nested_text_config() -> None:
    text = {
        "model_type": "gemma3_text",
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "hidden_size": 2304,
        "head_dim": 256,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
    }
    meta = _meta({"model_type": "gemma3", "text_config": text}, total_weight_bytes=1024**3)
    report = analyze_checkpoint(meta, TargetSpec(vram_gb_override=3.0, context_len=131072))
    # Four global layers and 22 sliding layers, plus the existing weight/runtime overheads.
    expected = round(1.18 + 0.65 + 2 * 4 * 256 * 2 * (4 * 131072 + 22 * 1024) / 1024**3, 2)
    assert report.estimated_vram_per_gpu_gb == expected
    assert any(f.rule_id == "QPF012" for f in report.findings)


def test_global_layer_count_never_exceeds_the_number_of_attention_layers() -> None:
    config = {**STANDARD, "sliding_window": 1000, "layer_types": ["full_attention"] * 9}
    assert _kv_bytes(config) <= _kv_bytes(STANDARD)


def test_block_type_list_counts_attention_layers() -> None:
    config = {**STANDARD, "layers_block_type": ["attention", "mamba", "mamba", "mamba"]}
    assert _kv_bytes(config) == _gib(2 * 1 * 8 * 100 * 2 * CONTEXT)


def test_incomplete_config_has_no_cache_estimate() -> None:
    assert _kv_bytes({"num_hidden_layers": 8}) == 0.0
    assert _kv_bytes({"hidden_size": 8, "num_attention_heads": 2}) == 0.0


# Concurrency


@pytest.mark.parametrize("concurrency", [1, 2, 8, 32])
def test_concurrency_scales_every_kv_cache_model(concurrency: int) -> None:
    def kv(config: dict[str, Any]) -> float:
        meta = _meta({})
        return estimate_per_gpu_vram_gb(meta, config, TargetSpec(context_len=CONTEXT, concurrency=concurrency))[2]

    assert kv(STANDARD) == _gib(2 * 8 * 8 * 100 * 2 * CONTEXT * concurrency)
    assert kv({**STANDARD, "kv_lora_rank": 512}) == _gib(8 * 576 * 2 * CONTEXT * concurrency)
    assert kv({**STANDARD, "model_type": "mamba", "state_size": 16, "intermediate_size": 1000}) == _gib(8 * 1000 * 16 * 2 * concurrency)


def test_concurrency_reaches_the_budget_message() -> None:
    meta = _meta(STANDARD, total_weight_bytes=1024**3)
    target = TargetSpec(vram_gb_override=4.0, context_len=4096, concurrency=64)
    finding = next(f for f in analyze_checkpoint(meta, target).findings if f.rule_id == "QPF012")
    assert "for 64 requests" in finding.message


# Measured on real GPUs: (weights per GPU in GiB, KV cache for one 4096-token request, memory vLLM really used)
MEASURED = [
    (0.92, 0.047, 1.76),
    (2.60, 0.109, 3.94),
    (5.19, 0.219, 7.30),
    (8.79, 0.563, 11.42),
    (9.00, 0.501, 11.44),
    (9.29, 0.753, 12.51),
]


@pytest.mark.parametrize(("weights", "kv", "used"), MEASURED)
def test_memory_estimate_stays_close_to_real_runs(weights: float, kv: float, used: float) -> None:
    meta = _meta({}, total_weight_bytes=int(weights * 1024**3))
    estimate = estimate_per_gpu_vram_gb(meta, {}, TargetSpec())[0] + kv
    assert abs(estimate - used) / used < 0.12


def test_kv_estimate_matches_a_real_run() -> None:
    config = {"num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584}
    assert _kv_bytes(config, context=4096) == pytest.approx(0.219, rel=0.03)
