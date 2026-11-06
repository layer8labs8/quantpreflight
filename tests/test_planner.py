"""The setup plan: which GPU count and context length start, and the command that starts them."""

from __future__ import annotations

import json
from typing import Any

import pytest

from quantpreflight.formatters import format_json_report, format_text_report
from quantpreflight.models import CheckpointMeta, QuantSpec, TargetSpec
from quantpreflight.planner import (
    CONTEXT_HEADROOM_GIB,
    _max_fitting_context,
    _round_down,
    blocks_start,
    plan_setup,
    serve_command,
)
from quantpreflight.rules import RUNTIME_GIB, WEIGHT_OVERHEAD, _effective_text_config, analyze_checkpoint

GIB = 1024**3
# 32 layers and 8 KV heads of 128 dims, so the cache takes 0.125 MiB a token in 16-bit
LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "max_position_embeddings": 32768,
    "vocab_size": 32000,
}


# 16 GiB of weights on one L4 leave about this many tokens of cache, at 0.125 MiB a token. The
# estimate rounds to 0.01 GiB, which moves the count by up to 64 tokens
L4_FIT = int((24 * 0.9 - 16 * WEIGHT_OVERHEAD - RUNTIME_GIB - CONTEXT_HEADROOM_GIB) * 1024 * 8)
L4_CONTEXT = L4_FIT - L4_FIT % 1024


def _meta(weights_gib: float = 10, *, config: dict[str, Any] | None = None, **kwargs: Any) -> CheckpointMeta:
    return CheckpointMeta(
        source=kwargs.pop("source", "org/model"),
        container_format=kwargs.pop("container", "safetensors"),
        config=LLAMA if config is None else config,
        total_weight_bytes=int(weights_gib * GIB),
        **kwargs,
    )


def test_a_model_that_fits_is_ready_as_asked() -> None:
    plan = plan_setup(_meta(10), TargetSpec(gpu="H100"))
    assert plan.status == "ready" and plan.tp == 1 and not plan.changes
    assert plan.context_len == 32768 and plan.max_context == 32768
    # The full context is vLLM's default, so the command doesn't repeat it
    assert plan.command == "vllm serve org/model"


def test_setup_uses_tokenizer_context_cap_and_rope_scaling() -> None:
    config = {**LLAMA, "max_position_embeddings": 4096, "rope_scaling": {"type": "linear", "factor": 4}}
    meta = _meta(config=config, tokenizer_config={"model_max_length": 12000})
    plan = plan_setup(meta, TargetSpec(gpu="H100", context_len=20000))
    assert plan.context_len == 12000


def test_heads_that_do_not_split_move_to_the_closest_gpu_count_with_fewer_gpus_on_a_tie() -> None:
    plan = plan_setup(_meta(10), TargetSpec(gpu="H100", tp=3))
    assert plan.status == "changed" and plan.tp == 2
    assert plan.changes == ("Use --tp 2 instead of 3, so attention heads and experts split evenly.",)
    assert plan.command == "vllm serve org/model --tensor-parallel-size 2"


def test_weights_too_big_for_one_gpu_move_to_more_gpus() -> None:
    plan = plan_setup(_meta(30), TargetSpec(gpu="L4"))
    assert plan.status == "changed" and plan.tp == 2
    assert plan.changes[0] == "Use --tp 2 instead of 1, so the weights fit in memory."


def test_a_full_context_that_does_not_fit_is_cut_to_what_fits() -> None:
    plan = plan_setup(_meta(16), TargetSpec(gpu="L4"))
    assert plan.status == "changed" and plan.tp == 1
    assert plan.max_context is not None and abs(plan.max_context - L4_FIT) <= 64
    assert plan.context_len == L4_CONTEXT
    assert plan.changes == (f"Set the context to {L4_CONTEXT} tokens. The model's full 32768 won't fit in memory.",)
    assert plan.command == f"vllm serve org/model --max-model-len {L4_CONTEXT}"
    # The suggested setup passes the memory check with room to spare
    report = analyze_checkpoint(_meta(16), TargetSpec(gpu="L4", context_len=plan.context_len))
    assert not [f for f in report.findings if f.rule_id == "QPF012"]
    assert report.estimated_vram_per_gpu_gb is not None
    assert report.estimated_vram_per_gpu_gb + CONTEXT_HEADROOM_GIB <= 24 * 0.9


def test_a_context_above_the_model_limit_is_cut_to_the_limit() -> None:
    plan = plan_setup(_meta(10), TargetSpec(gpu="H100", context_len=200_000))
    assert plan.context_len == 32768
    assert plan.changes == ("Use a context of 32768 tokens instead of 200000, the model's limit.",)


def test_a_context_that_does_not_fit_in_memory_says_so() -> None:
    plan = plan_setup(_meta(16), TargetSpec(gpu="L4", context_len=30000))
    assert plan.context_len == L4_CONTEXT
    assert plan.changes == (f"Use a context of {L4_CONTEXT} tokens instead of 30000, what fits in memory.",)


def test_a_short_context_that_fits_is_kept() -> None:
    plan = plan_setup(_meta(16), TargetSpec(gpu="L4", context_len=8192))
    assert plan.status == "ready" and plan.context_len == 8192
    assert plan.command == "vllm serve org/model --max-model-len 8192"


def test_hardware_that_cannot_run_the_format_is_blocked() -> None:
    plan = plan_setup(_meta(5, quant=QuantSpec(method="nvfp4", bits=4)), TargetSpec(gpu="H100"))
    assert plan.status == "blocked" and plan.command is None
    assert {f.rule_id for f in plan.blocking} == {"QPF004"}
    assert plan.tp == 1 and not plan.changes


def test_a_model_too_big_for_every_gpu_count_says_how_far_it_looked() -> None:
    plan = plan_setup(_meta(500), TargetSpec(gpu="L4"))
    assert plan.status == "blocked"
    assert plan.changes == ("It doesn't fit in memory on up to 8 GPUs.",)
    assert {f.rule_id for f in plan.blocking} == {"QPF012"}


def test_the_gpu_count_of_this_machine_caps_the_search() -> None:
    plan = plan_setup(_meta(50), TargetSpec(gpu="L4"), gpu_count=2)
    assert plan.status == "blocked"
    assert plan.changes == ("It doesn't fit in memory on up to 2 GPUs.",)
    assert plan_setup(_meta(50), TargetSpec(gpu="L4"), gpu_count=4).tp == 4


def test_when_each_gpu_count_fails_differently_the_one_asked_for_is_reported() -> None:
    # 9 heads split at no even count, and one GPU is too small, so no rule fails everywhere
    config = {**LLAMA, "num_attention_heads": 9, "num_key_value_heads": 9, "hidden_size": 1152}
    plan = plan_setup(_meta(30, config=config), TargetSpec(gpu="L4"))
    assert plan.status == "blocked"
    assert {f.rule_id for f in plan.blocking} == {"QPF012"}


def test_security_findings_do_not_stop_a_start() -> None:
    meta = _meta(10, container="pytorch_bin", legacy_weight_files=("pytorch_model.bin",))
    report = analyze_checkpoint(meta, TargetSpec(gpu="H100"))
    assert any(f.rule_id == "QPF014" and f.severity.rank == 4 for f in report.findings)
    assert not any(blocks_start(f) for f in report.findings if f.rule_id == "QPF014")
    assert plan_setup(meta, TargetSpec(gpu="H100")).status == "ready"


def test_without_a_gpu_only_the_model_limit_applies() -> None:
    plan = plan_setup(_meta(500), TargetSpec(context_len=100_000))
    assert plan.status == "changed" and plan.max_context is None
    assert plan.context_len == 32768
    assert plan_setup(_meta(500), TargetSpec()).context_len is None


def test_llamacpp_keeps_the_gpu_count_it_was_given() -> None:
    meta = _meta(30, container="gguf", source="org/model-GGUF (model-Q4_K_M.gguf)", shard_files=("model-Q4_K_M.gguf",))
    plan = plan_setup(meta, TargetSpec(engine="llamacpp", gpu="L4"))
    assert plan.status == "blocked" and plan.tp == 1


@pytest.mark.parametrize(
    ("value", "rounded"),
    [(8122, 7168), (4096, 4096), (3000, 2816), (2048, 2048), (100, 256)],
)
def test_contexts_round_down_to_tidy_numbers(value: int, rounded: int) -> None:
    assert _round_down(value) == rounded


def test_no_context_fits_when_the_weights_alone_are_too_big() -> None:
    meta = _meta(100)
    assert _max_fitting_context(meta, _effective_text_config(meta.config), TargetSpec(), 1, 20.0, 32768) == 0


SERVE_CASES = [
    ("vllm", "safetensors", "org/model", (), 4, 8192, 0.9, "vllm serve org/model --tensor-parallel-size 4 --max-model-len 8192"),
    ("vllm", "safetensors", "org/model", (), 1, 32768, 0.85, "vllm serve org/model --gpu-memory-utilization 0.85"),
    ("sglang", "safetensors", "org/model", (), 2, 8192, 0.8,
     "python -m sglang.launch_server --model-path org/model --tp 2 --context-length 8192 --mem-fraction-static 0.8"),
    ("trtllm", "safetensors", "org/model", (), 2, 8192, 0.9, "trtllm-serve org/model --tp_size 2 --max_seq_len 8192"),
    ("llamacpp", "gguf", "org/model-GGUF (m-Q4_K_M.gguf)", ("m-Q4_K_M.gguf",), 1, 32768, 0.9,
     "llama-server --hf-repo org/model-GGUF --hf-file m-Q4_K_M.gguf -c 32768 -ngl 999"),
    ("llamacpp", "gguf", "https://example.com/m.gguf", ("m.gguf",), 1, 4096, 0.9, "llama-server -mu https://example.com/m.gguf -c 4096 -ngl 999"),
    ("llamacpp", "gguf", "./models/m.gguf", ("m.gguf",), 1, None, 0.9, "llama-server -m ./models/m.gguf -ngl 999"),
    ("vllm", "gguf", "org/model-GGUF (m.gguf)", ("m.gguf",), 1, 4096, 0.9, None),
    ("llamacpp", "safetensors", "org/model", (), 1, 4096, 0.9, None),
]


@pytest.mark.parametrize(("engine", "container", "source", "shards", "tp", "context", "util", "command"), SERVE_CASES)
def test_serve_commands(
    engine: str, container: str, source: str, shards: tuple[str, ...], tp: int, context: int | None, util: float, command: str | None
) -> None:
    meta = _meta(1, source=source, container=container, shard_files=shards)
    target = TargetSpec(engine=engine, gpu_memory_utilization=util)
    assert serve_command(meta, target, tp, context, 32768) == command


def test_paths_with_spaces_are_quoted() -> None:
    meta = _meta(1, source="/models/my model")
    assert serve_command(meta, TargetSpec(), 1, None, None) == "vllm serve '/models/my model'"


def test_reports_show_the_setup() -> None:
    meta = _meta(16)
    target = TargetSpec(gpu="L4")
    report = analyze_checkpoint(meta, target)
    report.setup = plan_setup(meta, target)
    text = format_text_report(report)
    assert "Setup: it starts with these changes." in text
    assert f"  Run:  vllm serve org/model --max-model-len {L4_CONTEXT}" in text
    assert json.loads(format_json_report(report))["setup"]["context_len"] == L4_CONTEXT

    blocked = _meta(5, quant=QuantSpec(method="nvfp4", bits=4))
    report = analyze_checkpoint(blocked, TargetSpec(gpu="H100"))
    report.setup = plan_setup(blocked, TargetSpec(gpu="H100"))
    text = format_text_report(report)
    assert "QPF004 needs a different checkpoint, GPU, or engine." in text
    assert "quantpreflight suggest org/model --gpu H100" in text
    assert "Setup" not in format_text_report(analyze_checkpoint(meta, target))


def test_concurrency_is_named_in_the_setup() -> None:
    meta = _meta(10)
    target = TargetSpec(gpu="H100", concurrency=4)
    report = analyze_checkpoint(meta, target)
    report.setup = plan_setup(meta, target)
    assert "tokens for each of 4 requests" in format_text_report(report)
