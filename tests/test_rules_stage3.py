"""Unit and adversarial tests for Stage 3: QPF013, QPF014, and Architecture-Aware KV cache."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from conftest import build_safetensors_header_file, write_checkpoint_dir
from quantpreflight.cli import main
from quantpreflight.loader import load_checkpoint_metadata
from quantpreflight.models import CheckpointMeta, Severity, TargetSpec, TensorSpec
from quantpreflight.rules import (
    RULE_CATALOG,
    analyze_checkpoint,
    estimate_per_gpu_vram_gb,
)


def test_rule_catalog_contains_stage3_rules() -> None:
    assert "QPF013" in RULE_CATALOG
    assert "QPF014" in RULE_CATALOG
    assert RULE_CATALOG["QPF013"][0] == Severity.CRITICAL
    assert RULE_CATALOG["QPF014"][0] == Severity.CRITICAL


def test_qpf013_overlapping_spans_caught() -> None:
    meta = CheckpointMeta(
        source="mock_overlap",
        container_format="safetensors",
        config={"architectures": ["LlamaForCausalLM"]},
        shard_files=("model.safetensors",),
        tensors={
            "t1": TensorSpec(
                name="t1",
                storage_dtype="F16",
                shape=(10, 10),
                shard="model.safetensors",
                byte_size=200,
                effective_bits=16.0,
                data_offsets=(0, 200),
            ),
            "t2": TensorSpec(
                name="t2",
                storage_dtype="F16",
                shape=(10, 10),
                shard="model.safetensors",
                byte_size=200,
                effective_bits=16.0,
                data_offsets=(100, 300),  # Overlaps from 100 to 200!
            ),
        },
    )
    report = analyze_checkpoint(meta)
    findings = [f for f in report.findings if f.rule_id == "QPF013"]
    assert len(findings) == 1
    assert "Overlapping Safetensors memory spans" in findings[0].title
    assert "t1" in findings[0].tensors and "t2" in findings[0].tensors


def test_qpf013_span_length_contradiction() -> None:
    meta = CheckpointMeta(
        source="mock_bad_span",
        container_format="safetensors",
        config={"architectures": ["LlamaForCausalLM"]},
        shard_files=("model.safetensors",),
        tensors={
            "t1": TensorSpec(
                name="t1",
                storage_dtype="F16",  # 2 bytes per element
                shape=(10, 10),       # 100 elements -> 200 bytes expected
                shard="model.safetensors",
                byte_size=150,        # Contradicts 200 bytes!
                effective_bits=16.0,
                data_offsets=(0, 150),
            ),
        },
    )
    report = analyze_checkpoint(meta)
    findings = [f for f in report.findings if f.rule_id == "QPF013"]
    assert any("Tensor span length mismatch" in f.title for f in findings)


def test_qpf013_truncated_file_eof_caught(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "truncated_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}', encoding="utf-8")

    # Header says tensor needs 1000 bytes, but we write header + only 100 bytes of payload
    hdr_bytes = build_safetensors_header_file({"weight": ("F16", [50, 10])})  # needs 1000 bytes payload
    (ckpt_dir / "model.safetensors").write_bytes(hdr_bytes + (b"\x00" * 100))

    meta = load_checkpoint_metadata(str(ckpt_dir))
    report = analyze_checkpoint(meta)
    findings = [f for f in report.findings if f.rule_id == "QPF013"]
    assert any("Tensor data span extends past file EOF" in f.title for f in findings)


def test_qpf014_pure_pickle_directory_caught(tmp_path: Path) -> None:
    pickle_dir = tmp_path / "legacy_pickle_model"
    pickle_dir.mkdir()
    (pickle_dir / "config.json").write_text('{"architectures": ["BertForMaskedLM"]}', encoding="utf-8")
    (pickle_dir / "pytorch_model.bin").write_bytes(b"PK\x03\x04mock_zip_pickle")

    meta = load_checkpoint_metadata(str(pickle_dir))
    assert meta.container_format == "pytorch_bin"
    assert "pytorch_model.bin" in meta.legacy_weight_files

    report = analyze_checkpoint(meta)
    findings = [f for f in report.findings if f.rule_id == "QPF014"]
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL
    assert "Insecure legacy PyTorch pickle checkpoint format" in findings[0].title


def test_qpf014_safetensors_with_lingering_pickle_file(tmp_path: Path) -> None:
    mixed_dir = tmp_path / "mixed_model"
    mixed_dir.mkdir()
    (mixed_dir / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}', encoding="utf-8")
    (mixed_dir / "model.safetensors").write_bytes(build_safetensors_header_file({"embed": ("F16", [10, 10])}))
    (mixed_dir / "pytorch_model.bin").write_bytes(b"PK\x03\x04mock_zip_pickle")

    meta = load_checkpoint_metadata(str(mixed_dir))
    assert meta.container_format == "safetensors"
    assert "pytorch_model.bin" in meta.legacy_weight_files

    report = analyze_checkpoint(meta)
    findings = [f for f in report.findings if f.rule_id == "QPF014"]
    assert len(findings) == 1
    assert findings[0].severity == Severity.HIGH
    assert "Redundant legacy pickle weight files" in findings[0].title


def test_qpf014_auto_map_custom_code_warning(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_code_model"
    custom_dir.mkdir()
    cfg = {
        "architectures": ["CustomCausalLM"],
        "auto_map": {
            "AutoModelForCausalLM": "modeling_custom.CustomCausalLM"
        }
    }
    (custom_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (custom_dir / "model.safetensors").write_bytes(build_safetensors_header_file({"weight": ("F16", [10, 10])}))
    (custom_dir / "modeling_custom.py").write_text("# Untrusted custom code\nprint('hello')\n", encoding="utf-8")

    meta = load_checkpoint_metadata(str(custom_dir))
    assert "modeling_custom.py" in meta.custom_code_files

    report = analyze_checkpoint(meta)
    findings = [f for f in report.findings if f.rule_id == "QPF014"]
    assert any("Unsandboxed custom Python modeling code required" in f.title for f in findings)


def test_deepseek_mla_kv_cache_calculation() -> None:
    # DeepSeek-V2 MLA parameters
    tcfg_mla = {
        "model_type": "deepseek_v2",
        "num_hidden_layers": 60,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "hidden_size": 2048,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    meta = CheckpointMeta(
        source="deepseek_mla",
        container_format="safetensors",
        total_weight_bytes=10 * (1024**3),  # 10 GiB weights
    )
    target = TargetSpec(tp=1, context_len=4096)
    total_mla, weights_mla, kv_mla = estimate_per_gpu_vram_gb(meta, tcfg_mla, target)

    # Standard GQA comparison (without MLA)
    tcfg_gqa = dict(tcfg_mla)
    del tcfg_gqa["kv_lora_rank"]
    total_gqa, weights_gqa, kv_gqa = estimate_per_gpu_vram_gb(meta, tcfg_gqa, target)

    # In MLA: 60 layers * (512 + 64) * 2 bytes * 4096 tokens = ~270 MiB (0.26 GiB)
    # In GQA: 2 * 60 layers * 128 heads * 128 head_dim * 2 bytes * 4096 tokens = ~15 GiB!
    assert kv_mla < 0.5, f"Expected MLA KV cache ~0.26 GiB, got {kv_mla} GiB"
    assert kv_gqa > 14.0, f"Expected standard GQA KV cache ~15 GiB, got {kv_gqa} GiB"
    assert total_mla < total_gqa


def test_sliding_window_kv_cache_calculation() -> None:
    # Mistral with 32k context and 4096 sliding window
    tcfg_mistral = {
        "model_type": "mistral",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "sliding_window": 4096,
        "max_position_embeddings": 32768,
    }
    meta = CheckpointMeta(
        source="mistral_swa",
        container_format="safetensors",
        total_weight_bytes=7 * (1024**3),
    )
    # Target context is 32768, but sliding window bounds it to 4096
    target_32k = TargetSpec(tp=1, context_len=32768)
    _, _, kv_with_window = estimate_per_gpu_vram_gb(meta, tcfg_mistral, target_32k)

    # Same config without sliding window
    tcfg_no_window = dict(tcfg_mistral)
    del tcfg_no_window["sliding_window"]
    _, _, kv_no_window = estimate_per_gpu_vram_gb(meta, tcfg_no_window, target_32k)

    # KV with sliding window should be exactly 8x smaller (4096 vs 32768)
    assert kv_with_window < 0.6
    assert kv_no_window > 3.8
    assert kv_with_window * 7 <= kv_no_window


def test_gemma2_alternating_sliding_window() -> None:
    tcfg_gemma2 = {
        "model_type": "gemma2",
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
        "head_dim": 256,
        "sliding_window": 4096,
    }
    meta = CheckpointMeta(source="gemma2", container_format="safetensors", total_weight_bytes=5 * (1024**3))
    target = TargetSpec(tp=1, context_len=16384)
    _, _, kv_gemma = estimate_per_gpu_vram_gb(meta, tcfg_gemma2, target)

    # Half layers (13) use 4096, half layers (13) use 16384
    # Expected weighted context: (13 * 4096 + 13 * 16384) / 26 = 10240
    tcfg_full = dict(tcfg_gemma2)
    del tcfg_full["sliding_window"]
    _, _, kv_full = estimate_per_gpu_vram_gb(meta, tcfg_full, target)

    assert kv_gemma < kv_full
    assert round(kv_gemma, 1) == round(kv_full * (10240 / 16384), 1)


def test_mamba_and_jamba_hybrid_memory_estimation() -> None:
    # Pure Mamba: 0 attention layers
    tcfg_mamba = {
        "model_type": "mamba",
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "state_size": 16,
    }
    meta = CheckpointMeta(source="mamba", container_format="safetensors", total_weight_bytes=3 * (1024**3))
    target = TargetSpec(tp=1, context_len=8192)
    _, _, kv_mamba = estimate_per_gpu_vram_gb(meta, tcfg_mamba, target)
    assert kv_mamba <= 0.02  # Fixed tiny SSM state buffer (~6 MB)

    # Jamba hybrid: only 4 of 32 layers are attention
    tcfg_jamba = {
        "model_type": "jamba",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "attn_layer_period": 8,
    }
    _, _, kv_jamba = estimate_per_gpu_vram_gb(meta, tcfg_jamba, target)
    tcfg_all_attn = dict(tcfg_jamba)
    tcfg_all_attn["model_type"] = "llama"
    del tcfg_all_attn["attn_layer_period"]
    _, _, kv_all_attn = estimate_per_gpu_vram_gb(meta, tcfg_all_attn, target)

    # Jamba attention memory should be exactly 1/8th of all-attention model
    assert kv_jamba * 7 <= kv_all_attn


def test_cli_list_rules_includes_stage3(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list-rules"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "QPF013" in out
    assert "QPF014" in out
