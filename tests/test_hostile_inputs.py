"""A checkpoint is untrusted input: every malformed config value must end in a report or a clean exit 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import write_checkpoint_dir
from quantpreflight.cli import SCAN_ERRORS, main
from quantpreflight.loader import _extract_tokenizer_meta
from quantpreflight.models import CheckpointMeta, TargetSpec
from quantpreflight.quant_detect import build_safetensors_specs, parse_quant_spec_from_config
from quantpreflight.rules import analyze_checkpoint
from quantpreflight.safetensors_io import RawSafetensorEntry

HOSTILE_VALUES: list[Any] = [None, True, -1, 0, 1.5, "x", "8", [], [1], ["x"], {}, {"a": 1}, 10**400]

CONFIG_KEYS = [
    "architectures",
    "model_type",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
    "quantization_config",
    "text_config",
    "auto_map",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "state_size",
    "d_state",
    "intermediate_size",
    "layers_block_type",
    "attn_layer_period",
    "sliding_window",
    "sliding_window_size",
    "layer_types",
    "sliding_window_pattern",
    "num_local_experts",
    "n_routed_experts",
    "num_experts",
    "torch_dtype",
    "dtype",
]

QUANT_METHODS = [
    "awq",
    "gptq",
    "auto-round",
    "fp8",
    "nvfp4",
    "mxfp4",
    "modelopt",
    "bitsandbytes",
    "bitnet",
    "compressed-tensors",
    "something-new",
]

QUANT_KEYS = [
    "bits",
    "w_bit",
    "group_size",
    "q_group_size",
    "sym",
    "desc_act",
    "zero_point",
    "version",
    "checkpoint_format",
    "weight_block_size",
    "activation_scheme",
    "modules_to_not_convert",
    "exclude_modules",
    "ignored_layers",
    "ignore",
    "quantization",
    "quant_algo",
    "producer",
    "config_groups",
    "format",
    "load_in_4bit",
    "load_in_8bit",
    "bnb_4bit_quant_type",
    "packing_format",
]

CONFIG_GROUP_VALUES: list[Any] = [
    {"g": None},
    {"g": {"weights": None}},
    {"g": {"weights": {"num_bits": "x", "group_size": "y", "block_structure": [0, 0]}}},
    {"g": {"weights": {"num_bits": 0}, "input_activations": {"num_bits": []}}},
    {"g": {"input_activations": {"num_bits": 8, "type": 5}}},
]

BLOCK_SIZES: list[Any] = [[0, 0], [0, 128], [128, 0], [-1, -1], [1, 1], ["a", "b"], [128], [1, 2, 3]]

SIDE_FILES = ["generation_config.json", "tokenizer_config.json", "tokenizer.json"]

TOKENIZER_VALUES: list[Any] = [
    {"model": None},
    {"model": {"vocab": None}},
    {"model": {"vocab": {"a": "x", "b": None, "c": 1}}},
    {"added_tokens": None},
    {"added_tokens": [None, 1, {"id": "x"}, {"id": None}, {"id": 5}]},
    {"added_tokens_decoder": None},
    {"added_tokens_decoder": {"x": {}, "1": None, "2": "y"}},
    {"eos_token": None},
    {"eos_token": 5},
    {"eos_token": {"content": 5}},
    {"eos_token_id": ["x", None, 1.5, True]},
]


BASE_CONFIG: dict[str, Any] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 64,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "max_position_embeddings": 128,
    "eos_token_id": 2,
}

TARGETS = [
    TargetSpec(gpu="H100", tp=2),
    TargetSpec(gpu="T4", engine="sglang", tp=3),
    TargetSpec(gpu="GB10", marlin_input_dtype="fp8"),
    TargetSpec(engine="llamacpp", vram_gb_override=1.0),
]


def _raw_tensors() -> dict[str, tuple[RawSafetensorEntry, str]]:
    shapes = {
        "model.embed_tokens.weight": ("F16", (64, 64)),
        "model.layers.0.self_attn.q_proj.weight": ("F8_E4M3", (64, 64)),
        "model.layers.0.self_attn.o_proj.qweight": ("I32", (64, 16)),
        "model.layers.0.self_attn.k_proj.weight_packed": ("U8", (64, 32)),
        "model.layers.0.self_attn.v_proj.weight": ("U8", (64, 32)),
        "lm_head.weight": ("F16", (64, 64)),
    }
    raw = {}
    offset = 0
    for name, (dtype, shape) in shapes.items():
        entry = RawSafetensorEntry(name, dtype, shape, (offset, offset + 8192), 8192)
        raw[name] = (entry, "model.safetensors")
        offset += 8192
    return raw


RAW_TENSORS = _raw_tensors()


def _analyze(config: dict[str, Any], **sections: Any) -> None:
    """Run the same steps as a scan. Only errors the CLI turns into exit 2 may escape."""
    try:
        quant = parse_quant_spec_from_config(config)
        specs = build_safetensors_specs(RAW_TENSORS, quant)
        tokenizer = sections.get("tokenizer_config", {})
        vocab = config.get("vocab_size")
        meta = CheckpointMeta(
            source="mock",
            container_format="safetensors",
            config=config,
            generation_config=sections.get("generation_config", {}),
            tokenizer_config=tokenizer,
            tokenizer_meta=_extract_tokenizer_meta(sections.get("tokenizer_json"), tokenizer, config_vocab_size=vocab),
            tensors=specs,
            shard_files=("model.safetensors",),
            total_weight_bytes=sum(s.byte_size for s in specs.values()),
            quant=quant,
        )
        for target in TARGETS:
            analyze_checkpoint(meta, target)
    except SCAN_ERRORS:
        pass


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
@pytest.mark.parametrize("key", CONFIG_KEYS)
def test_hostile_config_value(key: str, value: Any) -> None:
    _analyze({**BASE_CONFIG, key: value})


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
@pytest.mark.parametrize("key", ["layer_types", "sliding_window_pattern"])
def test_hostile_sliding_attention_metadata(key: str, value: Any) -> None:
    _analyze({**BASE_CONFIG, "sliding_window": 16, key: value})


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
@pytest.mark.parametrize("key", CONFIG_KEYS)
def test_hostile_text_config_value(key: str, value: Any) -> None:
    _analyze({**BASE_CONFIG, "model_type": "llava", "text_config": {**BASE_CONFIG, key: value}})


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
@pytest.mark.parametrize("key", CONFIG_KEYS)
def test_hostile_value_in_mamba_config(key: str, value: Any) -> None:
    _analyze({**BASE_CONFIG, "model_type": "mamba2", key: value})


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
@pytest.mark.parametrize("key", CONFIG_KEYS)
def test_hostile_value_in_latent_cache_config(key: str, value: Any) -> None:
    _analyze({**BASE_CONFIG, "kv_lora_rank": 512, "qk_rope_head_dim": 64, key: value})


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
@pytest.mark.parametrize("key", QUANT_KEYS)
@pytest.mark.parametrize("method", QUANT_METHODS)
def test_hostile_quantization_value(method: str, key: str, value: Any) -> None:
    _analyze({**BASE_CONFIG, "quantization_config": {"quant_method": method, "bits": 4, key: value}})


@pytest.mark.parametrize("groups", CONFIG_GROUP_VALUES, ids=repr)
def test_hostile_config_groups(groups: Any) -> None:
    qcfg = {"quant_method": "compressed-tensors", "config_groups": groups}
    _analyze({**BASE_CONFIG, "quantization_config": qcfg})


@pytest.mark.parametrize("block", BLOCK_SIZES, ids=repr)
@pytest.mark.parametrize("method", ["fp8", "compressed-tensors"])
def test_hostile_block_sizes(method: str, block: Any) -> None:
    weights = {"num_bits": 8, "block_structure": block}
    qcfg = {"quant_method": method, "weight_block_size": block, "config_groups": {"g": {"weights": weights}}}
    _analyze({**BASE_CONFIG, "quantization_config": qcfg})


@pytest.mark.parametrize("section", ["generation_config", "tokenizer_config", "tokenizer_json"])
@pytest.mark.parametrize("payload", TOKENIZER_VALUES, ids=repr)
def test_hostile_side_file(section: str, payload: dict[str, Any]) -> None:
    _analyze(BASE_CONFIG, **{section: payload})


def test_a_folder_named_config_json_ends_in_a_clean_exit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ckpt = write_checkpoint_dir(tmp_path / "ckpt", config={}, tensors={"a.weight": ("F16", [2, 2])})
    (ckpt / "config.json").unlink()
    (ckpt / "config.json").mkdir()

    assert main(["check", str(ckpt / "model.safetensors")]) in (0, 1, 2)
    capsys.readouterr()


@pytest.mark.parametrize("name", ["config.json", *SIDE_FILES, "model.safetensors.index.json"])
@pytest.mark.parametrize("text", ["", "{", "[]", "null", "5", '"x"', "﻿{}"], ids=repr)
def test_broken_json_file(tmp_path: Path, name: str, text: str) -> None:
    ckpt = write_checkpoint_dir(tmp_path / "ckpt", config={}, tensors={"a.weight": ("F16", [2, 2])})
    (ckpt / name).write_text(text, encoding="utf-8")
    assert main(["check", str(ckpt)]) in (0, 1, 2)


@pytest.mark.parametrize("name", ["config.json", *SIDE_FILES, "model.safetensors.index.json"])
def test_non_utf8_json_file(tmp_path: Path, name: str) -> None:
    ckpt = write_checkpoint_dir(tmp_path / "ckpt", config={}, tensors={"a.weight": ("F16", [2, 2])})
    (ckpt / name).write_bytes(b"\xff\xfe\x00{}")
    assert main(["check", str(ckpt)]) in (0, 1, 2)


@pytest.mark.parametrize(
    "weight_map",
    [None, {}, [], {"": "model.safetensors"}, {"a": None}, {"a": 5}, {"a": "../x.safetensors"}, {"a": "x.bin"}],
    ids=repr,
)
def test_hostile_index_weight_map(tmp_path: Path, weight_map: Any) -> None:
    ckpt = write_checkpoint_dir(tmp_path / "ckpt", config={}, tensors={"a.weight": ("F16", [2, 2])})
    index = {"weight_map": weight_map, "metadata": {"total_size": "x"}}
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    assert main(["check", str(ckpt)]) in (0, 1, 2)
