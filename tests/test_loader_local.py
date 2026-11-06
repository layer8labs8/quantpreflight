"""Loading local folders and files, and reading tokenizer metadata."""

from __future__ import annotations

import json
import pickle
import os
from pathlib import Path
from typing import Any

import pytest

from conftest import build_gguf_header_bytes, build_safetensors_header_file, write_checkpoint_dir
from quantpreflight.loader import (
    CheckpointLoadError,
    _extract_tokenizer_meta,
    _parse_hf_target,
    _resolve_hf_token,
    load_checkpoint_metadata,
)
from quantpreflight.rules import analyze_checkpoint
from quantpreflight.safetensors_io import SafetensorsFormatError

TENSORS = {"model.embed_tokens.weight": ("F16", [8, 8]), "lm_head.weight": ("F16", [8, 8])}


def _ckpt(root: Path, **kwargs: Any) -> Path:
    return write_checkpoint_dir(root / "ckpt", config=kwargs.pop("config", {"vocab_size": 8}), tensors=TENSORS, **kwargs)


def _shard(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    path.write_bytes(build_safetensors_header_file(tensors))


def _index(ckpt: Path, weight_map: dict[str, str], total: int | None = None) -> None:
    index: dict[str, Any] = {"weight_map": weight_map}
    if total is not None:
        index["metadata"] = {"total_size": total}
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


# Choosing what to read


def test_folder_with_a_single_safetensors_file(tmp_path: Path) -> None:
    meta = load_checkpoint_metadata(str(_ckpt(tmp_path)))
    assert meta.container_format == "safetensors"
    assert meta.shard_files == ("model.safetensors",)
    assert set(meta.tensors) == set(TENSORS)
    assert meta.total_weight_bytes == 256
    assert meta.source == str(tmp_path / "ckpt")


def test_header_sizes_are_recorded(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path)
    meta = load_checkpoint_metadata(str(ckpt))
    size = (ckpt / "model.safetensors").stat().st_size
    assert meta.shard_file_sizes == {"model.safetensors": size}
    assert meta.shard_header_lengths == {"model.safetensors": size}
    assert meta.header_bytes_fetched == size


@pytest.mark.parametrize("name", ["config.json", "model.safetensors.index.json"])
def test_pointing_at_a_metadata_file_scans_its_folder(tmp_path: Path, name: str) -> None:
    ckpt = _ckpt(tmp_path)
    if name != "config.json":
        _index(ckpt, {key: "model.safetensors" for key in TENSORS})
    meta = load_checkpoint_metadata(str(ckpt / name))
    assert meta.source == str(ckpt)
    assert set(meta.tensors) == set(TENSORS)


def test_pointing_at_a_safetensors_file_reads_the_config_next_to_it(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path, config={"model_type": "llama", "quantization_config": {"quant_method": "awq", "bits": 4}})
    meta = load_checkpoint_metadata(str(ckpt / "model.safetensors"))
    assert meta.config["model_type"] == "llama"
    assert meta.quant.method == "awq"
    assert meta.shard_files == ("model.safetensors",)
    assert meta.total_weight_bytes == 256


def test_a_safetensors_file_alone_has_an_empty_config(tmp_path: Path) -> None:
    path = tmp_path / "solo.safetensors"
    _shard(path, TENSORS)
    meta = load_checkpoint_metadata(str(path))
    assert meta.config == {}
    assert meta.quant.method == "none"
    assert meta.shard_file_sizes == {"solo.safetensors": path.stat().st_size}
    assert meta.safetensors_metadata == {}


def test_uppercase_extensions_are_recognised(tmp_path: Path) -> None:
    path = tmp_path / "SOLO.SAFETENSORS"
    _shard(path, TENSORS)
    assert load_checkpoint_metadata(str(path)).tensors

    gguf = tmp_path / "SOLO.GGUF"
    gguf.write_bytes(build_gguf_header_bytes())
    assert load_checkpoint_metadata(str(gguf)).container_format == "gguf"


def test_unnamed_safetensors_files_are_found_in_a_folder(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _shard(ckpt / "b.safetensors", {"b.weight": ("F16", [2, 2])})
    _shard(ckpt / "a.safetensors", {"a.weight": ("F16", [2, 2])})
    assert load_checkpoint_metadata(str(ckpt)).shard_files == ("a.safetensors", "b.safetensors")


def test_gguf_folder_uses_the_first_file_by_name(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "b.gguf").write_bytes(build_gguf_header_bytes(kv_strings={"general.architecture": "b"}))
    (ckpt / "a.gguf").write_bytes(build_gguf_header_bytes(kv_strings={"general.architecture": "a"}))
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.config["model_type"] == "a"
    assert meta.shard_files == ("a.gguf",)


def test_gguf_file_metadata_reaches_the_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(build_gguf_header_bytes(tokens=["a", "b", "c"], tensors=[("t", (32,), 0)]))
    meta = load_checkpoint_metadata(str(path))
    assert meta.tokenizer_meta.max_token_id == 2
    assert meta.tokenizer_meta.max_token_str == "c"
    assert meta.total_weight_bytes == 128
    assert meta.header_bytes_fetched == path.stat().st_size


def test_safetensors_win_over_gguf_in_the_same_folder(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / "m.gguf").write_bytes(build_gguf_header_bytes())
    assert load_checkpoint_metadata(str(ckpt)).container_format == "safetensors"


def test_gguf_next_to_an_index_is_not_chosen(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "m.gguf").write_bytes(build_gguf_header_bytes())
    _index(ckpt, {"a": "missing.safetensors"})
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.container_format == "safetensors"
    assert meta.missing_shards == ("missing.safetensors",)


@pytest.mark.parametrize(
    ("setup", "message"),
    [("empty", "No .safetensors or .gguf"), ("text", "Unsupported checkpoint file"), ("missing", "neither an existing")],
)
def test_nothing_to_scan(tmp_path: Path, setup: str, message: str) -> None:
    target = tmp_path / "x"
    if setup == "empty":
        target.mkdir()
    elif setup == "text":
        target = tmp_path / "notes.txt"
        target.write_text("x", encoding="utf-8")
    else:
        target = tmp_path / "a" / "b" / "c"
    with pytest.raises(CheckpointLoadError, match=message):
        load_checkpoint_metadata(str(target))


# Shards


def test_sharded_folder_merges_tensors_in_index_order(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _shard(ckpt / "s2.safetensors", {"b.weight": ("F16", [2, 2])})
    _shard(ckpt / "s1.safetensors", {"a.weight": ("F16", [2, 2])})
    _index(ckpt, {"b.weight": "s2.safetensors", "a.weight": "s1.safetensors"})
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.shard_files == ("s1.safetensors", "s2.safetensors")
    assert meta.tensors["b.weight"].shard == "s2.safetensors"
    assert meta.missing_shards == ()
    assert meta.dangling_index_tensors == ()
    assert meta.header_bytes_fetched == sum(meta.shard_header_lengths.values())


def test_index_size_only_applies_when_shards_are_missing(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _shard(ckpt / "s1.safetensors", {"a.weight": ("F16", [2, 2])})

    _index(ckpt, {"a.weight": "s1.safetensors"}, total=9999)
    assert load_checkpoint_metadata(str(ckpt)).total_weight_bytes == 8

    _index(ckpt, {"a.weight": "s1.safetensors", "b.weight": "s2.safetensors"}, total=9999)
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.total_weight_bytes == 9999
    assert meta.missing_shards == ("s2.safetensors",)
    assert meta.dangling_index_tensors == ()


def test_missing_shards_without_a_total_fall_back_to_what_was_read(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _shard(ckpt / "s1.safetensors", {"a.weight": ("F16", [2, 2])})
    _index(ckpt, {"a.weight": "s1.safetensors", "b.weight": "s2.safetensors"})
    assert load_checkpoint_metadata(str(ckpt)).total_weight_bytes == 8


def test_duplicates_and_dangling_entries(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _shard(ckpt / "s1.safetensors", {"dup.weight": ("F16", [2, 2]), "x.weight": ("F16", [2, 2])})
    _shard(ckpt / "s2.safetensors", {"dup.weight": ("F16", [2, 2])})
    _index(ckpt, {"dup.weight": "s2.safetensors", "x.weight": "s1.safetensors", "ghost.weight": "s1.safetensors"})
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.duplicate_tensors == ("dup.weight",)
    assert meta.dangling_index_tensors == ("ghost.weight",)
    assert meta.tensors["dup.weight"].shard == "s2.safetensors"


def test_shard_metadata_is_merged(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "s1.safetensors").write_bytes(build_safetensors_header_file({"a": ("F16", [1])}, metadata={"one": "1"}))
    (ckpt / "s2.safetensors").write_bytes(build_safetensors_header_file({"b": ("F16", [1])}, metadata={"two": "2"}))
    assert load_checkpoint_metadata(str(ckpt)).safetensors_metadata == {"one": "1", "two": "2"}


def test_index_cannot_escape_the_folder(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path)
    _index(ckpt, {"a": "../outside.safetensors"})
    with pytest.raises(SafetensorsFormatError, match="traversal"):
        load_checkpoint_metadata(str(ckpt))


def test_symlink_to_a_folder_sharing_the_prefix_is_rejected(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    sibling = tmp_path / "ckpt-evil"
    ckpt.mkdir()
    sibling.mkdir()
    _shard(sibling / "s.safetensors", {"a": ("F16", [1])})
    try:
        os.symlink(sibling, ckpt / "link", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks needs extra rights on this machine")
    _index(ckpt, {"a": "link/s.safetensors"})
    with pytest.raises(SafetensorsFormatError, match="escapes"):
        load_checkpoint_metadata(str(ckpt))


# Side files


@pytest.mark.parametrize(("name", "qcfg"), [("hf_quant_config.json", {"quant_algo": "FP8"}), ("quantize_config.json", {"bits": 4, "group_size": 128})])
def test_external_quant_configs_fill_a_gap(tmp_path: Path, name: str, qcfg: dict[str, Any]) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / name).write_text(json.dumps(qcfg), encoding="utf-8")
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.config["quantization_config"] == qcfg


def test_hf_quant_config_wins_over_quantize_config(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / "hf_quant_config.json").write_text('{"quant_algo": "FP8"}', encoding="utf-8")
    (ckpt / "quantize_config.json").write_text('{"bits": 4, "group_size": 128}', encoding="utf-8")
    assert load_checkpoint_metadata(str(ckpt)).quant.method == "fp8"


def test_config_json_wins_over_external_quant_configs(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path, config={"quantization_config": {"quant_method": "awq", "bits": 4}})
    (ckpt / "hf_quant_config.json").write_text('{"quant_algo": "FP8"}', encoding="utf-8")
    assert load_checkpoint_metadata(str(ckpt)).quant.method == "awq"


def test_generation_and_tokenizer_files_are_loaded(tmp_path: Path) -> None:
    ckpt = _ckpt(
        tmp_path,
        generation_config={"eos_token_id": 3},
        tokenizer_config={"eos_token": "<e>"},
        tokenizer_json={"added_tokens": [{"id": 7, "content": "<e>"}]},
    )
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.generation_config == {"eos_token_id": 3}
    assert meta.tokenizer_config == {"eos_token": "<e>"}
    assert meta.tokenizer_meta.eos_token_ids == (7,)


def test_invalid_json_names_the_file(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / "generation_config.json").write_text("{", encoding="utf-8")
    with pytest.raises(CheckpointLoadError, match="generation_config.json"):
        load_checkpoint_metadata(str(ckpt))


# torch.save zip / classic pickle protocol magic (issue #6)
_PICKLE_ZIP = b"PK\x03\x04rest"
_PICKLE_PROTO = b"\x80\x02abc"
_PICKLE_PROTO0 = pickle.dumps({"data": [1, 2]}, protocol=0)


@pytest.mark.parametrize(
    ("name", "legacy"),
    [
        ("pytorch_model.bin", True),
        ("pytorch_model-00001-of-00002.bin", True),
        ("model.pt", True),
        ("weights.pth", True),
        ("consolidated.00.pth", True),
        ("MODEL.BIN", True),
        ("training_args.bin", False),
        ("optimizer.pt", False),
        ("model.txt", False),
    ],
)
def test_legacy_weight_names(tmp_path: Path, name: str, legacy: bool) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / name).write_bytes(_PICKLE_PROTO)
    assert (name in load_checkpoint_metadata(str(ckpt)).legacy_weight_files) is legacy


def test_python_files_are_listed_as_custom_code(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / "modeling_x.py").write_text("x", encoding="utf-8")
    (ckpt / "UPPER.PY").write_text("x", encoding="utf-8")
    assert load_checkpoint_metadata(str(ckpt)).custom_code_files == ("UPPER.PY", "modeling_x.py")


def test_pickle_only_folder(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"vocab_size": 8, "quantization_config": {"quant_method": "gptq", "bits": 4}}', encoding="utf-8")
    (ckpt / "b_model.bin").write_bytes(_PICKLE_ZIP)
    (ckpt / "a_model.bin").write_bytes(_PICKLE_PROTO)
    (ckpt / "run.py").write_text("x", encoding="utf-8")
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.container_format == "pytorch_bin"
    assert meta.shard_files == ("a_model.bin", "b_model.bin")
    assert meta.legacy_weight_files == ("a_model.bin", "b_model.bin")
    assert meta.custom_code_files == ("run.py",)
    assert meta.quant.method == "gptq"
    assert meta.tokenizer_meta.max_token_id == 7
    assert {f.rule_id for f in analyze_checkpoint(meta).findings} >= {"QPF014"}


def test_metal_model_bin_is_not_treated_as_pickle(tmp_path: Path) -> None:
    """gpt-oss Metal weights start with 'GPT-OSS …', not torch pickle magic (#6)."""
    ckpt = _ckpt(tmp_path)
    (ckpt / "metal_model.bin").write_bytes(b"GPT-OSS v1.0\x00weights")
    (ckpt / "pytorch_model.bin").write_bytes(_PICKLE_ZIP)
    meta = load_checkpoint_metadata(str(ckpt))
    assert meta.legacy_weight_files == ("pytorch_model.bin",)
    assert "metal_model.bin" not in meta.legacy_weight_files


def test_protocol_0_pickle_is_treated_as_pickle(tmp_path: Path) -> None:
    """Pickle protocol 0 starts with ASCII opcode bytes (issue #6), must still be flagged."""
    ckpt = _ckpt(tmp_path)
    (ckpt / "metal_model.bin").write_bytes(b"GPT-OSS v1.0\x00weights")
    (ckpt / "zip_model.bin").write_bytes(_PICKLE_ZIP)
    (ckpt / "pytorch_model.bin").write_bytes(_PICKLE_PROTO0)
    meta = load_checkpoint_metadata(str(ckpt))
    assert set(meta.legacy_weight_files) == {"pytorch_model.bin", "zip_model.bin"}
    assert "metal_model.bin" not in meta.legacy_weight_files


def test_added_tokens_decoder_reaches_the_vocab_rule(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path, tokenizer_config={"added_tokens_decoder": {"8": {"content": "<x>"}}})
    report = analyze_checkpoint(load_checkpoint_metadata(str(ckpt)))
    assert any(f.rule_id == "QPF002" and "<x>" in f.message for f in report.findings)


# Tokenizer metadata


def test_tokenizer_defaults_are_empty() -> None:
    meta = _extract_tokenizer_meta(None, {})
    assert (meta.vocab_max_id, meta.max_token_id, meta.max_token_str, meta.total_tokens) == (None, None, None, None)
    assert (meta.eos_token, meta.eos_token_ids) == (None, ())


@pytest.mark.parametrize(("vocab", "max_id"), [(10, 9), (1, 0), (0, None), (-1, None), ("x", None)])
def test_tokenizer_config_vocab_size(vocab: Any, max_id: int | None) -> None:
    meta = _extract_tokenizer_meta(None, {}, config_vocab_size=vocab)
    assert meta.max_token_id == max_id
    assert meta.vocab_max_id == max_id
    assert meta.total_tokens == (None if max_id is None else max_id + 1)
    if max_id is not None:
        assert meta.max_token_str == f"<vocab_size:{vocab}>"


def test_tokenizer_vocab_table_can_raise_the_bound() -> None:
    meta = _extract_tokenizer_meta({"model": {"vocab": {"a": 1, "b": 20, "c": 3}}}, {}, config_vocab_size=10)
    assert (meta.vocab_max_id, meta.max_token_id, meta.max_token_str) == (20, 20, "b")


def test_tokenizer_vocab_table_below_the_config_keeps_the_config_bound() -> None:
    meta = _extract_tokenizer_meta({"model": {"vocab": {"a": 1}}}, {}, config_vocab_size=10)
    assert meta.max_token_id == 9


def test_added_tokens_raise_only_the_max_id() -> None:
    tokenizer = {"model": {"vocab": {"a": 1}}, "added_tokens": [{"id": 50, "content": "<x>"}, {"id": 40}]}
    meta = _extract_tokenizer_meta(tokenizer, {})
    assert (meta.vocab_max_id, meta.max_token_id, meta.max_token_str) == (1, 50, "<x>")


def test_added_token_without_content_gets_a_label() -> None:
    assert _extract_tokenizer_meta({"added_tokens": [{"id": 5}]}, {}).max_token_str == "<id:5>"


def test_added_tokens_decoder_raises_the_max_id() -> None:
    config = {"added_tokens_decoder": {"60": {"content": "<y>"}, "x": {}, "2": "plain"}}
    meta = _extract_tokenizer_meta({"added_tokens": [{"id": 50, "content": "<x>"}]}, config)
    assert (meta.max_token_id, meta.max_token_str, meta.total_tokens) == (60, "<y>", 61)


def test_added_tokens_decoder_entry_without_content() -> None:
    assert _extract_tokenizer_meta(None, {"added_tokens_decoder": {"3": "plain"}}).max_token_str == "<id:3>"


@pytest.mark.parametrize("eos", ["<e>", {"content": "<e>", "lstrip": False}])
def test_eos_token_id_lookup(eos: Any) -> None:
    meta = _extract_tokenizer_meta({"model": {"vocab": {"<e>": 2}}}, {"eos_token": eos})
    assert meta.eos_token == "<e>"
    assert meta.eos_token_ids == (2,)


def test_unknown_eos_token_has_no_id() -> None:
    meta = _extract_tokenizer_meta({"model": {"vocab": {"a": 2}}}, {"eos_token": "<e>"})
    assert meta.eos_token == "<e>"
    assert meta.eos_token_ids == ()


def test_decoder_does_not_override_a_token_from_tokenizer_json() -> None:
    tokenizer = {"added_tokens": [{"id": 2, "content": "<e>"}]}
    config = {"eos_token": "<e>", "added_tokens_decoder": {"9": {"content": "<e>"}}}
    assert _extract_tokenizer_meta(tokenizer, config).eos_token_ids == (2,)


def test_gguf_token_table_replaces_the_bounds() -> None:
    table = {"__gguf_string_table_len__": 5, "last": "<z>"}
    meta = _extract_tokenizer_meta(None, {}, {"tokenizer.ggml.tokens": table, "tokenizer.ggml.eos_token_id": 2})
    assert (meta.vocab_max_id, meta.max_token_id, meta.max_token_str, meta.eos_token_ids) == (4, 4, "<z>", (2,))


def test_gguf_token_table_without_a_last_token_gets_a_label() -> None:
    meta = _extract_tokenizer_meta(None, {}, {"tokenizer.ggml.tokens": {"__gguf_string_table_len__": 5}})
    assert meta.max_token_str == "<id:4>"


def test_empty_gguf_token_table_is_ignored() -> None:
    assert _extract_tokenizer_meta(None, {}, {"tokenizer.ggml.tokens": {"__gguf_string_table_len__": 0}}).max_token_id is None


def test_eos_ids_are_unique() -> None:
    meta = _extract_tokenizer_meta({"added_tokens": [{"id": 2, "content": "<e>"}]}, {"eos_token": "<e>"}, {"tokenizer.ggml.eos_token_id": 2})
    assert meta.eos_token_ids == (2,)


# Targets and tokens


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("org/model", ("https://huggingface.co/org/model/resolve/rev", None)),
        ("/org/model/", ("https://huggingface.co/org/model/resolve/rev", None)),
        ("https://huggingface.co/org/model", ("https://huggingface.co/org/model/resolve/rev", None)),
        ("https://hf.co/org/model", ("https://huggingface.co/org/model/resolve/rev", None)),
        ("https://HuggingFace.co/org/model/tree/main", ("https://huggingface.co/org/model/resolve/rev", None)),
        ("https://huggingface.co/org/model/resolve/v1/m.gguf", ("https://huggingface.co/org/model/resolve/v1", "m.gguf")),
        ("https://huggingface.co/org/model/blob/main/sub/m.gguf", ("https://huggingface.co/org/model/resolve/main", "sub/m.gguf")),
        ("https://example.com/org/model", None),
        ("https://huggingface.co/org", None),
        ("org", None),
        ("a/b/c", None),
        ("org//", None),
        ("C:/model", None),
        ("org\\model", None),
    ],
)
def test_parse_hf_target(target: str, expected: Any) -> None:
    assert _parse_hf_target(target, "rev") == expected


def test_token_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert _resolve_hf_token() is None

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", " second ")
    assert _resolve_hf_token() == "second"

    monkeypatch.setenv("HF_TOKEN", " first ")
    assert _resolve_hf_token() == "first"
    assert _resolve_hf_token(" flag ") == "flag"

    monkeypatch.setenv("HF_TOKEN", "   ")
    assert _resolve_hf_token() == "second"


@pytest.mark.parametrize(
    ("tokenizer", "config", "label"),
    [
        ({"model": {"vocab": {"first": 5, "second": 5}}}, {}, "first"),
        ({"added_tokens": [{"id": 5, "content": "first"}, {"id": 5, "content": "second"}]}, {}, "first"),
        (None, {"added_tokens_decoder": {"5": {"content": "first"}, "05": {"content": "second"}}}, "first"),
    ],
)
def test_tokens_sharing_the_top_id_keep_the_first_label(tokenizer: Any, config: dict[str, Any], label: str) -> None:
    assert _extract_tokenizer_meta(tokenizer, config).max_token_str == label


# Split GGUF files


def _split_part(folder: Path, part: int, total: int = 3, **kwargs: Any) -> Path:
    path = folder / f"m-{part:05d}-of-{total:05d}.gguf"
    path.write_bytes(build_gguf_header_bytes(**kwargs))
    return path


def test_split_gguf_merges_every_part(tmp_path: Path) -> None:
    first = _split_part(tmp_path, 1, kv_strings={"general.architecture": "llama"}, tensors=[("a", (32,), 0)])
    _split_part(tmp_path, 2, tensors=[("b", (32,), 0)])
    _split_part(tmp_path, 3, tensors=[("c", (32,), 0)])
    meta = load_checkpoint_metadata(str(first))
    assert set(meta.tensors) == {"a", "b", "c"}
    assert meta.total_weight_bytes == 384
    assert meta.missing_shards == ()
    assert meta.config["model_type"] == "llama"
    assert meta.shard_files == ("m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf")


def test_split_gguf_reports_a_missing_part(tmp_path: Path) -> None:
    first = _split_part(tmp_path, 1, tensors=[("a", (32,), 0)])
    _split_part(tmp_path, 3, tensors=[("c", (32,), 0)])
    meta = load_checkpoint_metadata(str(first))
    assert meta.missing_shards == ("m-00002-of-00003.gguf",)
    finding = next(f for f in analyze_checkpoint(meta).findings if f.rule_id == "QPF009")
    assert finding.title == "Missing part(s) of a split GGUF file"
    assert finding.location == "m-00002-of-00003.gguf"


def test_split_gguf_found_through_its_folder(tmp_path: Path) -> None:
    _split_part(tmp_path, 1, tensors=[("a", (32,), 0)])
    _split_part(tmp_path, 2, tensors=[("b", (32,), 0)])
    assert load_checkpoint_metadata(str(tmp_path)).missing_shards == ("m-00003-of-00003.gguf",)


def test_scanning_any_part_reads_all_of_them(tmp_path: Path) -> None:
    _split_part(tmp_path, 1, tensors=[("a", (32,), 0)])
    second = _split_part(tmp_path, 2, tensors=[("b", (32,), 0)])
    _split_part(tmp_path, 3, tensors=[("c", (32,), 0)])
    assert set(load_checkpoint_metadata(str(second)).tensors) == {"a", "b", "c"}


def test_split_gguf_header_size_adds_up(tmp_path: Path) -> None:
    first = _split_part(tmp_path, 1, 2, tensors=[("a", (32,), 0)])
    second = _split_part(tmp_path, 2, 2, tensors=[("b", (32,), 0)])
    expected = first.stat().st_size + second.stat().st_size
    assert load_checkpoint_metadata(str(first)).header_bytes_fetched == expected


def test_file_that_only_looks_split_is_read_alone(tmp_path: Path) -> None:
    path = tmp_path / "m-00001-of-00000.gguf"
    path.write_bytes(build_gguf_header_bytes(tensors=[("a", (32,), 0)]))
    meta = load_checkpoint_metadata(str(path))
    assert meta.missing_shards == ()
    assert meta.shard_files == (path.name,)


def test_files_without_an_index_are_variants_not_duplicates(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _shard(ckpt / "a.safetensors", {"x.weight": ("F16", [2, 2])})
    _shard(ckpt / "b.safetensors", {"x.weight": ("F16", [2, 2])})
    assert load_checkpoint_metadata(str(ckpt)).duplicate_tensors == ()


def test_prefer_gguf_picks_it_over_safetensors(tmp_path: Path) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / "m.gguf").write_bytes(build_gguf_header_bytes())
    assert load_checkpoint_metadata(str(ckpt), prefer_gguf=True).container_format == "gguf"
    assert load_checkpoint_metadata(str(ckpt)).container_format == "safetensors"
