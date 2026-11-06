"""Hugging Face and direct URL loading against an in-memory repository."""

from __future__ import annotations

import io
import pickle
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from conftest import build_gguf_header_bytes, build_safetensors_header_file
from quantpreflight import loader
from quantpreflight.cli import main
from quantpreflight.loader import CheckpointLoadError, _build_opener, _http_fetch_bytes, load_checkpoint_metadata

BASE = "https://huggingface.co/org/model/resolve/main"
API = "https://huggingface.co/api/models/org/model"


class FakeHub:
    """Serves bytes by URL and records what was requested."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.errors: dict[str, int] = {}
        self.requests: list[str] = []
        self.ranges: list[tuple[int, int]] = []

    def add(self, name: str, data: Any) -> None:
        self.files[f"{BASE}/{name}"] = data if isinstance(data, bytes) else json.dumps(data).encode()

    def siblings(self, *names: str) -> None:
        self.files[API] = json.dumps({"siblings": [{"rfilename": name} for name in names]}).encode()

    def __call__(self, url: str, *, byte_range: tuple[int, int] | None = None, **_: Any) -> bytes:
        self.requests.append(url)
        if url in self.errors:
            raise urllib.error.HTTPError(url, self.errors[url], "boom", None, None)  # type: ignore[arg-type]
        if url not in self.files:
            raise urllib.error.HTTPError(url, 404, "missing", None, None)  # type: ignore[arg-type]
        data = self.files[url]
        if byte_range is None:
            return data
        self.ranges.append(byte_range)
        return data[byte_range[0] : byte_range[1] + 1]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()
    monkeypatch.setattr(loader, "_http_fetch_bytes", fake)
    return fake


def _shard(*names: str) -> bytes:
    return build_safetensors_header_file({name: ("F16", [8, 8]) for name in names})


def test_single_file_repository(hub: FakeHub) -> None:
    hub.add("config.json", {"vocab_size": 8})
    hub.add("model.safetensors", _shard("a.weight"))
    hub.siblings("config.json", "model.safetensors")
    meta = load_checkpoint_metadata("org/model")
    assert meta.container_format == "safetensors"
    assert meta.source == "org/model"
    assert meta.shard_files == ("model.safetensors",)
    assert meta.total_weight_bytes == 128
    assert meta.tokenizer_meta.max_token_id == 7
    assert meta.header_bytes_fetched > 0


def test_repository_without_an_api_listing_still_tries_model_safetensors(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    assert load_checkpoint_metadata("org/model").shard_files == ("model.safetensors",)


def test_a_broken_api_listing_is_ignored(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.files[API] = b"{"
    assert load_checkpoint_metadata("org/model").tensors


def test_revision_is_used_in_urls(hub: FakeHub) -> None:
    other = "https://huggingface.co/org/model/resolve/dev/model.safetensors"
    hub.files[other] = _shard("a.weight")
    assert load_checkpoint_metadata("org/model", revision="dev").tensors
    assert other in hub.requests


def test_sharded_repository_with_a_missing_and_a_dangling_entry(hub: FakeHub) -> None:
    hub.add("model.safetensors.index.json", {"weight_map": {"a.weight": "s1.safetensors", "ghost": "s1.safetensors", "b.weight": "s2.safetensors"}, "metadata": {"total_size": 999}})
    hub.add("s1.safetensors", _shard("a.weight"))
    meta = load_checkpoint_metadata("org/model")
    assert meta.shard_files == ("s1.safetensors", "s2.safetensors")
    assert meta.missing_shards == ("s2.safetensors",)
    assert meta.dangling_index_tensors == ("ghost",)
    assert meta.total_weight_bytes == 999


def test_duplicate_tensors_across_remote_shards(hub: FakeHub) -> None:
    hub.add("model.safetensors.index.json", {"weight_map": {"a.weight": "s1.safetensors", "b": "s2.safetensors"}, "metadata": {"total_size": 1}})
    hub.add("s1.safetensors", _shard("a.weight"))
    hub.add("s2.safetensors", _shard("a.weight"))
    meta = load_checkpoint_metadata("org/model")
    assert meta.duplicate_tensors == ("a.weight",)
    assert meta.total_weight_bytes == 128


def test_shard_read_error_is_reported(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.errors[f"{BASE}/model.safetensors"] = 500
    with pytest.raises(CheckpointLoadError, match="HTTP 500 reading shard"):
        load_checkpoint_metadata("org/model")


@pytest.mark.parametrize("code", [401, 403])
def test_gated_repository_says_how_to_continue(hub: FakeHub, code: int) -> None:
    hub.errors[f"{BASE}/config.json"] = code
    with pytest.raises(CheckpointLoadError, match="HF_TOKEN"):
        load_checkpoint_metadata("org/model")


def test_server_error_on_a_config_file(hub: FakeHub) -> None:
    hub.errors[f"{BASE}/config.json"] = 502
    with pytest.raises(CheckpointLoadError, match="HTTP 502"):
        load_checkpoint_metadata("org/model")


def test_network_error_on_a_config_file(hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url: str, **_: Any) -> bytes:
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(loader, "_http_fetch_bytes", fail)
    with pytest.raises(CheckpointLoadError, match="no route"):
        load_checkpoint_metadata("org/model")


def test_external_quant_configs_on_the_hub(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("hf_quant_config.json", {"quant_algo": "NVFP4"})
    hub.add("quantize_config.json", {"bits": 4, "group_size": 128})
    assert load_checkpoint_metadata("org/model").quant.method == "nvfp4"


def test_quantize_config_on_the_hub(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("quantize_config.json", {"bits": 4, "group_size": 128})
    assert load_checkpoint_metadata("org/model").quant.method == "gptq"


def test_tokenizer_json_is_skipped_when_the_config_has_added_tokens(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("tokenizer_config.json", {"added_tokens_decoder": {"5": {"content": "<x>"}}})
    load_checkpoint_metadata("org/model")
    assert f"{BASE}/tokenizer.json" not in hub.requests


def test_tokenizer_json_is_read_when_needed(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("tokenizer.json", {"added_tokens": [{"id": 500, "content": "<x>"}]})
    assert load_checkpoint_metadata("org/model").tokenizer_meta.max_token_id == 500


def test_gptq_base_name_selects_the_weights_file(hub: FakeHub) -> None:
    hub.add("config.json", {"quantization_config": {"quant_method": "gptq", "model_file_base_name": "gptq_model-4bit"}})
    hub.add("gptq_model-4bit.safetensors", _shard("a.weight"))
    hub.siblings("gptq_model-4bit.safetensors", "other.safetensors", "model.safetensors")
    assert load_checkpoint_metadata("org/model").shard_files == ("gptq_model-4bit.safetensors",)


def test_any_safetensors_files_are_used_as_a_fallback(hub: FakeHub) -> None:
    hub.add("one.safetensors", _shard("a.weight"))
    hub.add("two.SAFETENSORS", _shard("b.weight"))
    hub.siblings("one.safetensors", "two.SAFETENSORS", "notes.txt")
    assert load_checkpoint_metadata("org/model").shard_files == ("one.safetensors", "two.SAFETENSORS")


_PICKLE_ZIP = b"PK\x03\x04rest"
_PICKLE_PROTO0 = pickle.dumps({"data": [1, 2]}, protocol=0)
_METAL_BIN = b"GPT-OSS v1.0\x00weights"


def test_remote_legacy_and_code_files_are_recorded(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("pytorch_model.bin", _PICKLE_ZIP)
    hub.siblings("model.safetensors", "pytorch_model.bin", "training_args.bin", "modeling_x.py", "b.pt")
    meta = load_checkpoint_metadata("org/model")
    assert meta.legacy_weight_files == ("pytorch_model.bin",)
    assert meta.custom_code_files == ("modeling_x.py",)
    assert (BASE + "/pytorch_model.bin") in hub.requests
    assert (0, 7) in hub.ranges


def test_remote_metal_model_bin_is_not_treated_as_pickle(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("metal_model.bin", _METAL_BIN)
    hub.add("pytorch_model.bin", _PICKLE_PROTO0)
    hub.add("zip_model.bin", _PICKLE_ZIP)
    hub.siblings("model.safetensors", "metal_model.bin", "pytorch_model.bin", "zip_model.bin")
    meta = load_checkpoint_metadata("org/model")
    assert set(meta.legacy_weight_files) == {"pytorch_model.bin", "zip_model.bin"}
    assert "metal_model.bin" not in meta.legacy_weight_files
    assert (0, 7) in hub.ranges


@pytest.mark.parametrize(
    ("names", "chosen"),
    [
        (["a-q8_0.gguf", "a-q4_k_m.gguf", "a-q4_0.gguf"], "a-q4_k_m.gguf"),
        (["a-q8_0.gguf", "a-q4_0.gguf"], "a-q4_0.gguf"),
        (["a-f16.gguf", "a-q8_0.gguf"], "a-f16.gguf"),
        (["A-Q4_K_M.GGUF"], "A-Q4_K_M.GGUF"),
    ],
)
def test_bare_gguf_repository_picks_a_common_variant(hub: FakeHub, names: list[str], chosen: str) -> None:
    for name in names:
        hub.add(name, build_gguf_header_bytes(kv_strings={"general.architecture": "x"}))
    hub.siblings(*names)
    meta = load_checkpoint_metadata("org/model")
    assert meta.container_format == "gguf"
    assert meta.source == f"org/model ({chosen})"
    assert meta.shard_files == (chosen,)
    assert meta.header_bytes_fetched > 0


def test_pickle_only_repository(hub: FakeHub) -> None:
    hub.add("config.json", {"vocab_size": 8, "quantization_config": {"quant_method": "gptq"}})
    hub.add("pytorch_model.bin", _PICKLE_ZIP)
    hub.siblings("pytorch_model.bin", "run.py")
    meta = load_checkpoint_metadata("org/model")
    assert meta.container_format == "pytorch_bin"
    assert meta.legacy_weight_files == ("pytorch_model.bin",)
    assert meta.custom_code_files == ("run.py",)
    assert meta.quant.method == "gptq"
    assert meta.tokenizer_meta.max_token_id == 7


def test_unreadable_remote_legacy_candidate_is_still_counted(hub: FakeHub) -> None:
    """When the first bytes cannot be fetched, keep warning (issue #6)."""
    hub.add("config.json", {"vocab_size": 8})
    hub.siblings("pytorch_model.bin", "run.py")
    # No body for pytorch_model.bin → FakeHub 404 on the magic sniff.
    meta = load_checkpoint_metadata("org/model")
    assert meta.container_format == "pytorch_bin"
    assert meta.legacy_weight_files == ("pytorch_model.bin",)


def test_repository_with_nothing_to_read(hub: FakeHub) -> None:
    hub.siblings("README.md")
    with pytest.raises(CheckpointLoadError, match="No model.safetensors"):
        load_checkpoint_metadata("org/model")


def test_missing_single_file_url_is_not_a_gguf_fallback(hub: FakeHub) -> None:
    meta = load_checkpoint_metadata("https://huggingface.co/org/model/resolve/main/sub/weights.safetensors")
    assert meta.missing_shards == ("sub/weights.safetensors",)


def test_hub_url_to_a_gguf_file(hub: FakeHub) -> None:
    hub.add("sub/m.gguf", build_gguf_header_bytes(tensors=[("t", (32,), 0)]))
    meta = load_checkpoint_metadata("https://huggingface.co/org/model/resolve/main/sub/m.gguf")
    assert meta.container_format == "gguf"
    assert meta.shard_files == ("sub/m.gguf",)
    assert meta.total_weight_bytes == 128
    assert f"{BASE}/config.json" not in hub.requests


def test_hub_url_to_a_safetensors_file(hub: FakeHub) -> None:
    hub.add("sub/m.safetensors", _shard("a.weight"))
    hub.add("config.json", {"vocab_size": 8})
    meta = load_checkpoint_metadata("https://huggingface.co/org/model/resolve/main/sub/m.safetensors")
    assert meta.shard_files == ("sub/m.safetensors",)
    assert meta.config == {"vocab_size": 8}


def test_hub_url_does_not_call_the_listing_api(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    load_checkpoint_metadata("https://huggingface.co/org/model")
    assert API not in hub.requests


@pytest.mark.parametrize(
    ("url", "blob", "container"),
    [
        ("https://cdn.example/w/m.safetensors", _shard("a.weight"), "safetensors"),
        ("https://cdn.example/w/M.SAFETENSORS?x=1", _shard("a.weight"), "safetensors"),
        ("https://cdn.example/w/m.gguf", build_gguf_header_bytes(tensors=[("t", (32,), 0)]), "gguf"),
    ],
)
def test_direct_urls(hub: FakeHub, url: str, blob: bytes, container: str) -> None:
    hub.files[url] = blob
    meta = load_checkpoint_metadata(url)
    assert meta.container_format == container
    assert meta.source == url
    assert meta.total_weight_bytes > 0
    assert meta.header_bytes_fetched == sum(end - start + 1 for start, end in hub.ranges) or meta.header_bytes_fetched == len(blob)
    assert meta.shard_files[0].lower().startswith("m.")


def test_unsupported_direct_url(hub: FakeHub) -> None:
    with pytest.raises(CheckpointLoadError, match="Unsupported direct checkpoint URL"):
        load_checkpoint_metadata("https://cdn.example/w/m.zip")


def test_token_reaches_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens: list[Any] = []
    fake = FakeHub()
    fake.add("model.safetensors", _shard("a.weight"))

    def recording(url: str, **kwargs: Any) -> bytes:
        tokens.append(kwargs.get("token"))
        return fake(url, **kwargs)

    monkeypatch.setattr(loader, "_http_fetch_bytes", recording)
    load_checkpoint_metadata("org/model", token="secret")
    assert tokens
    assert set(tokens) == {"secret"}


@pytest.mark.parametrize("target", ["https://cdn.example/w/m.safetensors", "https://cdn.example/w/m.gguf"])
def test_cli_turns_a_missing_direct_url_into_exit_2(hub: FakeHub, target: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", target]) == 2
    assert "Error scanning" in capsys.readouterr().err


def test_cli_turns_a_missing_hub_gguf_into_exit_2(hub: FakeHub, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "https://huggingface.co/org/model/resolve/main/m.gguf"]) == 2
    assert "Error scanning" in capsys.readouterr().err


# The HTTP layer


class _Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200) -> None:
        super().__init__(data)
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class _Opener:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[Any] = []

    def open(self, request: Any, timeout: float = 0) -> Any:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def opener(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(*outcomes: Any) -> _Opener:
        fake = _Opener(list(outcomes))
        monkeypatch.setattr(loader, "_build_opener", lambda: fake)
        monkeypatch.setattr(loader.time, "sleep", lambda _: None)
        return fake

    return install


def test_fetch_sends_range_token_and_identity_encoding(opener: Any) -> None:
    fake = opener(_Response(b"abcd", 206))
    assert _http_fetch_bytes("https://x/y", byte_range=(2, 5), token="t") == b"abcd"
    request = fake.requests[0]
    assert request.get_header("Range") == "bytes=2-5"
    assert request.get_header("Authorization") == "Bearer t"
    assert request.get_header("Accept-encoding") == "identity"
    assert request.get_header("User-agent") == loader.USER_AGENT


def test_fetch_without_a_token_sends_no_authorization(opener: Any) -> None:
    fake = opener(_Response(b"x"))
    _http_fetch_bytes("https://x/y")
    assert fake.requests[0].get_header("Authorization") is None
    assert fake.requests[0].get_header("Range") is None


def test_fetch_limits_a_range_answer_to_the_range_length(opener: Any) -> None:
    opener(_Response(b"abcdefgh", 206))
    assert _http_fetch_bytes("https://x/y", byte_range=(0, 3)) == b"abcd"


def test_fetch_accepts_a_full_answer_for_a_range_from_zero(opener: Any) -> None:
    opener(_Response(b"abcdefgh", 200))
    assert _http_fetch_bytes("https://x/y", byte_range=(0, 3)) == b"abcd"


def test_fetch_rejects_a_full_answer_for_a_later_range(opener: Any) -> None:
    opener(_Response(b"abcdefgh", 200))
    with pytest.raises(CheckpointLoadError, match="ignored HTTP Range"):
        _http_fetch_bytes("https://x/y", byte_range=(4, 7))


def test_fetch_size_ceiling(opener: Any) -> None:
    opener(_Response(b"abcd"))
    with pytest.raises(CheckpointLoadError, match="exceeds 3 byte limit"):
        _http_fetch_bytes("https://x/y", max_bytes=3)


def test_fetch_size_ceiling_allows_exactly_the_limit(opener: Any) -> None:
    opener(_Response(b"abc"))
    assert _http_fetch_bytes("https://x/y", max_bytes=3) == b"abc"


@pytest.mark.parametrize("failure", [urllib.error.URLError("reset"), TimeoutError("slow"), ConnectionResetError("reset")])
def test_fetch_retries_transient_failures(opener: Any, failure: Exception) -> None:
    fake = opener(failure, failure, _Response(b"ok"))
    assert _http_fetch_bytes("https://x/y") == b"ok"
    assert len(fake.requests) == 3


def test_fetch_gives_up_after_its_retries(opener: Any) -> None:
    failure = urllib.error.URLError("reset")
    fake = opener(failure, failure)
    with pytest.raises(CheckpointLoadError, match="after 2 attempts"):
        _http_fetch_bytes("https://x/y", retries=1)
    assert len(fake.requests) == 2


def test_fetch_does_not_retry_http_errors(opener: Any) -> None:
    fake = opener(urllib.error.HTTPError("https://x/y", 404, "missing", None, None), _Response(b"never"))  # type: ignore[arg-type]
    with pytest.raises(urllib.error.HTTPError):
        _http_fetch_bytes("https://x/y")
    assert len(fake.requests) == 1


def test_fetch_backs_off_between_retries(monkeypatch: pytest.MonkeyPatch, opener: Any) -> None:
    failure = urllib.error.URLError("reset")
    opener(failure, failure, _Response(b"ok"))
    naps: list[float] = []
    monkeypatch.setattr(loader.time, "sleep", naps.append)
    _http_fetch_bytes("https://x/y")
    assert naps == [0.4, 0.8]


def test_real_opener_has_the_range_preserving_handler() -> None:
    handlers = [type(h).__name__ for h in _build_opener().handlers]
    assert "_RangePreservingRedirectHandler" in handlers


# Redirects


def _redirected(url: str, new_url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer t", "Range": "bytes=0-9", "Accept-Encoding": "identity", "User-Agent": "qpf"},
    )
    request.add_unredirected_header("Authorization", "Bearer u")
    return loader._RangePreservingRedirectHandler().redirect_request(request, None, 302, "Found", {}, new_url)


def test_redirect_to_another_host_drops_the_token_everywhere() -> None:
    out = _redirected("https://huggingface.co/a", "https://cdn.example/b")
    assert out.get_header("Authorization") is None
    assert not [key for key in {**out.headers, **out.unredirected_hdrs} if key.lower() == "authorization"]


def test_redirect_to_another_port_counts_as_another_host() -> None:
    assert _redirected("http://localhost:1/a", "http://localhost:2/b").get_header("Authorization") is None


def test_redirect_on_the_same_host_keeps_the_token() -> None:
    assert _redirected("https://huggingface.co/a", "https://HuggingFace.co/b").get_header("Authorization") == "Bearer t"


def test_redirect_keeps_the_other_headers() -> None:
    out = _redirected("https://huggingface.co/a", "https://cdn.example/b")
    assert out.get_header("Range") == "bytes=0-9"
    assert out.get_header("Accept-encoding") == "identity"
    assert out.get_header("User-agent") == "qpf"
    assert set(out.unredirected_hdrs) == {"Range", "Accept-encoding"}


def test_redirect_that_urllib_refuses_stays_refused() -> None:
    request = urllib.request.Request("https://huggingface.co/a", method="POST")
    handler = loader._RangePreservingRedirectHandler()
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(request, io.BytesIO(), 305, "Use Proxy", {}, "https://cdn.example/b")


def test_tokenizer_json_is_read_when_the_config_has_no_added_tokens(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("tokenizer_config.json", {"eos_token": "<e>"})
    hub.add("tokenizer.json", {"added_tokens": [{"id": 500, "content": "<e>"}]})
    assert load_checkpoint_metadata("org/model").tokenizer_meta.eos_token_ids == (500,)


def test_gptq_base_name_is_ignored_when_that_file_does_not_exist(hub: FakeHub) -> None:
    hub.add("config.json", {"quantization_config": {"quant_method": "gptq", "model_file_base_name": "gone"}})
    hub.add("model.safetensors", _shard("a.weight"))
    hub.siblings("model.safetensors")
    assert load_checkpoint_metadata("org/model").shard_files == ("model.safetensors",)


def test_split_gguf_repository_reads_every_part(hub: FakeHub) -> None:
    names = ["m-q4_k_m-00001-of-00002.gguf", "m-q4_k_m-00002-of-00002.gguf"]
    hub.add(names[0], build_gguf_header_bytes(tensors=[("a", (32,), 0)]))
    hub.add(names[1], build_gguf_header_bytes(tensors=[("b", (32,), 0)]))
    hub.siblings(*names)
    meta = load_checkpoint_metadata("org/model")
    assert set(meta.tensors) == {"a", "b"}
    assert meta.shard_files == tuple(names)
    assert meta.missing_shards == ()


def test_split_gguf_repository_reports_a_missing_part(hub: FakeHub) -> None:
    name = "m-q4_k_m-00001-of-00002.gguf"
    hub.add(name, build_gguf_header_bytes(tensors=[("a", (32,), 0)]))
    hub.siblings(name)
    assert load_checkpoint_metadata("org/model").missing_shards == ("m-q4_k_m-00002-of-00002.gguf",)


def test_split_gguf_url_reports_a_missing_part(hub: FakeHub) -> None:
    hub.add("m-00001-of-00002.gguf", build_gguf_header_bytes(tensors=[("a", (32,), 0)]))
    meta = load_checkpoint_metadata("https://huggingface.co/org/model/resolve/main/m-00001-of-00002.gguf")
    assert meta.missing_shards == ("m-00002-of-00002.gguf",)
    assert meta.shard_files == ("m-00001-of-00002.gguf",)


def test_split_gguf_with_no_parts_at_all(hub: FakeHub) -> None:
    with pytest.raises(CheckpointLoadError, match="No part of the split GGUF"):
        load_checkpoint_metadata("https://huggingface.co/org/model/resolve/main/m-00001-of-00002.gguf")


def test_split_gguf_server_error_is_not_a_missing_part(hub: FakeHub) -> None:
    hub.add("m-00001-of-00002.gguf", build_gguf_header_bytes(tensors=[("a", (32,), 0)]))
    hub.errors[f"{BASE}/m-00002-of-00002.gguf"] = 500
    with pytest.raises(urllib.error.HTTPError):
        load_checkpoint_metadata("https://huggingface.co/org/model/resolve/main/m-00001-of-00002.gguf")


def test_prefer_gguf_picks_it_over_safetensors_on_the_hub(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("m-q4_k_m.gguf", build_gguf_header_bytes(tensors=[("t", (32,), 0)]))
    hub.siblings("model.safetensors", "m-q4_k_m.gguf")
    assert load_checkpoint_metadata("org/model", prefer_gguf=True).container_format == "gguf"
    assert load_checkpoint_metadata("org/model").container_format == "safetensors"


def test_prefer_gguf_falls_back_to_safetensors(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.siblings("model.safetensors")
    assert load_checkpoint_metadata("org/model", prefer_gguf=True).container_format == "safetensors"


def test_cli_prefers_gguf_for_llama_cpp(hub: FakeHub, capsys: pytest.CaptureFixture[str]) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("m.gguf", build_gguf_header_bytes(tensors=[("t", (32,), 0)]))
    hub.siblings("model.safetensors", "m.gguf")
    assert main(["check", "org/model", "--engine", "llamacpp"]) == 0
    assert "(gguf)" in capsys.readouterr().out


def test_broken_tokenizer_json_does_not_stop_a_scan(hub: FakeHub) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    hub.add("tokenizer.json", b"{")
    assert load_checkpoint_metadata("org/model").tokenizer_meta.max_token_id is None


def test_tokenizer_json_over_the_limit_does_not_stop_a_scan(hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    hub.add("model.safetensors", _shard("a.weight"))
    original = loader._http_fetch_bytes

    def limited(url: str, **kwargs: Any) -> bytes:
        if url.endswith("tokenizer.json"):
            raise CheckpointLoadError("Resource exceeds the limit")
        return original(url, **kwargs)

    monkeypatch.setattr(loader, "_http_fetch_bytes", limited)
    assert load_checkpoint_metadata("org/model").tensors


def test_duplicates_need_an_index_on_the_hub(hub: FakeHub) -> None:
    hub.add("one.safetensors", _shard("a.weight"))
    hub.add("two.safetensors", _shard("a.weight"))
    hub.siblings("one.safetensors", "two.safetensors")
    assert load_checkpoint_metadata("org/model").duplicate_tensors == ()


@pytest.mark.parametrize("code", [429, 502, 503, 504])
def test_fetch_retries_rate_limits_and_server_errors(opener: Any, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    failure = urllib.error.HTTPError("https://x/y", code, "busy", None, None)  # type: ignore[arg-type]
    fake = opener(failure, _Response(b"ok"))
    naps: list[float] = []
    monkeypatch.setattr(loader.time, "sleep", naps.append)
    assert _http_fetch_bytes("https://x/y") == b"ok"
    assert len(fake.requests) == 2
    assert naps == [3.0]


def test_fetch_gives_up_on_a_lasting_rate_limit(opener: Any) -> None:
    failure = urllib.error.HTTPError("https://x/y", 429, "busy", None, None)  # type: ignore[arg-type]
    fake = opener(failure, failure)
    with pytest.raises(urllib.error.HTTPError):
        _http_fetch_bytes("https://x/y", retries=1)
    assert len(fake.requests) == 2
