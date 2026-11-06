"""Unified local and remote HTTP-Range checkpoint header loader (zero weight downloads)."""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .gguf_io import ParsedGGUFHeader, merge_gguf_parts, read_gguf_header_via_range, split_gguf_names
from .models import CheckpointMeta, QuantSpec, TokenizerMeta
from .quant_detect import build_safetensors_specs, normalize_gguf_metadata, parse_quant_spec_from_config
from .safetensors_io import (
    ParsedSafetensorsHeader,
    RawSafetensorEntry,
    SafetensorsFormatError,
    parse_safetensors_index,
    read_safetensors_header_via_range,
)

MAX_JSON_CONFIG_BYTES = 128 * 1024 * 1024  # Ceiling for tokenizer.json and index.json
USER_AGENT = f"quantpreflight/{__version__}"
RETRYABLE_HTTP_CODES = (429, 502, 503, 504)
RETRY_WAIT_SECONDS = 3.0


class CheckpointLoadError(RuntimeError):
    """Raised when checkpoint headers or metadata cannot be located or fetched."""


_LEGACY_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth")
_LEGACY_WEIGHT_NAME_HINTS = ("model", "pytorch", "weight", "consolidated")
# Formats known not to be torch pickle (issue #6)
_KNOWN_NON_PICKLE_PREFIXES: tuple[bytes, ...] = (b"GPT-OSS ", b"GGUF")


def _looks_like_legacy_weight_name(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(_LEGACY_WEIGHT_SUFFIXES) and any(
        hint in lower for hint in _LEGACY_WEIGHT_NAME_HINTS
    )


def _header_is_known_non_pickle(header: bytes) -> bool:
    """True when the first bytes match a format known not to be torch pickle."""
    return any(header.startswith(prefix) for prefix in _KNOWN_NON_PICKLE_PREFIXES)


def _local_legacy_weight_files(directory: Path, filenames: list[str]) -> list[str]:
    """Name-filter then sniff magic bytes so Metal/other non-pickle formats are not flagged."""
    found: list[str] = []
    for fn in filenames:
        if not _looks_like_legacy_weight_name(fn):
            continue
        path = directory / fn
        try:
            with path.open("rb") as handle:
                header = handle.read(8)
        except OSError:
            # Unreadable files still warn — a scanner should err on the side of caution.
            found.append(fn)
            continue
        if not _header_is_known_non_pickle(header):
            found.append(fn)
    return found


def _remote_legacy_weight_files(
    filenames: list[str],
    *,
    base_url: str,
    token: str | None,
    timeout: float,
) -> list[str]:
    candidates = [fn for fn in filenames if _looks_like_legacy_weight_name(fn)]
    if not candidates:
        return []

    def _probe_candidate(fn: str) -> bool:
        url = f"{base_url}/{fn}"
        try:
            header = _http_fetch_bytes(url, byte_range=(0, 7), token=token, timeout=timeout)
        except Exception:
            return True
        return not _header_is_known_non_pickle(header)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        matches = list(pool.map(_probe_candidate, candidates))

    return [fn for fn, is_pickle in zip(candidates, matches) if is_pickle]


class _RangePreservingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Preserve `Range` across cross-origin CDN redirects while stripping `Authorization`."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        old_host = urllib.parse.urlparse(req.full_url).netloc.lower()
        new_host = urllib.parse.urlparse(newurl).netloc.lower()

        for hdr_key, hdr_val in list(req.header_items()):
            low = hdr_key.lower()
            if low == "range":
                redirected.add_unredirected_header("Range", hdr_val)
            elif low == "accept-encoding":
                redirected.add_unredirected_header("Accept-Encoding", hdr_val)

        if old_host != new_host:
            for key in list(redirected.headers.keys()):
                if key.lower() == "authorization":
                    del redirected.headers[key]
            for key in list(redirected.unredirected_hdrs.keys()):
                if key.lower() == "authorization":
                    del redirected.unredirected_hdrs[key]

        return redirected


def _build_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RangePreservingRedirectHandler())


def _resolve_hf_token(explicit_token: str | None = None) -> str | None:
    if explicit_token:
        return explicit_token.strip()
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    return None


def _http_fetch_bytes(
    url: str,
    *,
    byte_range: tuple[int, int] | None = None,
    max_bytes: int = MAX_JSON_CONFIG_BYTES,
    token: str | None = None,
    timeout: float = 25.0,
    retries: int = 2,
) -> bytes:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
    }
    if byte_range is not None:
        start, end = byte_range
        headers["Range"] = f"bytes={start}-{end}"
        max_bytes = min(max_bytes, end - start + 1)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            opener = _build_opener()
            with opener.open(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if byte_range is not None and status == 200 and byte_range[0] > 0:
                    raise CheckpointLoadError(
                        f"Remote server at {url} ignored HTTP Range header (returned HTTP 200 for offset {byte_range[0]})."
                    )
                data = resp.read(max_bytes + 1)
                if byte_range is None and len(data) > max_bytes:
                    raise CheckpointLoadError(f"Resource at {url} exceeds {max_bytes} byte limit.")
                return data[:max_bytes]
        except urllib.error.HTTPError as exc:
            # Rate limits and server hiccups pass, so they're worth a slower retry
            if exc.code in RETRYABLE_HTTP_CODES and attempt < retries:
                time.sleep(RETRY_WAIT_SECONDS * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
                continue
            raise CheckpointLoadError(f"Network error fetching {url} after {retries + 1} attempts: {exc}") from exc
    if last_err:
        raise last_err
    return b""


def _extract_tokenizer_meta(
    tokenizer_json: dict[str, Any] | None,
    tokenizer_config: dict[str, Any],
    gguf_metadata: dict[str, Any] | None = None,
    config_vocab_size: int | None = None,
) -> TokenizerMeta:
    vocab_max_id: int | None = None
    max_id: int | None = None
    max_tok_str: str | None = None
    token_to_id: dict[str, int] = {}

    if isinstance(config_vocab_size, int) and config_vocab_size > 0:
        vocab_max_id = config_vocab_size - 1
        max_id = config_vocab_size - 1
        max_tok_str = f"<vocab_size:{config_vocab_size}>"

    if isinstance(tokenizer_json, dict):
        model_sec = tokenizer_json.get("model")
        if isinstance(model_sec, dict) and isinstance(model_sec.get("vocab"), dict):
            for tok, tid in model_sec["vocab"].items():
                if isinstance(tid, int):
                    token_to_id[str(tok)] = tid
                    if vocab_max_id is None or tid > vocab_max_id:
                        vocab_max_id = tid
                    if max_id is None or tid > max_id:
                        max_id = tid
                        max_tok_str = str(tok)

        added = tokenizer_json.get("added_tokens")
        if isinstance(added, list):
            for item in added:
                if isinstance(item, dict) and isinstance(item.get("id"), int):
                    tid = item["id"]
                    content = str(item.get("content") or f"<id:{tid}>")
                    token_to_id[content] = tid
                    if max_id is None or tid > max_id:
                        max_id = tid
                        max_tok_str = content

    added_dec = tokenizer_config.get("added_tokens_decoder")
    if isinstance(added_dec, dict):
        for tid_str, item in added_dec.items():
            try:
                tid = int(tid_str)
            except (TypeError, ValueError):
                continue
            content = str(item.get("content")) if isinstance(item, dict) and "content" in item else f"<id:{tid}>"
            token_to_id.setdefault(content, tid)
            if max_id is None or tid > max_id:
                max_id = tid
                max_tok_str = content

    eos_raw = tokenizer_config.get("eos_token")
    eos_str: str | None = None
    if isinstance(eos_raw, str):
        eos_str = eos_raw
    elif isinstance(eos_raw, dict) and isinstance(eos_raw.get("content"), str):
        eos_str = eos_raw["content"]

    eos_ids: list[int] = []
    if eos_str and eos_str in token_to_id:
        eos_ids.append(token_to_id[eos_str])

    if gguf_metadata:
        tok_tbl = gguf_metadata.get("tokenizer.ggml.tokens")
        if isinstance(tok_tbl, dict) and "__gguf_string_table_len__" in tok_tbl:
            cnt = int(tok_tbl["__gguf_string_table_len__"])
            if cnt > 0:
                vocab_max_id = cnt - 1
                max_id = cnt - 1
                max_tok_str = str(tok_tbl.get("last") or f"<id:{cnt - 1}>")
        gguf_eos = gguf_metadata.get("tokenizer.ggml.eos_token_id")
        if isinstance(gguf_eos, int):
            eos_ids.append(gguf_eos)

    total_tokens = (max_id + 1) if max_id is not None else None
    return TokenizerMeta(
        vocab_max_id=vocab_max_id,
        max_token_id=max_id,
        max_token_str=max_tok_str,
        total_tokens=total_tokens,
        eos_token=eos_str,
        eos_token_ids=tuple(dict.fromkeys(eos_ids)),
    )


def _load_local_safetensors_file(path: Path) -> ParsedSafetensorsHeader:
    def _range(start: int, end: int) -> bytes:
        with path.open("rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)

    return read_safetensors_header_via_range(_range)


def _load_local_checkpoint(target_path: Path, prefer_gguf: bool = False) -> CheckpointMeta:
    if target_path.is_file():
        if target_path.suffix.lower() == ".gguf":
            return _load_local_gguf_file(target_path)
        if target_path.name in ("config.json", "model.safetensors.index.json"):
            target_path = target_path.parent
        elif target_path.suffix.lower() == ".safetensors":
            parsed = _load_local_safetensors_file(target_path)
            cfg_path = target_path.parent / "config.json"
            config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            quant = parse_quant_spec_from_config(config)
            raw_map = {k: (v, target_path.name) for k, v in parsed.tensors.items()}
            specs = build_safetensors_specs(raw_map, quant)
            return CheckpointMeta(
                source=str(target_path),
                container_format="safetensors",
                config=config,
                tensors=specs,
                shard_files=(target_path.name,),
                shard_file_sizes={target_path.name: target_path.stat().st_size},
                shard_header_lengths={target_path.name: parsed.header_byte_length},
                safetensors_metadata=parsed.metadata,
                header_bytes_fetched=parsed.header_byte_length,
                total_weight_bytes=sum(t.byte_size for t in specs.values()),
                quant=quant,
            )
        else:
            raise CheckpointLoadError(f"Unsupported checkpoint file: {target_path}")

    if not target_path.is_dir():
        raise CheckpointLoadError(f"Local path does not exist: {target_path}")

    gguf_files = sorted(target_path.glob("*.gguf"))
    st_index_path = target_path / "model.safetensors.index.json"
    st_single_path = target_path / "model.safetensors"
    st_glob = sorted(target_path.glob("*.safetensors"))

    if gguf_files and (prefer_gguf or (not st_index_path.is_file() and not st_glob)):
        return _load_local_gguf_file(gguf_files[0])

    def _read_opt_json(filename: str) -> dict[str, Any]:
        p = target_path / filename
        if not p.is_file():
            return {}
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError as exc:
            raise CheckpointLoadError(f"Invalid JSON in {p}: {exc}") from exc

    config = _read_opt_json("config.json")
    if not config.get("quantization_config"):
        hf_quant = _read_opt_json("hf_quant_config.json")
        if hf_quant:
            config["quantization_config"] = hf_quant
        else:
            q_cfg = _read_opt_json("quantize_config.json")
            if q_cfg:
                config["quantization_config"] = q_cfg

    gen_config = _read_opt_json("generation_config.json")
    tok_config = _read_opt_json("tokenizer_config.json")
    tok_json_path = target_path / "tokenizer.json"
    tok_json = _read_opt_json("tokenizer.json") if tok_json_path.is_file() else None

    all_dir_files = [p.name for p in target_path.iterdir() if p.is_file()]
    legacy_files = _local_legacy_weight_files(target_path, all_dir_files)
    custom_code_files = [
        fn for fn in all_dir_files
        if fn.lower().endswith(".py")
    ]

    shard_names: list[str] = []
    expected_weight_map: dict[str, str] | None = None
    index_total_size: int | None = None

    if st_index_path.is_file():
        idx_obj = _read_opt_json("model.safetensors.index.json")
        expected_weight_map, index_total_size = parse_safetensors_index(idx_obj)
        shard_names = sorted(set(expected_weight_map.values()))
    elif st_single_path.is_file():
        shard_names = ["model.safetensors"]
    elif st_glob:
        shard_names = [p.name for p in st_glob]
    elif legacy_files:
        quant = parse_quant_spec_from_config(config)
        return CheckpointMeta(
            source=str(target_path),
            container_format="pytorch_bin",
            config=config,
            generation_config=gen_config,
            tokenizer_config=tok_config,
            tokenizer_meta=_extract_tokenizer_meta(tok_json, tok_config, config_vocab_size=config.get("vocab_size")),
            tensors={},
            shard_files=tuple(sorted(legacy_files)),
            legacy_weight_files=tuple(sorted(legacy_files)),
            custom_code_files=tuple(sorted(custom_code_files)),
            quant=quant,
        )
    else:
        raise CheckpointLoadError(
            f"No .safetensors or .gguf files found in directory: {target_path}"
        )

    missing_shards: list[str] = []
    duplicate_tensors: list[str] = []
    raw_entries: dict[str, tuple[RawSafetensorEntry, str]] = {}
    merged_meta: dict[str, str] = {}
    shard_sizes: dict[str, int] = {}
    shard_hdr_lens: dict[str, int] = {}
    header_bytes = 0

    for sname in shard_names:
        spath = (target_path / sname).resolve()
        if not spath.is_relative_to(target_path.resolve()):
            raise SafetensorsFormatError(f"Shard path escapes checkpoint directory: {sname}")
        if not spath.is_file():
            missing_shards.append(sname)
            continue
        shard_sizes[sname] = spath.stat().st_size
        parsed = _load_local_safetensors_file(spath)
        shard_hdr_lens[sname] = parsed.header_byte_length
        header_bytes += parsed.header_byte_length
        merged_meta.update(parsed.metadata)
        for tname, entry in parsed.tensors.items():
            # Without an index, several files are usually variants of one model
            if tname in raw_entries and expected_weight_map is not None:
                duplicate_tensors.append(tname)
            raw_entries[tname] = (entry, sname)

    dangling: list[str] = []
    if expected_weight_map is not None:
        for tname, sname in expected_weight_map.items():
            if sname not in missing_shards and tname not in raw_entries:
                dangling.append(tname)

    quant = parse_quant_spec_from_config(config)
    specs = build_safetensors_specs(raw_entries, quant)
    calc_bytes = sum(t.byte_size for t in specs.values())
    total_bytes = index_total_size if (index_total_size and missing_shards) else calc_bytes

    return CheckpointMeta(
        source=str(target_path),
        container_format="safetensors",
        config=config,
        generation_config=gen_config,
        tokenizer_config=tok_config,
        tokenizer_meta=_extract_tokenizer_meta(tok_json, tok_config, config_vocab_size=config.get("vocab_size")),
        tensors=specs,
        shard_files=tuple(shard_names),
        missing_shards=tuple(missing_shards),
        dangling_index_tensors=tuple(sorted(dangling)),
        duplicate_tensors=tuple(sorted(set(duplicate_tensors))),
        legacy_weight_files=tuple(sorted(legacy_files)),
        custom_code_files=tuple(sorted(custom_code_files)),
        shard_file_sizes=shard_sizes,
        shard_header_lengths=shard_hdr_lens,
        safetensors_metadata=merged_meta,
        header_bytes_fetched=header_bytes,
        total_weight_bytes=total_bytes,
        quant=quant,
    )


def _read_gguf_parts(
    name: str,
    read_part: Callable[[str], ParsedGGUFHeader],
) -> tuple[ParsedGGUFHeader, tuple[str, ...], tuple[str, ...]]:
    """Read a GGUF file, or every part of a split one. Returns `(header, found, missing)`."""
    found: list[str] = []
    missing: list[str] = []
    parts: list[ParsedGGUFHeader] = []
    for part_name in split_gguf_names(name):
        try:
            parts.append(read_part(part_name))
            found.append(part_name)
        except FileNotFoundError:
            missing.append(part_name)
    if not parts:
        raise CheckpointLoadError(f"No part of the split GGUF file {name} could be found.")
    return merge_gguf_parts(parts), tuple(found), tuple(missing)


def _load_local_gguf_file(path: Path) -> CheckpointMeta:
    def _read(part_name: str) -> ParsedGGUFHeader:
        part_path = path.parent / part_name
        with part_path.open("rb") as handle:
            def _range(start: int, end: int) -> bytes:
                handle.seek(start)
                return handle.read(end - start + 1)

            return read_gguf_header_via_range(_range)

    parsed, found, missing = _read_gguf_parts(path.name, _read)
    config, quant, specs = normalize_gguf_metadata(parsed)
    tok_meta = _extract_tokenizer_meta(None, {}, parsed.metadata)
    return CheckpointMeta(
        source=str(path),
        container_format="gguf",
        config=config,
        tokenizer_meta=tok_meta,
        tensors=specs,
        shard_files=found,
        missing_shards=missing,
        gguf_metadata=parsed.metadata,
        header_bytes_fetched=parsed.header_bytes_consumed,
        total_weight_bytes=sum(t.byte_size for t in specs.values()),
        quant=quant,
    )


def _parse_hf_target(target: str, revision: str) -> tuple[str, str | None] | None:
    """Return `(base_resolve_url, single_filename)` if `target` is a Hugging Face repo or URL."""
    if target.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(target)
        if parsed.netloc.lower() in ("huggingface.co", "hf.co"):
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                if len(parts) >= 5 and parts[2] in ("resolve", "blob"):
                    rev = parts[3]
                    subpath = "/".join(parts[4:])
                    return f"https://huggingface.co/{repo_id}/resolve/{rev}", subpath
                return f"https://huggingface.co/{repo_id}/resolve/{revision}", None
        return None

    # Bare repo ID `org/model` when no local path exists
    parts = target.strip("/").split("/")
    if len(parts) == 2 and all(parts) and not any(c in target for c in ("\\", ":")):
        return f"https://huggingface.co/{parts[0]}/{parts[1]}/resolve/{revision}", None
    return None


def load_checkpoint_metadata(
    target: str,
    *,
    revision: str = "main",
    token: str | None = None,
    timeout: float = 20.0,
    prefer_gguf: bool = False,
) -> CheckpointMeta:
    """Load zero-weight header and config metadata from a local path, URL, or Hugging Face repo."""
    candidate_path = Path(target)
    if candidate_path.exists():
        return _load_local_checkpoint(candidate_path, prefer_gguf)

    resolved_token = _resolve_hf_token(token)
    hf_parsed = _parse_hf_target(target, revision)

    if hf_parsed is None and target.startswith(("http://", "https://")):
        # Direct URL to .safetensors or .gguf
        path_lower = urllib.parse.urlparse(target).path.lower()
        fetched_total = 0

        def _url_range(start: int, end: int) -> bytes:
            nonlocal fetched_total
            chunk = _http_fetch_bytes(
                target, byte_range=(start, end), token=resolved_token, timeout=timeout
            )
            fetched_total += len(chunk)
            return chunk

        if path_lower.endswith(".gguf"):
            parsed_gguf = read_gguf_header_via_range(_url_range)
            config, quant, specs = normalize_gguf_metadata(parsed_gguf)
            return CheckpointMeta(
                source=target,
                container_format="gguf",
                config=config,
                tokenizer_meta=_extract_tokenizer_meta(None, {}, parsed_gguf.metadata),
                tensors=specs,
                shard_files=( os.path.basename(path_lower) or "remote.gguf",),
                gguf_metadata=parsed_gguf.metadata,
                header_bytes_fetched=fetched_total,
                total_weight_bytes=sum(t.byte_size for t in specs.values()),
                quant=quant,
            )
        if path_lower.endswith(".safetensors"):
            parsed_st = read_safetensors_header_via_range(_url_range)
            quant = QuantSpec(method="none")
            raw_map = {k: (v, os.path.basename(path_lower) or "model.safetensors") for k, v in parsed_st.tensors.items()}
            specs = build_safetensors_specs(raw_map, quant)
            return CheckpointMeta(
                source=target,
                container_format="safetensors",
                tensors=specs,
                shard_files=(os.path.basename(path_lower) or "model.safetensors",),
                safetensors_metadata=parsed_st.metadata,
                header_bytes_fetched=fetched_total,
                total_weight_bytes=sum(t.byte_size for t in specs.values()),
                quant=quant,
            )
        raise CheckpointLoadError(f"Unsupported direct checkpoint URL (expected .safetensors or .gguf): {target}")

    if hf_parsed is None:
        raise CheckpointLoadError(
            f"Target {target!r} is neither an existing local path nor a valid Hugging Face 'org/model' ID or URL."
        )

    base_url, single_subpath = hf_parsed
    bytes_fetched = 0

    def _fetch_opt_json(rel_path: str) -> dict[str, Any] | None:
        nonlocal bytes_fetched
        url = f"{base_url}/{rel_path}"
        try:
            raw = _http_fetch_bytes(url, token=resolved_token, timeout=timeout)
            bytes_fetched += len(raw)
            obj = json.loads(raw.decode("utf-8"))
            return obj if isinstance(obj, dict) else None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (401, 403):
                raise CheckpointLoadError(
                    f"Repository or file not found or gated (HTTP {exc.code}) fetching {url}. "
                    "Verify the repo ID or set HF_TOKEN for gated models."
                ) from exc
            raise CheckpointLoadError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise CheckpointLoadError(f"Network error fetching {url}: {exc.reason}") from exc

    def _read_remote_gguf(part_name: str) -> ParsedGGUFHeader:
        part_url = f"{base_url}/{part_name}"

        def _gguf_range(start: int, end: int) -> bytes:
            nonlocal bytes_fetched
            chunk = _http_fetch_bytes(part_url, byte_range=(start, end), token=resolved_token, timeout=timeout)
            bytes_fetched += len(chunk)
            return chunk

        try:
            return read_gguf_header_via_range(_gguf_range)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(part_url) from exc
            raise

    if single_subpath and single_subpath.lower().endswith(".gguf"):
        parsed_gguf, found_parts, missing_parts = _read_gguf_parts(single_subpath, _read_remote_gguf)
        config, quant, specs = normalize_gguf_metadata(parsed_gguf)
        return CheckpointMeta(
            source=target,
            container_format="gguf",
            config=config,
            tokenizer_meta=_extract_tokenizer_meta(None, {}, parsed_gguf.metadata),
            tensors=specs,
            shard_files=found_parts,
            missing_shards=missing_parts,
            gguf_metadata=parsed_gguf.metadata,
            header_bytes_fetched=bytes_fetched,
            total_weight_bytes=sum(t.byte_size for t in specs.values()),
            quant=quant,
        )

    # Fetch JSON configs concurrently
    json_targets = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "hf_quant_config.json",
        "quantize_config.json",
    ]
    fetched_jsons: dict[str, dict[str, Any] | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fut_map = {pool.submit(_fetch_opt_json, rel): rel for rel in json_targets}
        for fut in concurrent.futures.as_completed(fut_map):
            fetched_jsons[fut_map[fut]] = fut.result()

    config = fetched_jsons.get("config.json") or {}
    if not config.get("quantization_config"):
        if fetched_jsons.get("hf_quant_config.json"):
            config["quantization_config"] = fetched_jsons["hf_quant_config.json"]
        elif fetched_jsons.get("quantize_config.json"):
            config["quantization_config"] = fetched_jsons["quantize_config.json"]

    gen_config = fetched_jsons.get("generation_config.json") or {}
    tok_config = fetched_jsons.get("tokenizer_config.json") or {}
    idx_json = fetched_jsons.get("model.safetensors.index.json")

    # Only fetch large tokenizer.json if tokenizer_config didn't provide added_tokens_decoder
    tok_json = None
    if not (tok_config and tok_config.get("added_tokens_decoder")):
        try:
            tok_json = _fetch_opt_json("tokenizer.json")
        except (CheckpointLoadError, ValueError):
            # The tokenizer only adds to the token checks, so it shouldn't stop a scan
            tok_json = None

    parts = target.strip("/").split("/")
    remote_siblings: list[str] = []
    if len(parts) == 2 and not target.startswith(("http://", "https://")):
        api_url = f"https://huggingface.co/api/models/{parts[0]}/{parts[1]}"
        try:
            raw_api = _http_fetch_bytes(api_url, token=resolved_token, timeout=timeout)
            api_data = json.loads(raw_api.decode("utf-8"))
            for sib in api_data.get("siblings", []):
                fn = str(sib.get("rfilename") or "")
                if fn:
                    remote_siblings.append(fn)
        except Exception:
            pass

    def _load_repo_gguf() -> CheckpointMeta | None:
        gguf_candidates = [fn for fn in remote_siblings if fn.lower().endswith(".gguf")]
        if not gguf_candidates:
            return None
        # Auto-select the standard Q4_K_M or Q4_0 or first GGUF file
        chosen = next((f for f in gguf_candidates if "q4_k_m" in f.lower()), None)
        if chosen is None:
            chosen = next((f for f in gguf_candidates if "q4" in f.lower()), gguf_candidates[0])
        parsed_gguf, found_parts, missing_parts = _read_gguf_parts(chosen, _read_remote_gguf)
        g_config, g_quant, g_specs = normalize_gguf_metadata(parsed_gguf)
        return CheckpointMeta(
            source=f"{target} ({chosen})",
            container_format="gguf",
            config=g_config,
            tokenizer_meta=_extract_tokenizer_meta(None, {}, parsed_gguf.metadata),
            tensors=g_specs,
            shard_files=found_parts,
            missing_shards=missing_parts,
            gguf_metadata=parsed_gguf.metadata,
            header_bytes_fetched=bytes_fetched,
            total_weight_bytes=sum(t.byte_size for t in g_specs.values()),
            quant=g_quant,
        )

    if prefer_gguf and not single_subpath:
        preferred = _load_repo_gguf()
        if preferred is not None:
            return preferred

    expected_weight_map: dict[str, str] | None = None
    index_total_size: int | None = None
    if single_subpath and single_subpath.lower().endswith(".safetensors"):
        shard_names = [single_subpath]
    elif idx_json is not None:
        expected_weight_map, index_total_size = parse_safetensors_index(idx_json)
        shard_names = sorted(set(expected_weight_map.values()))
    else:
        q_base = config.get("quantization_config", {}).get("model_file_base_name")
        if q_base and f"{q_base}.safetensors" in remote_siblings:
            shard_names = [f"{q_base}.safetensors"]
        elif "model.safetensors" in remote_siblings:
            shard_names = ["model.safetensors"]
        else:
            st_files = [fn for fn in remote_siblings if fn.lower().endswith(".safetensors")]
            shard_names = st_files if st_files else ["model.safetensors"]

    def _fetch_one_shard(sname: str) -> tuple[str, ParsedSafetensorsHeader | None, int]:
        shard_url = f"{base_url}/{sname}"
        local_bytes = 0

        def _srange(start: int, end: int) -> bytes:
            nonlocal local_bytes
            chunk = _http_fetch_bytes(
                shard_url, byte_range=(start, end), token=resolved_token, timeout=timeout
            )
            local_bytes += len(chunk)
            return chunk

        try:
            parsed = read_safetensors_header_via_range(_srange)
            return sname, parsed, local_bytes
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return sname, None, local_bytes
            raise CheckpointLoadError(f"HTTP {exc.code} reading shard {shard_url}") from exc

    missing_shards: list[str] = []
    duplicate_tensors: list[str] = []
    raw_entries: dict[str, tuple[RawSafetensorEntry, str]] = {}
    merged_meta: dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(shard_names)))) as pool:
        results = list(pool.map(_fetch_one_shard, shard_names))

    for sname, parsed_hdr, shard_hdr_bytes in results:
        bytes_fetched += shard_hdr_bytes
        if parsed_hdr is None:
            missing_shards.append(sname)
            continue
        merged_meta.update(parsed_hdr.metadata)
        for tname, entry in parsed_hdr.tensors.items():
            # Without an index, several files are usually variants of one model
            if tname in raw_entries and expected_weight_map is not None:
                duplicate_tensors.append(tname)
            raw_entries[tname] = (entry, sname)

    remote_legacy = _remote_legacy_weight_files(
        remote_siblings, base_url=base_url, token=resolved_token, timeout=timeout
    )
    remote_code = [
        fn for fn in remote_siblings
        if fn.lower().endswith(".py")
    ]

    if not raw_entries and missing_shards == ["model.safetensors"] and idx_json is None:
        gguf_meta = _load_repo_gguf()
        if gguf_meta is not None:
            return gguf_meta

        if remote_legacy:
            return CheckpointMeta(
                source=target,
                container_format="pytorch_bin",
                config=config,
                generation_config=gen_config,
                tokenizer_config=tok_config,
                tokenizer_meta=_extract_tokenizer_meta(tok_json, tok_config, config_vocab_size=config.get("vocab_size")),
                tensors={},
                shard_files=tuple(sorted(remote_legacy)),
                legacy_weight_files=tuple(sorted(remote_legacy)),
                custom_code_files=tuple(sorted(remote_code)),
                header_bytes_fetched=bytes_fetched,
                quant=parse_quant_spec_from_config(config),
            )

        raise CheckpointLoadError(
            f"No model.safetensors or model.safetensors.index.json found in {target}. "
            "If this is a GGUF-only repo, pass the .gguf file URL or subpath."
        )

    dangling: list[str] = []
    if expected_weight_map is not None:
        for tname, sname in expected_weight_map.items():
            if sname not in missing_shards and tname not in raw_entries:
                dangling.append(tname)

    quant = parse_quant_spec_from_config(config)
    specs = build_safetensors_specs(raw_entries, quant)
    calc_bytes = sum(t.byte_size for t in specs.values())
    total_bytes = index_total_size if (index_total_size and missing_shards) else calc_bytes

    return CheckpointMeta(
        source=target,
        container_format="safetensors",
        config=config,
        generation_config=gen_config,
        tokenizer_config=tok_config,
        tokenizer_meta=_extract_tokenizer_meta(tok_json, tok_config, config_vocab_size=config.get("vocab_size")),
        tensors=specs,
        shard_files=tuple(shard_names),
        missing_shards=tuple(missing_shards),
        dangling_index_tensors=tuple(sorted(dangling)),
        duplicate_tensors=tuple(sorted(set(duplicate_tensors))),
        legacy_weight_files=tuple(sorted(remote_legacy)),
        custom_code_files=tuple(sorted(remote_code)),
        safetensors_metadata=merged_meta,
        header_bytes_fetched=bytes_fetched,
        total_weight_bytes=total_bytes,
        quant=quant,
    )
