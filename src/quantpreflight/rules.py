"""Core zero-GPU pre-flight rule engine."""

from __future__ import annotations

import math
import re
from typing import Any

from .models import CheckpointMeta, Finding, QuantSpec, ScanReport, Severity, TargetSpec, TensorSpec
from .profiles import (
    ArchitectureProfile,
    EngineProfile,
    GPUProfile,
    load_advisories,
    load_architecture_profiles,
    load_engine_profiles,
    load_gpu_profiles,
    resolve_architecture_profile,
    resolve_gpu_profile,
)
from .safetensors_io import SAFETENSORS_DTYPE_BITS

MAX_CONFIG_INT = 10**12  # Keeps absurd config values out of the VRAM arithmetic

# Fitted to real vLLM 0.33 loads on Tesla T4. See benchmark/gpu_check_t4.json.
WEIGHT_OVERHEAD = 1.18
RUNTIME_GIB = 0.65

RULE_CATALOG: dict[str, tuple[Severity, str, str]] = {
    "QPF001": (
        Severity.CRITICAL,
        "Bit-starved sensitive layer",
        "Sensitive routing, recurrent state, normalization, or vision layers are quantized below safe precision.",
    ),
    "QPF002": (
        Severity.CRITICAL,
        "Tokenizer or vocabulary ID out of embedding bounds",
        "Token IDs or vocab_size exceed embed_tokens or lm_head row bounds, causing CUDA device-side asserts.",
    ),
    "QPF003": (
        Severity.MEDIUM,
        "Stop-token (eos_token_id) drift across configs",
        "Chat or generation stop-token IDs disagree across config.json, generation_config.json, and tokenizer.",
    ),
    "QPF004": (
        Severity.CRITICAL,
        "GPU hardware quantization incompatibility",
        "Target GPU compute capability lacks native kernel or tensor-core support for this quantization format.",
    ),
    "QPF005": (
        Severity.CRITICAL,
        "Silent kernel corruption or ignored dtype dispatch",
        "Known engine + GPU + quantization combination silently corrupts output or ignores dtype flags.",
    ),
    "QPF006": (
        Severity.CRITICAL,
        "Tensor-parallel head or expert divisibility mismatch",
        "Attention heads, KV heads, or MoE experts cannot be evenly partitioned across TP ranks.",
    ),
    "QPF007": (
        Severity.HIGH,
        "Quantized tensor group-size or Marlin tile alignment mismatch",
        "Post-TP sharded linear dimensions violate group_size, Marlin (128x64), or FP8 block alignment.",
    ),
    "QPF008": (
        Severity.CRITICAL,
        "GPTQ desc_act (act_order) is rejected or slow",
        "vLLM rejects GPTQ checkpoints with desc_act=true. Other engines pay a gather cost across TP ranks.",
    ),
    "QPF009": (
        Severity.CRITICAL,
        "Broken shard index or speculative MTP projection shape mismatch",
        "Shard files or tensors declared in index.json are missing, duplicated, or incompatible with hidden_size.",
    ),
    "QPF010": (
        Severity.HIGH,
        "Quantization config vs. tensor header contradiction",
        "config.json quantization metadata contradicts actual Safetensors weight dtypes.",
    ),
    "QPF011": (
        Severity.HIGH,
        "Serving engine container or quant format mismatch",
        "Checkpoint container or sub-4-bit quant scheme is unsupported by the target serving engine.",
    ),
    "QPF012": (
        Severity.CRITICAL,
        "Pre-flight GPU VRAM budget exceeded",
        "Per-rank weight footprint plus minimal KV cache exceeds available GPU memory.",
    ),
    "QPF013": (
        Severity.CRITICAL,
        "SafeTensors span integrity or data offset corruption",
        "Tensor data offset spans overlap within a shard, contradict declared tensor shapes, or exceed file size.",
    ),
    "QPF014": (
        Severity.CRITICAL,
        "Insecure legacy weight format or unpinned custom modeling code",
        "Checkpoint uses legacy pickle serialization (arbitrary code execution hazard) or requires unvetted trust_remote_code.",
    ),
    "QPF015": (
        Severity.HIGH,
        "Engine version with published security advisories",
        "The engine version passed with --engine-version falls inside the affected range of known advisories.",
    ),
    "QPF016": (
        Severity.CRITICAL,
        "Context length longer than the model allows",
        "vLLM refuses a --context-len above the model's max_position_embeddings.",
    ),
}


def _effective_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return nested `text_config` merged with top-level `config` for multimodal checkpoints."""
    sub = config.get("text_config")
    if isinstance(sub, dict):
        merged = dict(config)
        for k, v in sub.items():
            merged.setdefault(k, v)
        # Prefer text_config head/hidden dimensions when present
        for dim_key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "head_dim",
        ):
            if dim_key in sub:
                merged[dim_key] = sub[dim_key]
        return merged
    return config


def _to_int_set(val: Any) -> set[int]:
    if isinstance(val, int) and not isinstance(val, bool):
        return {val}
    if isinstance(val, list):
        return {x for x in val if isinstance(x, int) and not isinstance(x, bool)}
    return set()


def _check_qpf001_sensitive_layers(
    meta: CheckpointMeta,
    arch_profile: ArchitectureProfile,
) -> list[Finding]:
    grouped: dict[str, list[tuple[TensorSpec, float, str, str]]] = {}

    for tspec in meta.tensors.values():
        best_rule = None
        for srule in arch_profile.sensitive_layers:
            if not srule.compiled_regex.search(tspec.name):
                continue
            if srule.compiled_exclude and srule.compiled_exclude.search(tspec.name):
                continue
            if best_rule is None or srule.min_bits > best_rule.min_bits:
                best_rule = srule

        if best_rule is not None and tspec.effective_bits < best_rule.min_bits:
            grouped.setdefault(best_rule.name, []).append(
                (tspec, best_rule.min_bits, best_rule.severity, best_rule.reason)
            )

    findings: list[Finding] = []
    for rule_name, items in sorted(grouped.items()):
        items.sort(key=lambda x: x[0].name)
        sample_tspec, min_bits, sev_str, reason = items[0]
        severity = Severity(sev_str)
        examples = ", ".join(
            f"{t.name} ({t.storage_dtype}, {t.effective_bits:g}-bit)" for t, _, _, _ in items[:4]
        )
        more = f" (+{len(items) - 4} more)" if len(items) > 4 else ""
        findings.append(
            Finding(
                rule_id="QPF001",
                severity=severity,
                title=f"Bit-starved sensitive layer ({rule_name})",
                message=(
                    f"{len(items)} sensitive tensor(s) matched '{rule_name}' with effective precision "
                    f"below {min_bits:g}-bit: {examples}{more}. {reason}"
                ),
                remediation=(
                    f"Add matching layer patterns for '{rule_name}' to your quantizer 'ignore' / "
                    f"'modules_to_not_convert' list so they remain in BF16/FP16/FP32 (>={min_bits:g}-bit)."
                ),
                tensors=tuple(t.name for t, _, _, _ in items[:16]),
                location=sample_tspec.shard,
            )
        )
    return findings


def _matches_module(name: str, modules: tuple[str, ...]) -> bool:
    """Match whole dotted parts, so GGUF's `attn_output.weight` is not `output.weight`."""
    return name in modules or name.endswith(tuple(f".{module}" for module in modules))


def _find_embedding_and_head(meta: CheckpointMeta) -> tuple[TensorSpec | None, TensorSpec | None]:
    embed: TensorSpec | None = None
    lm_head: TensorSpec | None = None
    embed_suffixes = (
        "model.embed_tokens.weight",
        "model.embed_tokens.qweight",
        "model.embed_tokens.weight_packed",
        "transformer.wte.weight",
        "tok_embeddings.weight",
        "token_embd.weight",
        "embed_tokens.weight",
    )
    head_suffixes = (
        "lm_head.weight",
        "lm_head.qweight",
        "lm_head.weight_packed",
        "output.weight",
    )
    for name, spec in meta.tensors.items():
        if embed is None and _matches_module(name, embed_suffixes):
            embed = spec
        if lm_head is None and _matches_module(name, head_suffixes):
            lm_head = spec
    return embed, lm_head


def _check_qpf002_vocab_bounds(meta: CheckpointMeta, tcfg: dict[str, Any]) -> list[Finding]:
    embed, lm_head = _find_embedding_and_head(meta)
    if embed is None or len(embed.unpacked_shape) < 1:
        return []

    embed_rows = embed.unpacked_shape[0]
    findings: list[Finding] = []
    tmeta = meta.tokenizer_meta

    if tmeta.max_token_id is not None and tmeta.max_token_id >= embed_rows:
        findings.append(
            Finding(
                rule_id="QPF002",
                severity=Severity.CRITICAL,
                title="Tokenizer ID exceeds embedding table row bounds",
                message=(
                    f"Tokenizer defines token {tmeta.max_token_str!r} at ID {tmeta.max_token_id}, "
                    f"but {embed.name} has shape {list(embed.unpacked_shape)} "
                    f"(valid token indices 0..{embed_rows - 1}). Sampling or encoding this token "
                    f"triggers an unrecoverable GPU CUDA device-side assert."
                ),
                remediation=(
                    f"Resize token embeddings (`model.resize_token_embeddings({tmeta.max_token_id + 1})`) "
                    f"before quantizing, or remove out-of-range added_tokens from tokenizer.json."
                ),
                tensors=(embed.name,),
                location="tokenizer.json",
                references=("https://osv.dev/vulnerability/CVE-2026-115916",),
            )
        )

    cfg_vocab = tcfg.get("vocab_size")
    if isinstance(cfg_vocab, int) and cfg_vocab > embed_rows:
        findings.append(
            Finding(
                rule_id="QPF002",
                severity=Severity.CRITICAL,
                title="config.json vocab_size exceeds embedding rows",
                message=(
                    f"config.json declares vocab_size={cfg_vocab}, which is larger than "
                    f"{embed.name} row count ({embed_rows})."
                ),
                remediation=(
                    f"Align config.json vocab_size with {embed.name} shape[0] ({embed_rows}) "
                    "or re-export the checkpoint with resized embeddings."
                ),
                tensors=(embed.name,),
                location="config.json",
                references=("https://osv.dev/vulnerability/CVE-2026-115916",),
            )
        )

    for key_name, src_dict, src_file in (
        ("eos_token_id", meta.config, "config.json"),
        ("bos_token_id", meta.config, "config.json"),
        ("pad_token_id", meta.config, "config.json"),
        ("eos_token_id", meta.generation_config, "generation_config.json"),
    ):
        ids = _to_int_set(src_dict.get(key_name))
        if key_name == "pad_token_id":
            # Many configs use -1 for "no padding token"
            ids.discard(-1)
        bad_ids = sorted(i for i in ids if i >= embed_rows or i < 0)
        if bad_ids:
            findings.append(
                Finding(
                    rule_id="QPF002",
                    severity=Severity.CRITICAL,
                    title=f"Out-of-bounds {key_name} in {src_file}",
                    message=(
                        f"{src_file} sets {key_name}={bad_ids}, outside {embed.name} "
                        f"valid row range 0..{embed_rows - 1}."
                    ),
                    remediation=f"Update {key_name} in {src_file} to a valid token ID < {embed_rows}.",
                    tensors=(embed.name,),
                    location=src_file,
                )
            )

    if (
        lm_head is not None
        and len(lm_head.unpacked_shape) >= 1
        and tmeta.max_token_id is not None
        and tmeta.max_token_id >= lm_head.unpacked_shape[0]
    ):
        findings.append(
            Finding(
                rule_id="QPF002",
                severity=Severity.CRITICAL,
                title="Tokenizer ID exceeds lm_head output vocabulary rows",
                message=(
                    f"Tokenizer max token ID {tmeta.max_token_id} ({tmeta.max_token_str!r}) "
                    f"exceeds {lm_head.name} row count ({lm_head.unpacked_shape[0]})."
                ),
                remediation=f"Resize {lm_head.name} to at least {tmeta.max_token_id + 1} rows.",
                tensors=(lm_head.name,),
                location=lm_head.shard,
            )
        )

    return findings


def _check_qpf003_eos_drift(meta: CheckpointMeta) -> list[Finding]:
    cfg_eos = _to_int_set(meta.config.get("eos_token_id"))
    gen_eos = _to_int_set(meta.generation_config.get("eos_token_id"))
    tok_eos = set(meta.tokenizer_meta.eos_token_ids)

    active_stop_set = gen_eos if gen_eos else cfg_eos
    findings: list[Finding] = []

    if tok_eos and active_stop_set and not tok_eos.issubset(active_stop_set):
        missing = sorted(tok_eos - active_stop_set)
        target_file = "generation_config.json" if gen_eos else "config.json"
        findings.append(
            Finding(
                rule_id="QPF003",
                severity=Severity.MEDIUM,
                title="Chat stop token missing from generation eos_token_id",
                message=(
                    f"tokenizer_config.json sets eos_token={meta.tokenizer_meta.eos_token!r} "
                    f"(token ID {missing}), which is absent from {target_file} "
                    f"eos_token_id={sorted(active_stop_set)}. vLLM also stops on the tokenizer's token, but "
                    f"other clients that read only this file may generate until max_tokens."
                ),
                remediation=(
                    f"Include token ID(s) {missing} in {target_file} 'eos_token_id' list "
                    f"(for example: {sorted(active_stop_set | tok_eos)})."
                ),
                location=target_file,
            )
        )
    elif cfg_eos and gen_eos and cfg_eos.isdisjoint(gen_eos):
        findings.append(
            Finding(
                rule_id="QPF003",
                severity=Severity.MEDIUM,
                title="Disjoint eos_token_id sets between config.json and generation_config.json",
                message=(
                    f"config.json eos_token_id={sorted(cfg_eos)} and generation_config.json "
                    f"eos_token_id={sorted(gen_eos)} share no common stop token IDs."
                ),
                remediation="Include all valid stop token IDs in generation_config.json 'eos_token_id'.",
                location="generation_config.json",
            )
        )
    return findings


def _is_modelopt(quant: QuantSpec) -> bool:
    raw = quant.raw_config
    return "quant_algo" in raw or "quantization" in raw or str(raw.get("quant_method")).lower() == "modelopt"


def _check_qpf004_gpu_compatibility(
    meta: CheckpointMeta,
    gpu: GPUProfile | None,
    engine: EngineProfile,
) -> list[Finding]:
    if gpu is None:
        return []

    findings: list[Finding] = []
    q = meta.quant
    dtypes = {t.storage_dtype for t in meta.tensors.values() if len(t.shape) >= 2}

    if not gpu.supports_bf16:
        cfg_dtype = str(meta.config.get("torch_dtype") or meta.config.get("dtype") or "").lower()
        if ("BF16" in dtypes and q.method == "none") or (q.method != "none" and cfg_dtype == "bfloat16"):
            findings.append(
                Finding(
                    rule_id="QPF004",
                    severity=Severity.MEDIUM,
                    title=f"BF16 runs as FP16 on {gpu.id} ({gpu.arch})",
                    message=(
                        f"Checkpoint uses BF16 weights or activations, but {gpu.id} ({gpu.arch}) has no "
                        f"native BFloat16 support, so the engine casts them to FP16 at load."
                    ),
                    remediation="It should still load. Check output quality, or deploy on sm_80+ (A100/L4/H100) to keep BF16.",
                    location="config.json",
                )
            )

    if q.method == "nvfp4" and not gpu.supports_nvfp4:
        findings.append(
            Finding(
                rule_id="QPF004",
                severity=Severity.CRITICAL,
                title=f"NVFP4 requires Blackwell (sm_100+), target is {gpu.id} ({gpu.arch})",
                message=(
                    f"Checkpoint uses NVFP4 4-bit floating-point quantization, which is not supported "
                    f"on {gpu.id} ({gpu.arch})."
                ),
                remediation="Deploy NVFP4 checkpoints on Blackwell GPUs (B200/GB10/RTX5090) or use AWQ/GPTQ/FP8.",
                location="config.json",
            )
        )

    if engine.id == "vllm" and _is_modelopt(q) and gpu.vendor == "nvidia" and gpu.compute_capability < 80:
        findings.append(
            Finding(
                rule_id="QPF004",
                severity=Severity.CRITICAL,
                title=f"ModelOpt checkpoints need sm_80 or newer, target is {gpu.id} ({gpu.arch})",
                message=f"vLLM rejects ModelOpt quantization on {gpu.id} ({gpu.arch}), below its minimum of compute capability 80.",
                remediation="Deploy on sm_80 or newer, or use a Compressed-Tensors, AWQ, or GPTQ checkpoint.",
                location="config.json",
            )
        )

    is_fp8_ckpt = q.method == "fp8" or q.activation_dtype == "fp8" or ("F8_E4M3" in dtypes and q.method != "none")
    if is_fp8_ckpt:
        if q.weight_block_size is not None and gpu.arch.lower() in engine.fp8_block_restricted_arches:
            findings.append(
                Finding(
                    rule_id="QPF004",
                    severity=Severity.HIGH,
                    title=f"Block-scaled FP8 kernel dispatch restriction on {gpu.id} ({gpu.arch})",
                    message=(
                        f"Checkpoint uses block-scaled FP8 (weight_block_size={list(q.weight_block_size)}), "
                        f"which hits CUTLASS block-FP8 dispatch restrictions on {gpu.id} ({gpu.arch}) in "
                        f"{engine.name}, causing startup rejection or slow Triton fallback."
                    ),
                    remediation=(
                        "Deploy block-scaled FP8 on Hopper/Blackwell (H100/H200/B200) or MI300X, "
                        "or use per-tensor FP8 / W4A16 AWQ on Ada Lovelace (L4/L40S/RTX4090)."
                    ),
                    location="config.json",
                    references=("https://github.com/vllm-project/vllm/issues/58241",),
                )
            )
        elif not gpu.supports_fp8_w8a8:
            if not gpu.supports_marlin or q.activation_dtype == "fp8" or q.weight_block_size is not None:
                findings.append(
                    Finding(
                        rule_id="QPF004",
                        severity=Severity.HIGH,
                        title=f"No FP8 cores on {gpu.id} ({gpu.arch}), so FP8 may fail to start",
                        message=(
                            f"Checkpoint uses FP8 activations or blocks, but {gpu.id} ({gpu.arch}) has no native "
                            f"FP8 tensor cores (those need sm_89 or gfx942). On T4, some FP8 models loaded and others crashed."
                        ),
                        remediation=(
                            f"Use W4A16 (AWQ/GPTQ) or BF16/FP16 on {gpu.id}, or move FP8 serving to L40S/H100/MI300X."
                        ),
                        location="config.json",
                    )
                )

    if q.format_version == "marlin" and not gpu.supports_marlin:
        findings.append(
            Finding(
                rule_id="QPF004",
                severity=Severity.CRITICAL,
                title=f"Pre-packed Marlin checkpoint unsupported on {gpu.id} ({gpu.arch})",
                message=(
                    f"Checkpoint is formatted for NVIDIA Marlin kernels, which cannot execute on "
                    f"{gpu.id} ({gpu.vendor} {gpu.arch})."
                ),
                remediation="Deploy on NVIDIA Ampere or newer (A100, L40S, H100), or use standard unpacked GPTQ/AWQ.",
                location="config.json",
            )
        )

    return findings


def _check_qpf005_kernel_corruptions(
    meta: CheckpointMeta,
    target: TargetSpec,
    gpu: GPUProfile | None,
) -> list[Finding]:
    findings: list[Finding] = []
    q = meta.quant
    marlin_in = (target.marlin_input_dtype or "").strip().lower()

    if (
        target.engine == "vllm"
        and q.bits == 4.0
        and q.method in ("awq", "gptq", "compressed-tensors", "autoround")
        and gpu is not None
        and gpu.compute_capability in (120, 121)
    ):
        if marlin_in == "fp8":
            findings.append(
                Finding(
                    rule_id="QPF005",
                    severity=Severity.CRITICAL,
                    title=f"Silent Marlin W4A8-FP8 output corruption on {gpu.id} ({gpu.arch})",
                    message=(
                        f"Enabling VLLM_MARLIN_INPUT_DTYPE=fp8 on {gpu.id} ({gpu.arch}) with 4-bit "
                        f"{q.method.upper()} weights triggers silent numerical corruption (such as "
                        f"infinite '</think>' loops at temperature=0)."
                    ),
                    remediation="Unset VLLM_MARLIN_INPUT_DTYPE (keep FP16/BF16 Marlin inputs) on sm_120/sm_121 GPUs.",
                    location="config.json",
                    references=("https://github.com/vllm-project/vllm/issues/49546",),
                )
            )
        elif any("experts" in t for t in meta.tensors):
            findings.append(
                Finding(
                    rule_id="QPF005",
                    severity=Severity.MEDIUM,
                    title=f"Avoid VLLM_MARLIN_INPUT_DTYPE=fp8 for 4-bit MoE on {gpu.id} ({gpu.arch})",
                    message=(
                        f"4-bit MoE checkpoint targeted at {gpu.id} ({gpu.arch}) on vLLM: keep default "
                        f"16-bit Marlin activations and do not set VLLM_MARLIN_INPUT_DTYPE=fp8."
                    ),
                    remediation="Serve with default 16-bit Marlin activations on GB10 / sm_121.",
                    location="config.json",
                    references=("https://github.com/vllm-project/vllm/issues/49546",),
                )
            )

    if target.engine == "vllm" and q.method in ("autoround", "compressed-tensors") and marlin_in == "int8":
        findings.append(
            Finding(
                rule_id="QPF005",
                severity=Severity.HIGH,
                title=f"VLLM_MARLIN_INPUT_DTYPE=int8 silently ignored on {q.method} checkpoints",
                message=(
                    f"Checkpoint uses {q.method} quantization, where vLLM silently ignores "
                    "VLLM_MARLIN_INPUT_DTYPE=int8 instead of dispatching INT8 Marlin kernels."
                ),
                remediation="Convert checkpoint with llm-compressor W4A8-INT8 or omit VLLM_MARLIN_INPUT_DTYPE=int8.",
                location="config.json",
                references=("https://github.com/vllm-project/vllm/issues/48904",),
            )
        )

    return findings


def _check_qpf006_tp_divisibility(tcfg: dict[str, Any], target: TargetSpec) -> list[Finding]:
    tp = target.tp
    if tp <= 1:
        return []

    findings: list[Finding] = []
    q_heads = tcfg.get("num_attention_heads")
    kv_heads = tcfg.get("num_key_value_heads")
    if kv_heads is None:
        kv_heads = q_heads

    if isinstance(q_heads, int) and q_heads > 0 and q_heads % tp != 0:
        findings.append(
            Finding(
                rule_id="QPF006",
                severity=Severity.CRITICAL,
                title="num_attention_heads not divisible by TP degree",
                message=f"num_attention_heads ({q_heads}) is not divisible by tp={tp} ({q_heads} % {tp} != 0).",
                remediation=f"Choose a TP degree that divides {q_heads} evenly (e.g., {_valid_divisors(q_heads)}).",
                location="config.json",
            )
        )

    if isinstance(kv_heads, int) and kv_heads > 0:
        if kv_heads >= tp and kv_heads % tp != 0:
            findings.append(
                Finding(
                    rule_id="QPF006",
                    severity=Severity.CRITICAL,
                    title="GQA num_key_value_heads not divisible by TP degree",
                    message=(
                        f"num_key_value_heads ({kv_heads}) cannot be partitioned across tp={tp} "
                        f"({kv_heads} % {tp} != 0), crashing engine initialization."
                    ),
                    remediation=f"Set --tp to a divisor of {kv_heads} (such as {_valid_divisors(kv_heads)}).",
                    location="config.json",
                )
            )
        elif kv_heads < tp:
            if tp % kv_heads != 0:
                findings.append(
                    Finding(
                        rule_id="QPF006",
                        severity=Severity.CRITICAL,
                        title="TP degree is not a multiple of num_key_value_heads",
                        message=(
                            f"tp={tp} is larger than num_key_value_heads ({kv_heads}) and not an integer "
                            f"multiple ({tp} % {kv_heads} != 0), preventing KV-head replication."
                        ),
                        remediation=f"Use a TP degree dividing or multiplying {kv_heads} cleanly.",
                        location="config.json",
                    )
                )
            else:
                rep = tp // kv_heads
                findings.append(
                    Finding(
                        rule_id="QPF006",
                        severity=Severity.MEDIUM,
                        title="KV cache heads will be replicated across TP ranks",
                        message=(
                            f"tp={tp} exceeds num_key_value_heads={kv_heads}; KV heads will be replicated "
                            f"{rep}x across GPUs, increasing total KV-cache memory."
                        ),
                        remediation=f"Use tp<={kv_heads} if KV-cache capacity is tight.",
                        location="config.json",
                    )
                )

    experts = (
        tcfg.get("num_local_experts")
        or tcfg.get("n_routed_experts")
        or tcfg.get("num_experts")
    )
    if isinstance(experts, int) and experts > 1 and experts % tp != 0:
        findings.append(
            Finding(
                rule_id="QPF006",
                severity=Severity.HIGH,
                title="MoE expert count not divisible by TP degree",
                message=(
                    f"MoE checkpoint defines {experts} experts, which is not divisible by tp={tp} "
                    f"without expert-parallel padding or uneven routing."
                ),
                remediation=f"Use a TP/EP degree that divides {experts} ({_valid_divisors(experts)}).",
                location="config.json",
            )
        )

    return findings


def _valid_divisors(n: int) -> str:
    divs = [str(d) for d in (1, 2, 4, 8, 16) if n % d == 0]
    return ", ".join(divs) if divs else "1"


def _classify_parallel_axis(
    tensor_name: str,
    arch_profile: ArchitectureProfile,
) -> str | None:
    parts = tensor_name.split(".")
    if len(parts) < 2:
        return None
    module_name = parts[-2]
    if module_name in arch_profile.col_parallel_suffixes:
        return "col"
    if module_name in arch_profile.row_parallel_suffixes:
        return "row"
    return None


def _check_qpf007_alignment(
    meta: CheckpointMeta,
    arch_profile: ArchitectureProfile,
    target: TargetSpec,
    gpu: GPUProfile | None,
    engine: EngineProfile,
) -> list[Finding]:
    q = meta.quant
    if q.method in ("none", "gguf", "bitsandbytes"):
        return []

    tp = max(1, target.tp)
    check_marlin = (
        engine.marlin_in_align > 1
        and q.method in ("awq", "gptq", "compressed-tensors", "autoround")
        and q.bits in (4.0, 8.0)
        and "float" not in (q.format_version or "")
        and (gpu is None or gpu.supports_marlin)
    )

    tp_div_errors: list[tuple[str, str]] = []
    group_errors: list[tuple[str, str]] = []
    marlin_errors: list[tuple[str, str]] = []
    block_errors: list[tuple[str, str]] = []

    for tspec in meta.tensors.values():
        if tspec.effective_bits > 8.0 or len(tspec.unpacked_shape) != 2:
            continue
        if not tspec.name.endswith((".weight", ".qweight", ".weight_packed")):
            continue

        axis = _classify_parallel_axis(tspec.name, arch_profile)
        if axis is None:
            continue

        out_f, in_f = tspec.unpacked_shape
        if axis == "col":
            if tp > 1 and out_f % tp != 0:
                tp_div_errors.append((tspec.name, f"out_features={out_f} not divisible by tp={tp}"))
                continue
            out_s, in_s = out_f // tp, in_f
        else:
            if tp > 1 and in_f % tp != 0:
                tp_div_errors.append((tspec.name, f"in_features={in_f} not divisible by tp={tp}"))
                continue
            out_s, in_s = out_f, in_f // tp

        if axis == "row" and q.group_size and q.group_size > 0 and not q.desc_act:
            if in_s % q.group_size != 0:
                group_errors.append(
                    (tspec.name, f"sharded in_features={in_s} not divisible by group_size={q.group_size}")
                )

        if check_marlin:
            if in_s % engine.marlin_in_align != 0 or out_s % engine.marlin_out_align != 0:
                marlin_errors.append(
                    (
                        tspec.name,
                        f"sharded [out={out_s}, in={in_s}] violates Marlin {engine.marlin_out_align}x{engine.marlin_in_align} alignment",
                    )
                )

        if q.weight_block_size is not None:
            bh, bw = q.weight_block_size
            if out_s % bh != 0 or in_s % bw != 0:
                block_errors.append(
                    (tspec.name, f"sharded [out={out_s}, in={in_s}] not divisible by FP8 block {bh}x{bw}")
                )

    findings: list[Finding] = []
    if tp_div_errors:
        sample = "; ".join(f"{n} ({msg})" for n, msg in tp_div_errors[:3])
        findings.append(
            Finding(
                rule_id="QPF007",
                severity=Severity.CRITICAL,
                title="Quantized projection dimension not divisible by TP degree",
                message=f"{len(tp_div_errors)} quantized linear layer(s) cannot shard across tp={tp}: {sample}.",
                remediation=f"Reduce --tp so all sharded linear dimensions divide evenly.",
                tensors=tuple(n for n, _ in tp_div_errors[:12]),
            )
        )

    if group_errors:
        sample = "; ".join(f"{n} ({msg})" for n, msg in group_errors[:3])
        findings.append(
            Finding(
                rule_id="QPF007",
                severity=Severity.CRITICAL,
                title="Row-parallel sharded input dimension violates quantization group_size",
                message=(
                    f"{len(group_errors)} row-parallel tensor(s) break group_size={q.group_size} "
                    f"at tp={tp}: {sample}."
                ),
                remediation=f"Use a smaller TP degree or re-quantize with group_size=64 / channelwise scales.",
                tensors=tuple(n for n, _ in group_errors[:12]),
            )
        )

    if marlin_errors:
        if q.format_version == "marlin":
            sev = Severity.CRITICAL
        else:
            # vLLM picks a slower kernel per layer for Compressed-Tensors instead of failing
            sev = Severity.MEDIUM if q.method == "compressed-tensors" else Severity.HIGH
        sample = "; ".join(f"{n} ({msg})" for n, msg in marlin_errors[:3])
        findings.append(
            Finding(
                rule_id="QPF007",
                severity=sev,
                title="Sharded quantized dimensions violate Marlin kernel tile alignment",
                message=(
                    f"{len(marlin_errors)} quantized tensor(s) fail Marlin alignment "
                    f"(in_features % {engine.marlin_in_align} == 0, out_features % {engine.marlin_out_align} == 0) "
                    f"at tp={tp}: {sample}."
                ),
                remediation=(
                    f"Adjust --tp so per-rank dimensions stay multiples of {engine.marlin_in_align}x{engine.marlin_out_align}, "
                    f"or force a non-Marlin fallback kernel."
                ),
                tensors=tuple(n for n, _ in marlin_errors[:12]),
                references=("https://github.com/vllm-project/vllm/issues/62595",),
            )
        )

    if block_errors:
        sample = "; ".join(f"{n} ({msg})" for n, msg in block_errors[:3])
        findings.append(
            Finding(
                rule_id="QPF007",
                severity=Severity.HIGH,
                title="Sharded FP8 tensor dimensions violate weight_block_size",
                message=f"{len(block_errors)} tensor(s) do not align with FP8 block shape at tp={tp}: {sample}.",
                remediation="Select a TP degree that preserves 128-element FP8 block boundaries.",
                tensors=tuple(n for n, _ in block_errors[:12]),
            )
        )

    return findings


def _check_qpf008_desc_act(meta: CheckpointMeta, target: TargetSpec) -> list[Finding]:
    if meta.quant.method != "gptq" or not meta.quant.desc_act:
        return []
    if target.engine == "vllm":
        return [
            Finding(
                rule_id="QPF008",
                severity=Severity.CRITICAL,
                title="vLLM rejects GPTQ checkpoints with desc_act=true",
                message="Checkpoint sets desc_act=true (act_order). Current vLLM no longer supports group activation ordering.",
                remediation="Use a checkpoint quantized with desc_act=false, or an AWQ or Compressed-Tensors version.",
                location="config.json",
            )
        ]
    if target.tp > 1:
        return [
            Finding(
                rule_id="QPF008",
                severity=Severity.HIGH,
                title="GPTQ desc_act=true incurs activation gather overhead under TP > 1",
                message=(
                    f"Checkpoint sets desc_act=true (act_order) with tp={target.tp}. Input channels are "
                    f"permuted globally via g_idx, preventing clean local row-parallel partial sums and "
                    f"degrading multi-GPU throughput."
                ),
                remediation="Prefer desc_act=false or AWQ / Compressed-Tensors static groups when serving with TP > 1.",
                location="config.json",
            )
        ]
    return []


def _check_qpf009_shards_and_mtp(meta: CheckpointMeta, tcfg: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if meta.missing_shards and meta.container_format == "gguf":
        findings.append(
            Finding(
                rule_id="QPF009",
                severity=Severity.CRITICAL,
                title="Missing part(s) of a split GGUF file",
                message=(
                    f"This GGUF model is split, but {len(meta.missing_shards)} part(s) are missing: "
                    f"{', '.join(meta.missing_shards[:5])}."
                ),
                remediation="Download every `-NNNNN-of-NNNNN.gguf` part into the same folder.",
                location=meta.missing_shards[0],
            )
        )
    elif meta.missing_shards:
        findings.append(
            Finding(
                rule_id="QPF009",
                severity=Severity.CRITICAL,
                title="Missing Safetensors shard file(s) referenced by index.json",
                message=(
                    f"model.safetensors.index.json references {len(meta.missing_shards)} missing shard(s): "
                    f"{', '.join(meta.missing_shards[:5])}."
                ),
                remediation="Re-upload or finish downloading all shard files listed in model.safetensors.index.json.",
                location="model.safetensors.index.json",
            )
        )

    if meta.dangling_index_tensors:
        findings.append(
            Finding(
                rule_id="QPF009",
                severity=Severity.CRITICAL,
                title="Tensors declared in weight_map are missing from shard headers",
                message=(
                    f"{len(meta.dangling_index_tensors)} tensor(s) in model.safetensors.index.json do not "
                    f"exist inside their assigned shard headers (e.g. {', '.join(meta.dangling_index_tensors[:4])})."
                ),
                remediation="Regenerate model.safetensors.index.json to match the exported Safetensors shards.",
                tensors=meta.dangling_index_tensors[:12],
                location="model.safetensors.index.json",
            )
        )

    if meta.duplicate_tensors:
        findings.append(
            Finding(
                rule_id="QPF009",
                severity=Severity.MEDIUM,
                title="Duplicate tensor definitions across multiple shards",
                message=(
                    f"{len(meta.duplicate_tensors)} tensor name(s) appear in more than one shard: "
                    f"{', '.join(meta.duplicate_tensors[:4])}. The extra copies cost memory, and the "
                    f"loaded copy depends on shard order."
                ),
                remediation="Deduplicate overlapping shard exports before deployment.",
                tensors=meta.duplicate_tensors[:12],
                location="model.safetensors.index.json",
            )
        )

    hidden = tcfg.get("hidden_size")
    if isinstance(hidden, int) and hidden > 0:
        for name, spec in meta.tensors.items():
            if ("mtp" in name.lower() or "eagle" in name.lower()) and "pre_projection" in name.lower():
                if len(spec.unpacked_shape) == 2 and hidden not in spec.unpacked_shape and (2 * hidden) not in spec.unpacked_shape:
                    findings.append(
                        Finding(
                            rule_id="QPF009",
                            severity=Severity.CRITICAL,
                            title="Speculative MTP pre_projection shape mismatch with hidden_size",
                            message=(
                                f"Speculative tensor {name} has shape {list(spec.unpacked_shape)}, which does "
                                f"not match base model hidden_size={hidden} (or 2*hidden_size={2 * hidden})."
                            ),
                            remediation="Verify MTP draft/Eagle weights match the target base checkpoint hidden_size.",
                            tensors=(name,),
                            location=spec.shard,
                            references=("https://github.com/vllm-project/vllm/issues/62340",),
                        )
                    )
    return findings


def _check_qpf010_config_contradiction(meta: CheckpointMeta) -> list[Finding]:
    if meta.container_format != "safetensors" or not meta.tensors:
        return []

    q = meta.quant
    has_sub16 = any(t.effective_bits < 16.0 for t in meta.tensors.values())
    has_packed_names = any(
        t.name.endswith((".qweight", ".marlin_qweight", ".qzeros", ".weight_packed")) for t in meta.tensors.values()
    )
    has_fp8_linears = any(
        t.storage_dtype in ("F8_E4M3", "F8_E5M2") and len(t.shape) == 2 for t in meta.tensors.values()
    )

    if q.method == "none" and meta.config and (has_packed_names or has_fp8_linears):
        return [
            Finding(
                rule_id="QPF010",
                severity=Severity.CRITICAL,
                title="Quantized tensors present without quantization_config in config.json",
                message=(
                    "Safetensors headers contain packed INT4/INT8 (.qweight/.weight_packed) or FP8 linear "
                    "tensors, but config.json has no 'quantization_config' block."
                ),
                remediation="Restore the matching 'quantization_config' section in config.json.",
                location="config.json",
            )
        ]

    stale = not has_sub16 and not has_packed_names and not meta.missing_shards
    if q.method not in ("none", "bitsandbytes", "mlx") and stale:
        return [
            Finding(
                rule_id="QPF010",
                severity=Severity.HIGH,
                title="config.json declares quantization, but all tensors are >=16-bit",
                message=(
                    f"config.json specifies quant_method={q.method!r}, yet every tensor in the checkpoint "
                    f"headers is >=16-bit ({sorted({t.storage_dtype for t in meta.tensors.values()})}). "
                    f"Deployment will consume full unquantized VRAM or fail weight loading."
                ),
                remediation="Export the quantized weights properly or remove stale quantization_config from config.json.",
                location="config.json",
            )
        ]
    return []


def _check_qpf011_engine_format(meta: CheckpointMeta, engine: EngineProfile) -> list[Finding]:
    findings: list[Finding] = []
    if meta.container_format not in engine.supported_containers:
        findings.append(
            Finding(
                rule_id="QPF011",
                severity=Severity.CRITICAL,
                title=f"Container format {meta.container_format!r} unsupported by {engine.name}",
                message=(
                    f"{engine.name} supports {list(engine.supported_containers)} checkpoints, "
                    f"but target checkpoint is {meta.container_format!r}."
                ),
                remediation=(
                    "Use --engine llamacpp for .gguf deployments or convert/download a Safetensors checkpoint."
                    if meta.container_format == "gguf"
                    else "Convert the Safetensors checkpoint to .gguf via llama.cpp convert_hf_to_gguf.py."
                ),
                location=meta.shard_files[0] if meta.shard_files else "config.json",
            )
        )
        return findings

    if meta.quant.method not in engine.supported_quant_methods:
        findings.append(
            Finding(
                rule_id="QPF011",
                severity=Severity.HIGH,
                title=f"Quantization method {meta.quant.method!r} unsupported by {engine.name}",
                message=(
                    f"Checkpoint uses quantization method {meta.quant.method!r}, which is not in "
                    f"{engine.name}'s supported formats ({', '.join(engine.supported_quant_methods)})."
                ),
                remediation=f"Choose a supported engine (such as vLLM) or re-quantize to AWQ/FP8/Compressed-Tensors.",
                location="config.json",
            )
        )

    if meta.container_format == "gguf" and engine.id in ("vllm", "sglang"):
        iquants = sorted(
            {
                t.storage_dtype
                for t in meta.tensors.values()
                if t.storage_dtype.startswith(("IQ1_", "IQ2_", "TQ1_", "TQ2_"))
            }
        )
        if iquants:
            findings.append(
                Finding(
                    rule_id="QPF011",
                    severity=Severity.HIGH,
                    title=f"Sub-3-bit GGUF I-quant/ternary tensors ({', '.join(iquants)}) on {engine.name}",
                    message=(
                        f"GGUF checkpoint contains {iquants} tensors, which frequently fail or run "
                        f"via slow dequantization paths in {engine.name}."
                    ),
                    remediation="Serve 1-bit/2-bit GGUF I-quants with --engine llamacpp, or use Q4_K_M / AWQ on vLLM.",
                    location=meta.shard_files[0] if meta.shard_files else "gguf",
                )
            )
    return findings


def _pos_int(value: Any) -> int | None:
    """Return a config value as a sane positive integer, or `None` when it isn't one."""
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_CONFIG_INT:
        return value
    return None


def estimate_per_gpu_vram_gb(
    meta: CheckpointMeta,
    tcfg: dict[str, Any],
    target: TargetSpec,
) -> tuple[float, float, float]:
    """Return `(total_per_gpu_gib, weights_per_gpu_gib, kv_per_gpu_gib)` with architecture-aware KV cache."""
    tp = max(1, target.tp)
    weights_gib = (meta.total_weight_bytes / tp) / (1024**3)

    layers = _pos_int(tcfg.get("num_hidden_layers"))
    q_heads = _pos_int(tcfg.get("num_attention_heads"))
    kv_heads = _pos_int(tcfg.get("num_key_value_heads")) or q_heads
    hidden = _pos_int(tcfg.get("hidden_size"))
    head_dim = _pos_int(tcfg.get("head_dim"))
    if head_dim is None and hidden and q_heads:
        head_dim = (hidden // q_heads) or None

    ctx = target.context_len or min(_pos_int(tcfg.get("max_position_embeddings")) or 4096, 4096)

    kv_elem_bytes = 1 if meta.quant.activation_dtype == "fp8" else 2
    model_type = str(tcfg.get("model_type") or "").lower()
    kv_gib = 0.0

    # 1. DeepSeek MLA (Multi-Head Latent Attention)
    kv_lora_rank = _pos_int(tcfg.get("kv_lora_rank"))
    if kv_lora_rank and layers:
        mla_dim = kv_lora_rank + (_pos_int(tcfg.get("qk_rope_head_dim")) or 64)
        kv_bytes = layers * mla_dim * kv_elem_bytes * ctx * target.concurrency
        kv_gib = kv_bytes / (1024**3)

    # 2. Pure Mamba / Mamba-2 SSM
    elif model_type in ("mamba", "mamba2"):
        d_state = _pos_int(tcfg.get("state_size")) or _pos_int(tcfg.get("d_state")) or 16
        d_inner = _pos_int(tcfg.get("intermediate_size")) or ((hidden or 2048) * 2)
        ssm_bytes = (layers or 1) * d_inner * d_state * 2 * target.concurrency
        kv_gib = ssm_bytes / (1024**3)

    # 3. Standard MHA / GQA / SWA / Hybrid Attention
    elif layers and kv_heads and head_dim:
        num_layers = layers
        block_types = tcfg.get("layers_block_type")
        if isinstance(block_types, list) and block_types:
            attn_count = sum(1 for b in block_types if "attention" in str(b).lower())
            if attn_count > 0:
                num_layers = attn_count
        elif model_type == "jamba" or "attn_layer_period" in tcfg:
            period = _pos_int(tcfg.get("attn_layer_period")) or 8
            num_layers = max(1, math.ceil(num_layers / period))

        kv_heads_per_rank = max(1, math.ceil(kv_heads / tp))

        # Check sliding-window attention
        sw = _pos_int(tcfg.get("sliding_window")) or _pos_int(tcfg.get("sliding_window_size"))
        if sw:
            layer_types = tcfg.get("layer_types")
            pattern = _pos_int(tcfg.get("sliding_window_pattern"))
            if isinstance(layer_types, list) and layer_types:
                num_global = min(num_layers, sum(
                    1 for kind in layer_types
                    if isinstance(kind, str) and ("full" in kind.lower() or "global" in kind.lower())
                ))
            elif pattern:
                num_global = num_layers // pattern
            elif model_type == "gemma2":
                num_global = num_layers - num_layers // 2
            else:
                num_global = 0
            num_sliding = num_layers - num_global
            eff_ctx = (num_sliding * min(ctx, sw) + num_global * ctx) / num_layers
        else:
            eff_ctx = ctx

        kv_bytes = 2 * num_layers * kv_heads_per_rank * head_dim * kv_elem_bytes * eff_ctx * target.concurrency
        kv_gib = kv_bytes / (1024**3)

    total_gib = weights_gib * WEIGHT_OVERHEAD + RUNTIME_GIB + kv_gib
    return round(total_gib, 2), round(weights_gib, 2), round(kv_gib, 2)


def _check_qpf012_vram_budget(
    meta: CheckpointMeta,
    tcfg: dict[str, Any],
    target: TargetSpec,
    gpu: GPUProfile | None,
) -> tuple[float | None, list[Finding]]:
    if meta.total_weight_bytes <= 0:
        return None, []

    total_gib, weights_gib, kv_gib = estimate_per_gpu_vram_gb(meta, tcfg, target)
    avail_vram = target.vram_gb_override if target.vram_gb_override is not None else (gpu.vram_gb if gpu else None)
    if avail_vram is None:
        return total_gib, []

    usable_gib = round(avail_vram * target.gpu_memory_utilization, 2)
    if total_gib > usable_gib:
        gpu_label = gpu.id if gpu else f"{avail_vram:g} GiB GPU"
        room_gib = max(usable_gib - RUNTIME_GIB - kv_gib, 0.1)
        min_tp = max(1, math.ceil(weights_gib * target.tp * WEIGHT_OVERHEAD / room_gib))
        return total_gib, [
            Finding(
                rule_id="QPF012",
                severity=Severity.CRITICAL,
                title=f"Estimated per-GPU memory ({total_gib:.2f} GiB) exceeds {gpu_label} budget ({usable_gib:.2f} GiB)",
                message=(
                    f"At tp={target.tp}, model weights require {weights_gib:.2f} GiB/GPU plus "
                    f"{kv_gib:.2f} GiB KV cache for {target.concurrency} request{'' if target.concurrency == 1 else 's'} "
                    f"(with runtime overhead, {total_gib:.2f} GiB/GPU), "
                    f"exceeding {gpu_label} usable VRAM ({avail_vram:g} GiB * {target.gpu_memory_utilization:.2f} = {usable_gib:.2f} GiB)."
                ),
                remediation=f"Increase --tp to at least {min_tp}, lower --context-len, or use a smaller bit-width checkpoint.",
                location="config.json",
            )
        ]
    return total_gib, []


def _check_qpf013_safetensors_spans(meta: CheckpointMeta) -> list[Finding]:
    """Check Safetensors tensor byte spans for overlaps, truncation, and bounds violations."""
    if meta.container_format != "safetensors" or not meta.tensors:
        return []

    findings: list[Finding] = []
    by_shard: dict[str, list[TensorSpec]] = {}
    for t in meta.tensors.values():
        if t.data_offsets is not None:
            by_shard.setdefault(t.shard, []).append(t)

    for shard_name, shard_tensors in by_shard.items():
        shard_size = meta.shard_file_sizes.get(shard_name)

        # 1. Check individual tensor span sanity and expected byte sizes
        for t in shard_tensors:
            start, end = t.data_offsets  # type: ignore[misc]
            span_len = end - start
            if start < 0 or end < start:
                findings.append(
                    Finding(
                        rule_id="QPF013",
                        severity=Severity.CRITICAL,
                        title=f"Inverted or negative data offsets on {t.name} in {shard_name}",
                        message=f"Tensor {t.name!r} has invalid offsets [{start}, {end}] in shard {shard_name!r}.",
                        remediation="Re-export the checkpoint using standard huggingface safetensors writer.",
                        tensors=(t.name,),
                        location=shard_name,
                    )
                )
                continue

            # Check expected byte size if standard dtype
            dtype_bits = SAFETENSORS_DTYPE_BITS.get(t.storage_dtype)
            if dtype_bits is not None:
                numel = 1
                for d in t.shape:
                    numel *= d
                expected_bytes = int((numel * dtype_bits + 7) // 8)
                if span_len != expected_bytes and not (t.logical_shape is not None and t.logical_shape != t.shape):
                    findings.append(
                        Finding(
                            rule_id="QPF013",
                            severity=Severity.CRITICAL,
                            title=f"Tensor span length mismatch on {t.name} in {shard_name}",
                            message=(
                                f"Tensor {t.name!r} declared byte span ({span_len} bytes, offsets [{start}, {end}]) "
                                f"contradicts shape {list(t.shape)} and storage dtype {t.storage_dtype} (expected {expected_bytes} bytes)."
                            ),
                            remediation="Check for corrupted tensor header or repair the exporting serialization script.",
                            tensors=(t.name,),
                            location=shard_name,
                        )
                    )

            # Check shard physical file bounds if file size is known and payload is present
            shard_hdr_len = meta.shard_header_lengths.get(shard_name, 0)
            if shard_size is not None and shard_size > shard_hdr_len and (shard_hdr_len + end) > shard_size:
                findings.append(
                    Finding(
                        rule_id="QPF013",
                        severity=Severity.CRITICAL,
                        title=f"Tensor data span extends past file EOF on {t.name} in {shard_name}",
                        message=(
                            f"Tensor {t.name!r} end offset {end} (physical file offset {shard_hdr_len + end}) "
                            f"exceeds shard file size ({shard_size} bytes) in {shard_name!r}. "
                            "Reading this tensor will fail with unexpected EOF or truncation crash."
                        ),
                        remediation="Verify download completion or re-download the truncated shard file.",
                        tensors=(t.name,),
                        location=shard_name,
                    )
                )

        # 2. Check for overlapping spans within the shard
        sorted_tensors = sorted(shard_tensors, key=lambda x: (x.data_offsets[0], x.data_offsets[1]))  # type: ignore[index]
        for i in range(len(sorted_tensors) - 1):
            curr = sorted_tensors[i]
            nxt = sorted_tensors[i + 1]
            curr_start, curr_end = curr.data_offsets  # type: ignore[misc]
            nxt_start, nxt_end = nxt.data_offsets  # type: ignore[misc]

            # Overlap condition: next tensor starts before current tensor ends
            if nxt_start < curr_end:
                findings.append(
                    Finding(
                        rule_id="QPF013",
                        severity=Severity.CRITICAL,
                        title=f"Overlapping Safetensors memory spans in {shard_name}: {curr.name} and {nxt.name}",
                        message=(
                            f"Tensors {curr.name!r} [{curr_start}, {curr_end}] and {nxt.name!r} [{nxt_start}, {nxt_end}] "
                            f"overlap by {curr_end - nxt_start} bytes in shard {shard_name!r}. "
                            "Overlapping Safetensors buffers violate the format specification and cause data corruption."
                        ),
                        remediation="Re-serialize checkpoint to eliminate shared/overlapping buffer spans across separate tensors.",
                        tensors=(curr.name, nxt.name),
                        location=shard_name,
                    )
                )

    return findings


def _check_qpf014_legacy_pickle_weights(meta: CheckpointMeta) -> list[Finding]:
    """Check for legacy PyTorch pickle weights (.bin/.pt) and unpinned trust_remote_code execution."""
    findings: list[Finding] = []

    # 1. Legacy pickle weight formats
    if meta.legacy_weight_files:
        if meta.container_format in ("pytorch_bin", "bin") or not meta.tensors:
            files_preview = ", ".join(meta.legacy_weight_files[:3])
            if len(meta.legacy_weight_files) > 3:
                files_preview += f" (and {len(meta.legacy_weight_files) - 3} more)"
            findings.append(
                Finding(
                    rule_id="QPF014",
                    severity=Severity.CRITICAL,
                    title="Insecure legacy PyTorch pickle checkpoint format (arbitrary code execution hazard)",
                    message=(
                        f"Checkpoint relies on legacy PyTorch pickle weights ({files_preview}). "
                        "Loading pickle files via torch.load() is unsafe and exposes serving environments to "
                        "arbitrary code execution (ACE) vulnerabilities."
                    ),
                    remediation="Convert checkpoint to zero-code .safetensors format using huggingface safetensors converter before deployment.",
                    location=meta.legacy_weight_files[0],
                )
            )
        elif meta.container_format == "safetensors":
            files_preview = ", ".join(meta.legacy_weight_files[:3])
            findings.append(
                Finding(
                    rule_id="QPF014",
                    severity=Severity.HIGH,
                    title="Redundant legacy pickle weight files lingering alongside Safetensors",
                    message=(
                        f"Repository contains legacy PyTorch pickle files ({files_preview}) alongside Safetensors. "
                        "Serving frameworks or fallback loaders may inadvertently load unvetted pickle weights."
                    ),
                    remediation="Remove or quarantine legacy .bin/.pt files from the model repository or container image.",
                    location=meta.legacy_weight_files[0],
                )
            )

    # 2. Custom modeling code requiring trust_remote_code
    auto_map = meta.config.get("auto_map")
    if isinstance(auto_map, dict) and auto_map and meta.custom_code_files:
        code_preview = ", ".join(meta.custom_code_files[:3])
        arch_targets = ", ".join(list(auto_map.keys())[:2])
        findings.append(
            Finding(
                rule_id="QPF014",
                severity=Severity.HIGH,
                title=f"Unsandboxed custom Python modeling code required ({arch_targets})",
                message=(
                    f"Model config specifies auto_map ({arch_targets}) backed by custom Python scripts ({code_preview}). "
                    "Loading this model requires trust_remote_code=True, executing unreviewed Python scripts on the host."
                ),
                remediation="Audit custom modeling scripts, vendor them into a reviewed Python library, or use native engine model implementations.",
                location="config.json",
            )
        )

    return findings


def _parse_version(raw: str) -> tuple[int, ...]:
    """Return the numeric release of a version string, without trailing zeros."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", raw)
    if match is None:
        raise ValueError(f"Invalid engine version {raw!r}. Expected something like 0.32.0.")
    parts = [int(part) for part in match.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _version_in_range(version: tuple[int, ...], item: dict[str, Any]) -> bool:
    if version < _parse_version(item["introduced"]):
        return False
    if item["fixed"]:
        return version < _parse_version(item["fixed"])
    if item.get("last_affected"):
        return version <= _parse_version(item["last_affected"])
    return True


def _check_qpf016_context_limit(tcfg: dict[str, Any], target: TargetSpec) -> list[Finding]:
    limit = _pos_int(tcfg.get("max_position_embeddings"))
    if target.engine != "vllm" or not target.context_len or not limit or target.context_len <= limit:
        return []
    return [
        Finding(
            rule_id="QPF016",
            severity=Severity.CRITICAL,
            title=f"Context length {target.context_len} is above the model's limit of {limit}",
            message=f"vLLM refuses to start with a max model length above max_position_embeddings ({limit}).",
            remediation=f"Set --context-len to {limit} or lower.",
            location="config.json",
        )
    ]


def _advisories(count: int, adjective: str = "") -> str:
    return f"{count} {adjective}{'advisory' if count == 1 else 'advisories'}"


def _check_qpf015_engine_advisories(target: TargetSpec, engine: EngineProfile) -> list[Finding]:
    if not target.engine_version:
        return []

    version = _parse_version(target.engine_version)
    data = load_advisories()
    hits: list[str] = []
    fixes: list[str] = []
    unfixed = 0
    for advisory in data["engines"].get(engine.id, []):
        matching = [item for item in advisory["ranges"] if _version_in_range(version, item)]
        if not matching:
            continue
        hits.append(advisory["id"])
        fixed = [item["fixed"] for item in matching if item["fixed"]]
        fixes += fixed
        unfixed += not fixed

    if not hits:
        return []

    newest = max(fixes, key=_parse_version) if fixes else None
    remediation = f"Upgrade {engine.name} to {newest} or newer." if newest else f"No fixed {engine.name} release is listed yet."
    if newest and unfixed:
        remediation += f" Review the {_advisories(unfixed, 'unfixed ')} too."
    return [
        Finding(
            rule_id="QPF015",
            severity=Severity.HIGH if newest else Severity.MEDIUM,
            title=f"{engine.name} {target.engine_version} has {_advisories(len(hits), 'published ')}",
            message=(
                f"{engine.name} {target.engine_version} is in the affected range of {_advisories(len(hits))}, "
                f"newest first: {', '.join(hits[:5])}. Advisory data from {data['updated']}."
            ),
            remediation=remediation,
            location=f"--engine-version {target.engine_version}",
            references=tuple(f"https://osv.dev/vulnerability/{item}" for item in hits[:5]),
        )
    ]


def format_quant_summary(meta: CheckpointMeta) -> str:
    q = meta.quant
    if q.method == "none":
        dtypes = sorted({t.storage_dtype for t in meta.tensors.values() if len(t.shape) >= 2})
        return f"unquantized ({'/'.join(dtypes) if dtypes else 'unknown'})"
    parts = [q.method]
    if q.bits is not None:
        parts.append(f"{q.bits:g}-bit")
    if q.group_size is not None:
        parts.append(f"g{q.group_size}")
    if q.format_version and q.format_version != q.method:
        parts.append(q.format_version)
    return " ".join(parts)


def analyze_checkpoint(
    meta: CheckpointMeta,
    target: TargetSpec | None = None,
    *,
    ignored_rules: set[str] | None = None,
) -> ScanReport:
    """Run all pre-flight checks on loaded checkpoint metadata."""
    tgt = target or TargetSpec()
    arch_profiles = load_architecture_profiles()
    gpu_profiles = load_gpu_profiles()
    engine_profiles = load_engine_profiles()

    eng_key = tgt.engine.strip().lower()
    if eng_key not in engine_profiles:
        known_eng = ", ".join(sorted(engine_profiles.keys()))
        raise ValueError(f"Unknown engine {tgt.engine!r}. Known engines: {known_eng}")
    engine_prof = engine_profiles[eng_key]

    gpu_prof: GPUProfile | None = resolve_gpu_profile(tgt.gpu, gpu_profiles) if tgt.gpu else None
    arch_prof = resolve_architecture_profile(meta.config, arch_profiles)
    tcfg = _effective_text_config(meta.config)

    findings: list[Finding] = []
    findings.extend(_check_qpf009_shards_and_mtp(meta, tcfg))
    findings.extend(_check_qpf010_config_contradiction(meta))
    findings.extend(_check_qpf011_engine_format(meta, engine_prof))
    findings.extend(_check_qpf001_sensitive_layers(meta, arch_prof))
    findings.extend(_check_qpf002_vocab_bounds(meta, tcfg))
    findings.extend(_check_qpf003_eos_drift(meta))
    findings.extend(_check_qpf004_gpu_compatibility(meta, gpu_prof, engine_prof))
    findings.extend(_check_qpf005_kernel_corruptions(meta, tgt, gpu_prof))
    findings.extend(_check_qpf006_tp_divisibility(tcfg, tgt))
    findings.extend(_check_qpf007_alignment(meta, arch_prof, tgt, gpu_prof, engine_prof))
    findings.extend(_check_qpf008_desc_act(meta, tgt))
    findings.extend(_check_qpf013_safetensors_spans(meta))
    findings.extend(_check_qpf014_legacy_pickle_weights(meta))
    findings.extend(_check_qpf015_engine_advisories(tgt, engine_prof))
    findings.extend(_check_qpf016_context_limit(tcfg, tgt))

    est_vram, vram_findings = _check_qpf012_vram_budget(meta, tcfg, tgt, gpu_prof)
    findings.extend(vram_findings)

    if ignored_rules:
        norm_ign = {r.strip().upper() for r in ignored_rules}
        findings = [f for f in findings if f.rule_id.upper() not in norm_ign]

    findings.sort(key=lambda f: (-f.severity.rank, f.rule_id, f.title))

    arch_names = meta.config.get("architectures")
    arch_label = (
        str(arch_names[0])
        if isinstance(arch_names, list) and arch_names
        else str(meta.config.get("model_type") or arch_prof.id)
    )

    return ScanReport(
        source=meta.source,
        container_format=meta.container_format,
        architecture=arch_label,
        quant_summary=format_quant_summary(meta),
        total_weight_bytes=meta.total_weight_bytes,
        header_bytes_fetched=meta.header_bytes_fetched,
        tensor_count=len(meta.tensors),
        shard_count=len(meta.shard_files),
        estimated_vram_per_gpu_gb=est_vram,
        target=tgt,
        findings=findings,
    )
