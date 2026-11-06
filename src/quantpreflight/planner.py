"""Work out a setup that should start: GPUs per replica, context length, and the serve command."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shlex

from .models import CheckpointMeta, Finding, ScanReport, SetupPlan, TargetSpec
from .profiles import resolve_gpu_profile
from .rules import _effective_text_config, analyze_checkpoint, context_limit, estimate_per_gpu_vram_gb

TP_CHOICES = (1, 2, 4, 8)
# They say a model is risky to load, not that it won't start
SECURITY_RULES = frozenset({"QPF014", "QPF015"})
# A context this short is rarely worth serving, so below it the memory rule reports the problem
MIN_CONTEXT = 2048
# Used when the config doesn't say how long a context the model takes
FALLBACK_CONTEXT = 131072
# Real vLLM starts on T4 had 0.12 to 0.31 GiB less room for the cache than the estimate, so a
# suggested context leaves this much spare
CONTEXT_HEADROOM_GIB = 0.5
TP_REASONS = {
    "QPF006": "so attention heads and experts split evenly",
    "QPF007": "so each GPU's shard fits the kernel tiles",
    "QPF008": "because desc_act is slow across GPUs",
    "QPF012": "so the weights fit in memory",
}


def blocks_start(finding: Finding) -> bool:
    """Whether a finding means the model shouldn't be started as is."""
    return finding.severity.rank >= 3 and finding.rule_id not in SECURITY_RULES


def _max_fitting_context(
    meta: CheckpointMeta, tcfg: dict, target: TargetSpec, tp: int, usable_gib: float, top: int
) -> int:
    """Return the longest context up to `top` that fits in `usable_gib`, or 0 when nothing fits."""

    def fits(context: int) -> bool:
        total, _, _ = estimate_per_gpu_vram_gb(meta, tcfg, replace(target, tp=tp, context_len=context))
        return total + CONTEXT_HEADROOM_GIB <= usable_gib

    if not fits(1):
        return 0
    low, high = 1, top
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


def _round_down(context: int) -> int:
    step = 1024 if context >= 4096 else 256
    return max(step, context - context % step)


def _context_for(
    meta: CheckpointMeta, tcfg: dict, target: TargetSpec, tp: int, limit: int | None, usable_gib: float | None
) -> tuple[int | None, int | None]:
    """Return `(context to run with, longest context that fits)` at one GPU count."""
    model_top = limit or FALLBACK_CONTEXT
    want = min(target.context_len, model_top) if target.context_len else model_top
    if usable_gib is None:
        # Without a memory budget, only the model's own limit applies
        if target.context_len and limit:
            return min(target.context_len, limit), None
        return target.context_len, None
    max_context = _max_fitting_context(meta, tcfg, target, tp, usable_gib, model_top)
    if max_context >= want:
        return want, max_context
    if max_context >= min(MIN_CONTEXT, want):
        return _round_down(max_context), max_context
    return min(MIN_CONTEXT, want), max_context


def _hub_repo(source: str) -> str | None:
    """Return the repo ID of a Hugging Face source, which the loader labels `org/model (file.gguf)` for GGUF."""
    repo = source.split(" (", 1)[0]
    parts = repo.split("/")
    if len(parts) == 2 and all(parts) and ":" not in repo and not Path(repo).exists():
        return repo
    return None


def serve_command(meta: CheckpointMeta, target: TargetSpec, tp: int, context: int | None, limit: int | None) -> str | None:
    """Return a command that starts the model with these settings, or `None` when there isn't a reliable one."""
    source = meta.source
    gguf = meta.container_format == "gguf"
    # The engine's own default is the model's limit, so the flag only matters below it
    set_context = context is not None and (context != limit or target.engine == "llamacpp")
    util = target.gpu_memory_utilization

    if target.engine == "llamacpp":
        if not gguf:
            return None
        if source.startswith(("http://", "https://")):
            args = ["llama-server", "-mu", source]
        elif _hub_repo(source) and meta.shard_files:
            args = ["llama-server", "--hf-repo", _hub_repo(source), "--hf-file", meta.shard_files[0]]
        else:
            path = Path(source)
            args = ["llama-server", "-m", str(path / meta.shard_files[0]) if path.is_dir() and meta.shard_files else source]
        if set_context:
            args += ["-c", str(context)]
        return shlex.join(args + ["-ngl", "999"])

    if gguf:
        return None
    if target.engine == "vllm":
        args = ["vllm", "serve", source]
        if tp > 1:
            args += ["--tensor-parallel-size", str(tp)]
        if set_context:
            args += ["--max-model-len", str(context)]
        if util != 0.90:
            args += ["--gpu-memory-utilization", f"{util:g}"]
    elif target.engine == "sglang":
        args = ["python", "-m", "sglang.launch_server", "--model-path", source]
        if tp > 1:
            args += ["--tp", str(tp)]
        if set_context:
            args += ["--context-length", str(context)]
        if util != 0.90:
            args += ["--mem-fraction-static", f"{util:g}"]
    elif target.engine == "trtllm":
        args = ["trtllm-serve", source]
        if tp > 1:
            args += ["--tp_size", str(tp)]
        if set_context:
            args += ["--max_seq_len", str(context)]
    else:
        return None
    return shlex.join(args)


def plan_setup(
    meta: CheckpointMeta,
    target: TargetSpec,
    *,
    ignored_rules: set[str] | None = None,
    gpu_count: int | None = None,
) -> SetupPlan:
    """Find the closest setup to `target` that clears every finding a setting can clear."""
    tcfg = _effective_text_config(meta.config)
    limit = context_limit(meta.config, meta.tokenizer_config)
    gpu = resolve_gpu_profile(target.gpu) if target.gpu else None
    vram = target.vram_gb_override if target.vram_gb_override is not None else (gpu.vram_gb if gpu else None)
    usable = vram * target.gpu_memory_utilization if vram else None

    # llama.cpp splits layers across GPUs on its own, so only the GPU count asked for matters
    choices = {target.tp} if target.engine == "llamacpp" else {target.tp, *TP_CHOICES}
    if gpu_count:
        choices = {tp for tp in choices if tp <= gpu_count} or {target.tp}
    # What was asked first, then the closest, with fewer GPUs on a tie
    order = sorted(choices, key=lambda tp: (abs(tp - target.tp), tp))

    tried: list[tuple[int, int | None, int | None, ScanReport, tuple[Finding, ...]]] = []
    for tp in order:
        context, max_context = _context_for(meta, tcfg, target, tp, limit, usable)
        report = analyze_checkpoint(meta, replace(target, tp=tp, context_len=context), ignored_rules=ignored_rules)
        blocking = tuple(f for f in report.findings if blocks_start(f))
        tried.append((tp, context, max_context, report, blocking))
        if not blocking:
            break

    tp, context, max_context, report, blocking = tried[-1]
    if blocking:
        # Report what blocks every GPU count, since a setting can't fix that
        tp, context, max_context, report, blocking = tried[0]
        stuck = set.intersection(*({f.rule_id for f in item[4]} for item in tried))
        blocking = tuple(f for f in blocking if f.rule_id in stuck) or blocking
        reasons = []
        if "QPF012" in stuck:
            reasons.append(f"It doesn't fit in memory on up to {max(order)} GPUs.")
        return SetupPlan(
            status="blocked",
            changes=tuple(reasons),
            tp=tp,
            context_len=context,
            max_context=max_context,
            estimated_vram_per_gpu_gb=report.estimated_vram_per_gpu_gb,
            blocking=blocking,
        )

    changes: list[str] = []
    if tp != target.tp:
        cleared = [f.rule_id for f in tried[0][4] if f.rule_id in TP_REASONS]
        reason = TP_REASONS[cleared[0]] if cleared else "so it starts"
        changes.append(f"Use --tp {tp} instead of {target.tp}, {reason}.")
    if context is not None:
        asked = target.context_len
        if asked and context < asked:
            why = "the model's limit" if limit and context == limit and (max_context is None or max_context >= limit) else "what fits in memory"
            changes.append(f"Use a context of {context} tokens instead of {asked}, {why}.")
        elif not asked and limit and context < limit:
            changes.append(f"Set the context to {context} tokens. The model's full {limit} won't fit in memory.")

    return SetupPlan(
        status="changed" if changes else "ready",
        tp=tp,
        context_len=context,
        max_context=max_context,
        estimated_vram_per_gpu_gb=report.estimated_vram_per_gpu_gb,
        changes=tuple(changes),
        command=serve_command(meta, target, tp, context, limit),
    )
