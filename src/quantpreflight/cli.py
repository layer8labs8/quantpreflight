"""Command-line interface for `quantpreflight`."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import pathlib
import sys
from typing import Sequence
import urllib.error

from . import __version__
from .detect import detect_engine_version, detect_gpus, resolve_detected_gpu
from .errors import SCAN_ERRORS
from .formatters import (
    format_json_report,
    format_sarif_report,
    format_suggestions,
    format_suggestions_json,
    format_text_report,
)
from .loader import load_checkpoint_metadata
from .models import Severity, TargetSpec
from .planner import plan_setup
from .profiles import (
    ProfileValidationError,
    load_engine_profiles,
    load_gpu_profiles,
    validate_all_profiles,
)
from .rules import RULE_CATALOG, analyze_checkpoint
from .suggest import suggest
from .suppress import load_project_config, resolve_ignored_rules


class _UsageError(Exception):
    """A bad option, reported with exit code 2."""


def _parse_fail_on(raw: str) -> Severity | None:
    low = raw.strip().lower()
    if low == "none":
        return None
    return Severity(low)


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    """Options that say where the model will run, shared by `check` and `suggest`."""
    parser.add_argument(
        "--gpu",
        default=None,
        help="Target GPU profile or SM arch (e.g. H100, A100, L40S, T4, B200, GB10, MI300X, sm_89), or auto to use this machine's GPU.",
    )
    parser.add_argument(
        "--engine",
        default="vllm",
        choices=["vllm", "sglang", "trtllm", "llamacpp"],
        help="Target inference engine (default: vllm).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel degree (default: 1).",
    )
    parser.add_argument(
        "--engine-version",
        default=None,
        help="Installed engine version, checked against published advisories (vLLM and SGLang), or auto to read it from this machine.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Requests served at once, for the KV-cache estimate (default: 1).",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Context length for the KV cache estimate. Defaults to the model limit, up to 4096.",
    )
    parser.add_argument(
        "--vram-gb",
        type=float,
        default=None,
        help="Override per-GPU physical VRAM in GiB.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Usable GPU memory fraction (default: 0.90).",
    )
    parser.add_argument(
        "--marlin-input-dtype",
        choices=["fp16", "bf16", "fp8", "int8"],
        default=os.environ.get("VLLM_MARLIN_INPUT_DTYPE"),
        help="Simulate VLLM_MARLIN_INPUT_DTYPE kernel flag (fp8 or int8).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN environment variable).",
    )


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Comma-separated rule IDs to suppress (e.g. QPF003,QPF012).",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="Path to .quantpreflight.toml configuration file.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantpreflight",
        description=(
            "Zero-GPU pre-flight scanner for 32/16/8/4/2-bit quantized AI checkpoints "
            "(Safetensors & GGUF)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"quantpreflight {__version__}")

    sub = parser.add_subparsers(dest="command")

    for cmd_name in ("check", "scan"):
        chk = sub.add_parser(
            cmd_name,
            help="Scan a Hugging Face repo ID, HTTPS URL, or local checkpoint path.",
        )
        chk.add_argument(
            "targets",
            nargs="+",
            help="Hugging Face repo (org/model), URL, or local directory/file path(s).",
        )
        _add_target_args(chk)
        chk.add_argument(
            "--revision",
            default="main",
            help="Hugging Face git revision / branch / commit (default: main).",
        )
        _add_project_args(chk)
        chk.add_argument(
            "--fail-on",
            choices=["critical", "high", "medium", "low", "none"],
            default="high",
            help="Exit with code 1 when any finding meets or exceeds this severity (default: high).",
        )
        chk.add_argument(
            "--format",
            choices=["text", "json", "sarif"],
            default="text",
            help="Output report format (default: text).",
        )
        chk.add_argument(
            "--output",
            type=pathlib.Path,
            default=None,
            help="Optional file path to write the rendered output.",
        )

    sug = sub.add_parser(
        "suggest",
        help="Find the quantized versions of a model that start on your GPUs.",
    )
    sug.add_argument("model", help="Hugging Face repo of the model, or of any quantized version of it.")
    _add_target_args(sug)
    _add_project_args(sug)
    sug.add_argument("--limit", type=int, default=15, help="How many quantized versions to check (default: 15).")
    sug.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text).")

    explain = sub.add_parser("explain", help="Show what a rule means.")
    explain.add_argument("rule", help="Rule ID, such as QPF006.")

    list_gpus = sub.add_parser("list-gpus", help="List bundled GPU hardware profiles.")
    list_rules = sub.add_parser("list-rules", help="List all pre-flight rules.")
    for listing in (list_gpus, list_rules):
        listing.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text).")
    sub.add_parser("validate-profiles", help="Validate all TOML architecture and hardware profiles.")

    return parser


def _cmd_list_gpus(output_format: str) -> int:
    gpus = load_gpu_profiles()
    if output_format == "json":
        print(json.dumps([asdict(gpus[gid]) for gid in sorted(gpus)], indent=2))
        return 0
    print(f"{'GPU':<12} {'Vendor':<8} {'Arch':<8} {'VRAM':<10} {'BF16':<6} {'FP8':<6} {'BlkFP8':<8} {'NVFP4':<6}")
    print("-" * 72)
    for gid in sorted(gpus.keys()):
        g = gpus[gid]
        print(
            f"{g.id:<12} {g.vendor:<8} {g.arch:<8} {g.vram_gb:>5.0f} GiB  "
            f"{str(g.supports_bf16):<6} {str(g.supports_fp8_w8a8):<6} "
            f"{str(g.supports_fp8_block_scale):<8} {str(g.supports_nvfp4):<6}"
        )
    return 0


def _cmd_list_rules(output_format: str) -> int:
    rules = sorted(RULE_CATALOG.items())
    if output_format == "json":
        print(json.dumps([
            {"id": rid, "severity": sev.value, "title": title, "description": desc}
            for rid, (sev, title, desc) in rules
        ], indent=2))
        return 0
    for rid, (sev, title, desc) in rules:
        print(f"{rid} [{sev.value.upper():<8}] {title}")
        print(f"    {desc}")
    return 0


def _cmd_explain(rule: str) -> int:
    rule_id = rule.strip().upper()
    if rule_id not in RULE_CATALOG:
        known = ", ".join(sorted(RULE_CATALOG))
        print(f"Unknown rule {rule!r}. Known rules: {known}", file=sys.stderr)
        return 2
    severity, title, description = RULE_CATALOG[rule_id]
    print(f"{rule_id}: {title}")
    print(f"Highest severity: {severity.value}")
    print(description)
    print(f"More: https://layer8labs8.github.io/quantpreflight/#{rule_id}")
    return 0


def _cmd_validate_profiles() -> int:
    try:
        n_arch, n_gpu, n_eng = validate_all_profiles()
    except ProfileValidationError as exc:
        print(f"Profile validation error: {exc}", file=sys.stderr)
        return 2
    print(f"Validated {n_arch} architecture profiles, {n_gpu} GPU profiles, and {n_eng} engine profiles.")
    return 0


def _resolve_target(args: argparse.Namespace, proj_cfg: dict) -> tuple[TargetSpec, int | None]:
    """Build the target from the options, finding this machine's GPU and engine version when asked."""
    if args.tp < 1:
        raise _UsageError("--tp must be >= 1.")
    if args.concurrency < 1:
        raise _UsageError("--concurrency must be >= 1.")
    if not (0.05 <= args.gpu_memory_utilization <= 1.0):
        raise _UsageError("--gpu-memory-utilization must be between 0.05 and 1.0.")

    gpu = args.gpu or proj_cfg.get("gpu")
    vram_gb = args.vram_gb
    gpu_count = None
    if isinstance(gpu, str) and gpu.strip().lower() == "auto":
        found = detect_gpus()
        if not found:
            raise _UsageError("--gpu auto found no NVIDIA GPU. Pass the GPU you plan to use, such as --gpu H100.")
        try:
            gpu, detected_vram = resolve_detected_gpu(found)
        except ProfileValidationError as exc:
            raise _UsageError(str(exc)) from exc
        vram_gb = vram_gb if vram_gb is not None else detected_vram
        gpu_count = len(found)
        print(f"Found {gpu_count}x {found[0].name}, checked as {gpu}.", file=sys.stderr)

    engine_version = args.engine_version
    if engine_version and engine_version.strip().lower() == "auto":
        engine_version = detect_engine_version(args.engine)
        if engine_version is None:
            name = load_engine_profiles()[args.engine].name
            print(f"Note: {name} isn't installed here, so its version wasn't checked.", file=sys.stderr)

    target = TargetSpec(
        engine=args.engine,
        gpu=gpu,
        tp=args.tp,
        context_len=args.context_len,
        vram_gb_override=vram_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        marlin_input_dtype=args.marlin_input_dtype,
        engine_version=engine_version,
        concurrency=args.concurrency,
    )
    return target, gpu_count


def _cmd_suggest(args: argparse.Namespace) -> int:
    if args.limit < 1:
        print("Error: --limit must be >= 1.", file=sys.stderr)
        return 2
    try:
        proj_cfg = load_project_config(args.config)
        ignored = resolve_ignored_rules(args.ignore, proj_cfg)
        target, gpu_count = _resolve_target(args, proj_cfg)
    except _UsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        base, candidates = suggest(
            args.model, target, token=args.token, limit=args.limit, ignored_rules=ignored, gpu_count=gpu_count
        )
    except (*SCAN_ERRORS, urllib.error.HTTPError) as exc:
        print(f"Error finding versions of {args.model}: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(format_suggestions_json(base, target, candidates))
    else:
        print(format_suggestions(base, target, candidates))
    return 0 if any(item.starts for item in candidates) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "list-gpus":
        return _cmd_list_gpus(args.format)
    if args.command == "list-rules":
        return _cmd_list_rules(args.format)
    if args.command == "explain":
        return _cmd_explain(args.rule)
    if args.command == "validate-profiles":
        return _cmd_validate_profiles()

    if args.command == "suggest":
        return _cmd_suggest(args)

    try:
        proj_cfg = load_project_config(args.config)
        ignored = resolve_ignored_rules(args.ignore, proj_cfg)
        fail_threshold = _parse_fail_on(args.fail_on)
        target_spec, gpu_count = _resolve_target(args, proj_cfg)
    except _UsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # Deduplicate targets when pre-commit passes both config.json and index.json from one directory
    seen_sources: set[str] = set()
    reports = []
    for raw_target in args.targets:
        p = pathlib.Path(raw_target)
        dedupe_key = str(p.parent.resolve()) if p.is_file() and p.name in ("config.json", "model.safetensors.index.json") else raw_target
        if dedupe_key in seen_sources:
            continue
        seen_sources.add(dedupe_key)

        try:
            meta = load_checkpoint_metadata(
                raw_target,
                revision=args.revision,
                token=args.token,
                prefer_gguf=args.engine == "llamacpp",
            )
            rep = analyze_checkpoint(meta, target_spec, ignored_rules=ignored)
            rep.setup = plan_setup(meta, target_spec, ignored_rules=ignored, gpu_count=gpu_count)
            reports.append(rep)
        except SCAN_ERRORS as exc:
            print(f"Error scanning {raw_target}: {exc}", file=sys.stderr)
            return 2

    if args.format == "json":
        rendered = format_json_report(reports[0]) if len(reports) == 1 else (
            "[\n" + ",\n".join(format_json_report(r) for r in reports) + "\n]"
        )
    elif args.format == "sarif":
        # Merge findings if multiple targets were passed
        primary = reports[0]
        for extra in reports[1:]:
            primary.findings.extend(extra.findings)
        rendered = format_sarif_report(primary)
    else:
        rendered = "\n\n".join(format_text_report(r) for r in reports)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    any_failed = any(not r.passed(fail_threshold) for r in reports)
    return 1 if any_failed else 0
