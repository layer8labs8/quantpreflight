"""Command line behavior: options, exit codes, outputs, and project configuration."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from conftest import write_checkpoint_dir
from quantpreflight import __version__
from quantpreflight.cli import main
from quantpreflight.rules import RULE_CATALOG
from quantpreflight.suppress import load_project_config, resolve_ignored_rules

CLEAN = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 8, "hidden_size": 8, "num_attention_heads": 2}
TENSORS = {"model.embed_tokens.weight": ("F16", [8, 8]), "lm_head.weight": ("F16", [8, 8])}


def _ckpt(root: Path, name: str = "ckpt", **config: Any) -> Path:
    return write_checkpoint_dir(root / name, config={**CLEAN, **config}, tensors=TENSORS)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VLLM_MARLIN_INPUT_DTYPE", raising=False)


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"quantpreflight {__version__}"


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["check", "scan"])
def test_both_scan_commands_work(tmp_path: Path, command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([command, str(_ckpt(tmp_path))]) == 0
    assert "Result: PASS (0 findings)" in capsys.readouterr().out


def test_module_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["quantpreflight", "check", str(_ckpt(tmp_path))])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("quantpreflight", run_name="__main__")
    assert exit_info.value.code == 0
    assert "PASS" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--tp", "0"], "--tp must be >= 1"),
        (["--tp", "-2"], "--tp must be >= 1"),
        (["--gpu-memory-utilization", "0.04"], "between 0.05 and 1.0"),
        (["--gpu-memory-utilization", "1.01"], "between 0.05 and 1.0"),
    ],
)
def test_invalid_numbers_exit_2(tmp_path: Path, args: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path)), *args]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0.05", "1.0"])
def test_memory_fraction_limits_are_allowed(tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path)), "--gpu-memory-utilization", value]) == 0
    capsys.readouterr()


@pytest.mark.parametrize(
    "args",
    [["--engine", "ollama"], ["--format", "xml"], ["--fail-on", "fatal"], ["--marlin-input-dtype", "x"], ["--tp", "two"]],
)
def test_argparse_rejects_bad_choices(tmp_path: Path, args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["check", str(_ckpt(tmp_path)), *args])
    assert exit_info.value.code == 2
    capsys.readouterr()


def test_unknown_gpu_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path)), "--gpu", "H9000"]) == 2
    assert "Unknown GPU target" in capsys.readouterr().err


def test_missing_target_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(tmp_path / "a" / "b" / "c")]) == 2
    assert "Error scanning" in capsys.readouterr().err


# Exit codes

FINDING_LEVELS = [
    ("critical", {"num_attention_heads": 3}, "critical", 1),
    ("critical", {"num_attention_heads": 3}, "none", 0),
    ("high", {"num_local_experts": 3}, "critical", 0),
    ("high", {"num_local_experts": 3}, "high", 1),
    ("medium", {"num_attention_heads": 8, "num_key_value_heads": 1}, "high", 0),
    ("medium", {"num_attention_heads": 8, "num_key_value_heads": 1}, "medium", 1),
    ("medium", {"num_attention_heads": 8, "num_key_value_heads": 1}, "low", 1),
]


@pytest.mark.parametrize(("label", "config", "fail_on", "code"), FINDING_LEVELS, ids=lambda v: str(v))
def test_fail_on_threshold(
    tmp_path: Path, label: str, config: dict[str, Any], fail_on: str, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    target = str(_ckpt(tmp_path, **config))
    assert main(["check", target, "--tp", "2", "--fail-on", fail_on]) == code
    capsys.readouterr()


def test_default_threshold_is_high(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path, num_local_experts=3)), "--tp", "2"]) == 1
    capsys.readouterr()


def test_any_failing_target_fails_the_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = _ckpt(tmp_path, "good")
    bad = _ckpt(tmp_path, "bad", num_attention_heads=3)
    assert main(["check", str(good), str(bad), "--tp", "2"]) == 1
    capsys.readouterr()


def test_one_broken_target_stops_the_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path)), str(tmp_path / "no" / "pe" / "x")]) == 2
    capsys.readouterr()


# Options that reach the analysis


def test_context_length_changes_the_estimate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = {"num_hidden_layers": 64, "num_attention_heads": 8, "hidden_size": 8192}
    target = str(_ckpt(tmp_path, **config))
    main(["check", target, "--context-len", "1000", "--format", "json"])
    small = json.loads(capsys.readouterr().out)["estimated_vram_per_gpu_gb"]
    main(["check", target, "--context-len", "100000", "--format", "json"])
    large = json.loads(capsys.readouterr().out)["estimated_vram_per_gpu_gb"]
    assert large > small


def test_vram_flags_reach_the_budget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = {"num_hidden_layers": 64, "num_attention_heads": 8, "hidden_size": 8192}
    target = str(_ckpt(tmp_path, **config))
    assert main(["check", target, "--vram-gb", "0.5"]) == 1
    assert "QPF012" in capsys.readouterr().out
    assert main(["check", target, "--vram-gb", "1000"]) == 0
    capsys.readouterr()


def test_marlin_flag_and_environment_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    target = str(_ckpt(tmp_path, quantization_config={"quant_method": "auto-round", "bits": 4}))
    assert main(["check", target, "--marlin-input-dtype", "int8", "--ignore", "QPF010"]) == 1
    capsys.readouterr()

    from quantpreflight import cli

    monkeypatch.setenv("VLLM_MARLIN_INPUT_DTYPE", "int8")
    assert cli.main(["check", target, "--ignore", "QPF010"]) == 1
    assert "QPF005" in capsys.readouterr().out


def test_gpu_is_passed_on(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = str(_ckpt(tmp_path, quantization_config={"quant_method": "nvfp4"}))
    assert main(["check", target, "--gpu", "H100", "--ignore", "QPF010"]) == 1
    assert "gpu=H100" in capsys.readouterr().out


# Ignoring findings


@pytest.mark.parametrize("value", ["QPF006", "qpf006", " QPF006 ,QPF001", "QPF001,QPF006,"])
def test_ignore_flag_forms(tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path, num_attention_heads=3)), "--tp", "2", "--ignore", value]) == 0
    capsys.readouterr()


def test_ignore_flag_can_be_repeated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = str(_ckpt(tmp_path, num_attention_heads=3, vocab_size=9))
    assert main(["check", target, "--tp", "2", "--ignore", "QPF006"]) == 1
    capsys.readouterr()
    assert main(["check", target, "--tp", "2", "--ignore", "QPF006", "--ignore", "QPF002"]) == 0
    capsys.readouterr()


def test_project_file_sets_gpu_and_ignores(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = str(_ckpt(tmp_path, num_attention_heads=3))
    (tmp_path / ".quantpreflight.toml").write_text('gpu = "L4"\nignore = ["QPF006"]\n', encoding="utf-8")
    assert main(["check", target, "--tp", "2"]) == 0
    assert "gpu=L4" in capsys.readouterr().out


def test_gpu_flag_wins_over_the_project_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / ".quantpreflight.toml").write_text('gpu = "L4"\n', encoding="utf-8")
    assert main(["check", str(_ckpt(tmp_path)), "--gpu", "H100"]) == 0
    assert "gpu=H100" in capsys.readouterr().out


def test_explicit_config_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "custom.toml"
    config.write_text('ignore = ["QPF006"]\n', encoding="utf-8")
    target = str(_ckpt(tmp_path, num_attention_heads=3))
    assert main(["check", target, "--tp", "2", "--config", str(config)]) == 0
    capsys.readouterr()


def test_broken_project_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / ".quantpreflight.toml").write_text("[[", encoding="utf-8")
    assert main(["check", str(_ckpt(tmp_path))]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_load_project_config_without_a_file(tmp_path: Path) -> None:
    assert load_project_config() == {}
    assert load_project_config(tmp_path / "missing.toml") == {}


@pytest.mark.parametrize(
    ("flags", "config", "expected"),
    [
        (None, {}, set()),
        ([], {"ignore": "QPF001"}, set()),
        ([], {"ignore": ["qpf001", " QPF002 ", "", 5, None]}, {"QPF001", "QPF002"}),
        (["qpf003,  ,QPF004"], {"ignore": ["QPF003"]}, {"QPF003", "QPF004"}),
        ([" "], {}, set()),
    ],
)
def test_resolve_ignored_rules(flags: list[str] | None, config: dict[str, Any], expected: set[str]) -> None:
    assert resolve_ignored_rules(flags, config) == expected


# Output


def test_json_report_fields(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = str(_ckpt(tmp_path, num_attention_heads=3))
    assert main(["check", target, "--gpu", "H100", "--tp", "2", "--context-len", "100", "--format", "json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["version"] == __version__
    assert report["source"] == target
    assert report["container_format"] == "safetensors"
    assert report["architecture"] == "LlamaForCausalLM"
    assert report["quant_summary"] == "unquantized (F16)"
    assert report["tensor_count"] == 2
    assert report["shard_count"] == 1
    assert report["target"] == {"engine": "vllm", "gpu": "H100", "tp": 2, "context_len": 100, "marlin_input_dtype": None}
    finding = report["findings"][0]
    assert set(finding) == {"rule_id", "severity", "title", "message", "remediation", "location", "tensors", "references"}


def test_multiple_targets_in_json_form_a_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path, "a")), str(_ckpt(tmp_path, "b")), "--format", "json"]) == 0
    assert [item["source"].rsplit("b", 1)[-1] for item in json.loads(capsys.readouterr().out)][1] == ""


def test_multiple_targets_in_text_are_separated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path, "a")), str(_ckpt(tmp_path, "b"))]) == 0
    assert capsys.readouterr().out.count("Result: PASS") == 2


def test_multiple_targets_in_sarif_merge_their_findings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = {"num_attention_heads": 3}
    assert main(["check", str(_ckpt(tmp_path, "a", **bad)), str(_ckpt(tmp_path, "b", **bad)), "--tp", "2", "--format", "sarif"]) == 1
    results = json.loads(capsys.readouterr().out)["runs"][0]["results"]
    assert len(results) >= 4


def test_sarif_lists_every_rule(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path)), "--format", "sarif"]) == 0
    sarif = json.loads(capsys.readouterr().out)
    driver = sarif["runs"][0]["tool"]["driver"]
    assert sarif["version"] == "2.1.0"
    assert driver["version"] == __version__
    assert {rule["id"] for rule in driver["rules"]} == set(RULE_CATALOG)


def test_sarif_severity_levels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = str(_ckpt(tmp_path, num_attention_heads=8, num_key_value_heads=1))
    main(["check", target, "--tp", "2", "--format", "sarif", "--fail-on", "none"])
    result = json.loads(capsys.readouterr().out)["runs"][0]["results"][0]
    assert result["level"] == "warning"
    assert result["properties"]["severity"] == "medium"
    assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "config.json"


def test_output_file_creates_folders(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "reports" / "deep" / "scan.json"
    assert main(["check", str(_ckpt(tmp_path)), "--format", "json", "--output", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text(encoding="utf-8"))["findings"] == []


def test_config_and_index_paths_of_one_folder_scan_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ckpt = _ckpt(tmp_path)
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in TENSORS}}), encoding="utf-8"
    )
    assert main(["check", str(ckpt / "config.json"), str(ckpt / "model.safetensors.index.json")]) == 0
    assert capsys.readouterr().out.count("Result: PASS") == 1


def test_list_gpus_shows_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-gpus"]) == 0
    line = next(row for row in capsys.readouterr().out.splitlines() if row.startswith("H100 "))
    assert "sm_90" in line
    assert "80 GiB" in line


def test_list_rules_shows_every_rule(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-rules"]) == 0
    output = capsys.readouterr().out
    assert all(rule in output for rule in RULE_CATALOG)
    assert "[CRITICAL]" in output


def test_list_gpus_json_includes_every_profile_field(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-gpus", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    profiles = json.loads(captured.out)
    ids = [profile["id"] for profile in profiles]
    assert ids == sorted(set(ids))
    h100 = next(profile for profile in profiles if profile["id"] == "H100")
    assert set(h100) == {
        "id", "aliases", "vendor", "arch", "compute_capability", "vram_gb",
        "supports_bf16", "supports_fp8_w8a8", "supports_fp8_block_scale",
        "supports_nvfp4", "supports_marlin",
    }
    assert h100["arch"] == "sm_90"
    assert h100["vram_gb"] == 80.0
    assert h100["supports_bf16"] is True
    assert "sm_90" in h100["aliases"]


def test_list_rules_json_preserves_the_rule_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-rules", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    expected = [
        {"id": rule_id, "severity": severity.value, "title": title, "description": description}
        for rule_id, (severity, title, description) in sorted(RULE_CATALOG.items())
    ]
    assert json.loads(captured.out) == expected


@pytest.mark.parametrize("command", ["list-gpus", "list-rules"])
def test_list_explicit_text_matches_default(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([command]) == 0
    default = capsys.readouterr()
    assert main([command, "--format", "text"]) == 0
    assert capsys.readouterr() == default


# Newer options


def test_engine_version_reaches_the_advisory_rule(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = str(_ckpt(tmp_path))
    assert main(["check", target, "--engine-version", "0.31.0"]) == 1
    assert "QPF015" in capsys.readouterr().out
    assert main(["check", target]) == 0
    capsys.readouterr()


def test_unreadable_engine_version_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path)), "--engine-version", "latest"]) == 2
    assert "Invalid engine version" in capsys.readouterr().err


def test_concurrency_changes_the_estimate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = {"num_hidden_layers": 64, "num_attention_heads": 8, "hidden_size": 8192}
    target = str(_ckpt(tmp_path, **config))
    main(["check", target, "--format", "json"])
    single = json.loads(capsys.readouterr().out)["estimated_vram_per_gpu_gb"]
    main(["check", target, "--format", "json", "--concurrency", "4"])
    assert json.loads(capsys.readouterr().out)["estimated_vram_per_gpu_gb"] > single * 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_concurrency_must_be_positive(tmp_path: Path, value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(_ckpt(tmp_path)), "--concurrency", value]) == 2
    assert "--concurrency must be >= 1" in capsys.readouterr().err


@pytest.mark.parametrize("rule", sorted(RULE_CATALOG))
def test_explain_every_rule(rule: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", rule.lower()]) == 0
    output = capsys.readouterr().out
    assert output.startswith(f"{rule}: {RULE_CATALOG[rule][1]}")
    assert RULE_CATALOG[rule][2] in output
    assert f"#{rule}" in output


def test_explain_unknown_rule(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain", "QPF999"]) == 2
    assert "Known rules: QPF001" in capsys.readouterr().err
