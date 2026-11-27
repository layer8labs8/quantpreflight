"""Finding this machine's GPU and engine version, for `--gpu auto` and `--engine-version auto`."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import subprocess
from typing import Any

import pytest

from conftest import write_checkpoint_dir
from quantpreflight import detect
from quantpreflight.cli import main
from quantpreflight.detect import DetectedGPU, detect_engine_version, detect_gpus, match_gpu_name, resolve_detected_gpu
from quantpreflight.profiles import ProfileValidationError


@pytest.mark.parametrize(
    ("name", "arch", "profile"),
    [
        ("Tesla T4", "sm_75", "T4"),
        ("NVIDIA L40S", "sm_89", "L40S"),
        ("NVIDIA L4", "sm_89", "L4"),
        ("NVIDIA GeForce RTX 4090", "sm_89", "RTX4090"),
        ("NVIDIA GeForce RTX 3090", "sm_86", "A10G"),
        ("NVIDIA A10", "sm_86", "A10G"),
        ("NVIDIA RTX A6000", "sm_86", "RTXA6000"),
        ("NVIDIA A40", "sm_86", "A40"),
        ("NVIDIA A40-48GB", None, "A40"),
        ("NVIDIA RTX A4000", "sm_86", "A10G"),
        ("NVIDIA RTX A4000", None, None),
        ("NVIDIA A100-SXM4-40GB", "sm_80", "A100"),
        ("NVIDIA H100 80GB HBM3", "sm_90", "H100"),
        ("NVIDIA GeForce RTX 5090", "sm_120", "RTX5090"),
        ("AMD Instinct MI300X", "gfx942", "MI300X"),
        ("Tesla P100-PCIE-16GB", "sm_60", None),
        ("Mystery card", None, None),
    ],
)
def test_device_names_match_profiles(name: str, arch: str | None, profile: str | None) -> None:
    assert match_gpu_name(name, arch) == profile


def _ckpt(root: Path) -> Path:
    config = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "vocab_size": 8, "hidden_size": 8, "num_attention_heads": 2}
    tensors = {"model.embed_tokens.weight": ("F16", [8, 8]), "lm_head.weight": ("F16", [8, 8])}
    return write_checkpoint_dir(root / "ckpt", config=config, tensors=tensors)


class FakeSmi:
    """Answers nvidia-smi queries from a table."""

    def __init__(self, rows: str, *, old_driver: bool = False) -> None:
        self.rows = rows
        self.old_driver = old_driver
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_: Any) -> subprocess.CompletedProcess:
        self.calls.append(command)
        if self.old_driver and "compute_cap" in command[1]:
            raise subprocess.CalledProcessError(2, command)
        return subprocess.CompletedProcess(command, 0, stdout=self.rows, stderr="")


@pytest.fixture
def smi(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(rows: str, **kwargs: Any) -> FakeSmi:
        fake = FakeSmi(rows, **kwargs)
        monkeypatch.setattr(detect.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(detect.subprocess, "run", fake)
        return fake

    return install


def test_gpus_are_read_from_nvidia_smi(smi: Any) -> None:
    smi("NVIDIA L40S, 46068, 8.9\nNVIDIA L40S, 46068, 8.9\n")
    gpus = detect_gpus()
    assert gpus == [DetectedGPU("NVIDIA L40S", 44.99, "sm_89")] * 2


def test_old_drivers_without_compute_cap_still_work(smi: Any) -> None:
    fake = smi("Tesla T4, 15360\n", old_driver=True)
    assert detect_gpus() == [DetectedGPU("Tesla T4", 15.0, None)]
    assert len(fake.calls) == 2


def test_unreadable_rows_are_skipped(smi: Any) -> None:
    smi("NVIDIA L4, [N/A], 8.9\nbroken\nNVIDIA L4, 23034, 8.9\n")
    assert detect_gpus() == [DetectedGPU("NVIDIA L4", 22.49, "sm_89")]


def test_no_nvidia_smi_means_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect.shutil, "which", lambda name: None)
    assert detect_gpus() == []


def test_a_failing_nvidia_smi_means_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(command: list[str], **_: Any) -> None:
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(detect.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(detect.subprocess, "run", fail)
    assert detect_gpus() == []


def test_memory_comes_from_the_profile_unless_the_card_differs() -> None:
    # nvidia-smi counts the T4's ECC memory, which PyTorch can't use
    assert resolve_detected_gpu([DetectedGPU("Tesla T4", 15.0, "sm_75")]) == ("T4", None)
    assert resolve_detected_gpu([DetectedGPU("NVIDIA A100-SXM4-40GB", 40.0, "sm_80")]) == ("A100", 40.0)


def test_a_gpu_without_a_profile_is_an_error() -> None:
    with pytest.raises(ProfileValidationError, match="MX230, which has no GPU profile"):
        resolve_detected_gpu([DetectedGPU("NVIDIA GeForce MX230", 2.0, "sm_61")])


def test_engine_versions_come_from_installed_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(package: str) -> str:
        if package == "vllm":
            return "0.33.0"
        raise metadata.PackageNotFoundError(package)

    monkeypatch.setattr(detect.metadata, "version", version)
    assert detect_engine_version("vllm") == "0.33.0"
    assert detect_engine_version("sglang") is None
    assert detect_engine_version("llamacpp") is None


def test_gpu_auto_on_the_command_line(smi: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    smi("NVIDIA L4, 23034, 8.9\nNVIDIA L4, 23034, 8.9\n")
    ckpt = _ckpt(tmp_path)
    assert main(["check", str(ckpt), "--gpu", "auto"]) == 0
    captured = capsys.readouterr()
    assert "Found 2x NVIDIA L4, checked as L4." in captured.err
    assert "gpu=L4" in captured.out


def test_gpu_auto_without_a_gpu_is_an_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(detect.shutil, "which", lambda name: None)
    assert main(["check", "org/model", "--gpu", "auto"]) == 2
    assert "--gpu auto found no NVIDIA GPU" in capsys.readouterr().err


def test_gpu_auto_with_an_unknown_card_is_an_error(smi: Any, capsys: pytest.CaptureFixture[str]) -> None:
    smi("NVIDIA GeForce MX230, 2048, 6.1\n")
    assert main(["suggest", "org/model", "--gpu", "auto"]) == 2
    assert "has no GPU profile yet" in capsys.readouterr().err


def test_engine_version_auto_without_the_engine_notes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("quantpreflight.cli.detect_engine_version", lambda engine: None)
    ckpt = _ckpt(tmp_path)
    assert main(["check", str(ckpt), "--engine-version", "auto"]) == 0
    assert "vLLM isn't installed here" in capsys.readouterr().err


def test_engine_version_auto_checks_the_installed_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("quantpreflight.cli.detect_engine_version", lambda engine: "0.31.0")
    ckpt = _ckpt(tmp_path)
    assert main(["check", str(ckpt), "--engine-version", "auto"]) == 1
    assert "vLLM 0.31.0 has" in capsys.readouterr().out
