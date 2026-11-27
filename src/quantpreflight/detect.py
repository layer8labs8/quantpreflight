"""Find the GPUs and the engine version on this machine, for `--gpu auto` and `--engine-version auto`."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import re
import shutil
import subprocess

from .profiles import ProfileValidationError, load_gpu_profiles, resolve_gpu_profile

ENGINE_PACKAGES = {"vllm": "vllm", "sglang": "sglang", "trtllm": "tensorrt_llm"}
# A detected card this far from its profile's memory is a different model, such as an A100 with 40 GB
MEMORY_TOLERANCE = 0.10


@dataclass(frozen=True)
class DetectedGPU:
    """What nvidia-smi reports for one GPU."""

    name: str
    memory_gib: float
    arch: str | None = None


def _squash(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def match_gpu_name(name: str, arch: str | None = None) -> str | None:
    """Match a device name such as `NVIDIA L40S` to a GPU profile, falling back to its architecture."""
    # Longest names first, so an L40S isn't read as an L4. For short IDs
    # require a whole model token: an RTX A4000 must not match an A40.
    normalized_name = _squash(name)
    tokens = set(re.findall(r"[A-Z0-9]+", name.upper()))
    for gpu_id in sorted(load_gpu_profiles(), key=len, reverse=True):
        normalized_id = _squash(gpu_id)
        if normalized_id in (tokens if len(normalized_id) <= 3 else normalized_name):
            return gpu_id
    if arch:
        try:
            return resolve_gpu_profile(arch).id
        except ProfileValidationError:
            return None
    return None


def _query(fields: str) -> list[list[str]]:
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20).stdout
    return [[part.strip() for part in row.split(",")] for row in output.strip().splitlines() if row.strip()]


def detect_gpus() -> list[DetectedGPU]:
    """Return the NVIDIA GPUs nvidia-smi can see, or an empty list when there are none."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        rows = _query("name,memory.total,compute_cap")
    except (subprocess.SubprocessError, OSError):
        # Drivers older than 510 don't know the compute_cap field
        try:
            rows = [row + [""] for row in _query("name,memory.total")]
        except (subprocess.SubprocessError, OSError):
            return []
    gpus = []
    for row in rows:
        if len(row) < 3:
            continue
        name, memory, cap = row[0], row[1], row[2]
        try:
            memory_gib = round(float(memory) / 1024, 2)
        except ValueError:
            continue
        arch = f"sm_{cap.replace('.', '')}" if cap.replace(".", "").isdigit() else None
        gpus.append(DetectedGPU(name=name, memory_gib=memory_gib, arch=arch))
    return gpus


def resolve_detected_gpu(gpus: list[DetectedGPU]) -> tuple[str, float | None]:
    """Return the profile for the first GPU, and a memory override when the card differs from the profile."""
    first = gpus[0]
    gpu_id = match_gpu_name(first.name, first.arch)
    if gpu_id is None:
        raise ProfileValidationError(
            f"Found {first.name}, which has no GPU profile yet. Pass --gpu with the closest match, or see CONTRIBUTING.md to add one."
        )
    profile = resolve_gpu_profile(gpu_id)
    if abs(first.memory_gib - profile.vram_gb) > profile.vram_gb * MEMORY_TOLERANCE:
        return gpu_id, first.memory_gib
    return gpu_id, None


def detect_engine_version(engine: str) -> str | None:
    """Return the installed version of the engine's Python package, or `None` when it isn't installed."""
    package = ENGINE_PACKAGES.get(engine)
    if package is None:
        return None
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
