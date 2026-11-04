# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- KV-cache estimates now count global attention layers declared by `layer_types` or `sliding_window_pattern` at the full context length instead of capping every layer at the sliding window.

## [0.4.0] - 2026-11-03

### Added
- Every report ends with a setup that should start: the closest GPU count that works, the longest context that fits in memory, and the command for vLLM, SGLang, TensorRT-LLM, or llama.cpp. When no setting helps, it says what blocks the start.
- `quantpreflight suggest <model>`, which scans the quantized versions of a model on Hugging Face and ranks the ones that start on your GPUs.
- `--gpu auto`, which reads this machine's NVIDIA GPUs, and `--engine-version auto`, which reads the installed vLLM or SGLang.
- `tools/setup_check.py`, which starts vLLM with the suggested setups and compares. On Tesla T4, 9 of 9 were right. A suggested context leaves 0.5 GiB spare, because the estimate gave the cache up to 0.31 GiB more room than vLLM did.
- The browser scanner shows the setup too.

### Changed
- `tools/gpu_check.py` matches GPU names with the same code as `--gpu auto`.

## [0.3.0] - 2026-11-03

### Added
- The first release on PyPI: `pip install quantpreflight`.
- Rule `QPF015` and `--engine-version`, which check your vLLM or SGLang version against bundled OSV advisories, with a weekly job to refresh them.
- Split GGUF support: every `-NNNNN-of-NNNNN.gguf` part is read, and a missing part fires `QPF009`.
- `--concurrency`, which scales the KV cache in the memory estimate.
- `quantpreflight explain <rule>`.
- A project website with a browser scanner that runs the real rules on a pasted `config.json` or a Hugging Face model ID.
- Model profiles for Phi and GLM, and new names for Qwen3.5, Qwen4-Exp, DeepSeek V3.2 and V4, Gemma 4 Unified, and Mistral 3.
- `tools/gpu_check.py` compares predictions with real vLLM loads, and `tools/scan_public_models.py` scans public models.
- Hugging Face rate limits and server errors (429, 502, 503, 504) are retried with a short wait.
- 132 labelled benchmark cases in ten groups, with recall and precision for each run.
- Header-only demo checkpoints in `tests/fixtures/`.
- Workflows for the website, releases, and a self-test of the GitHub Action.
- GPU profiles for RTX 4090 and MI325X.
- A [GPU testing](docs/gpu-testing.md) guide and an issue form for sharing results from your own GPUs.

### Fixed
- Results checked against real vLLM 0.33 runs on Tesla T4, and against 500 public models:
  - `QPF002` no longer mistakes GGUF `attn_output.weight` for `lm_head`, which fired on 33 of the 73 GGUF models.
  - `QPF008` now fires at any `--tp` on vLLM, which rejects `desc_act=true`.
  - BF16 on GPUs without native support is now medium, because those models load. FP8 without FP8 cores is high, because real results were mixed.
  - ModelOpt below compute capability 80 is now critical, because vLLM refuses it.
  - `QPF012` is fitted to measured memory use. The old estimate was about 20% low.
  - The T4 is listed at 14.56 GiB, the memory PyTorch really sees.
  - vLLM's table no longer lists `bitsandbytes` or `bitnet`, and it now accepts `torchao`, `quark`, and pickle weights.
  - `QPF003` is medium, because vLLM also stops on the tokenizer's token.
  - `QPF007` is medium for Compressed-Tensors, which falls back per layer, and skips float formats.
  - Variant files with no index no longer count as duplicate tensors.
  - MLX models are named as MLX instead of being read as stale GPTQ.
  - Marlin-packed weights no longer fire `QPF010`, and duplicate tensors are medium.
  - `--engine llamacpp` prefers GGUF files in a repository that has both.
  - GGML types newer than the table, tokenizers that can't be read, and indexes over 32 MiB no longer stop a scan.
- Malformed `config.json` values no longer crash a scan. Wrong types, empty lists, and huge numbers are now ignored or reported.
- A `weight_block_size` containing zero no longer raises `ZeroDivisionError`.
- Deeply nested JSON and failed HTTP reads now exit with code 2 and a message instead of a traceback.
- `sm_80` and `sm_89` now resolve to the GPUs that declare them as aliases, `A100` and `L40S`.
- `pad_token_id: -1` no longer fires `QPF002`, because many configs use it to mean "no padding token".
- A shard reached through a symlink to a folder with a similar name is now rejected.
- The quantization summary no longer repeats the method name, as in `gptq 4-bit g128 gptq`.
- `--gpu rtx4090` used the 48 GiB of an L40S, and `--gpu mi325x` used the 192 GiB of an MI300X. Both now have their own profiles.
- The text report names the GPU an alias resolves to, such as `L40S (from sm_89)`.
- `tools/gpu_check.py` no longer empties your Hugging Face cache. It downloads into its own folder, detects any GPU PyTorch can see, and no longer reads an L40S as an L4.
- The benchmark tables count cases passed, instead of findings next to case counts.

### Changed
- `QPF002` docs now describe its relation to `CVE-2026-115916` accurately, and the README states the limits of the `QPF012` estimate.
- CI runs on Python 3.14 and builds the website.

### Removed
- The synthetic 100,000-test matrix (`tools/run_mass_scale_matrix.py`) and its results file, replaced by focused tests.
- `tools/probe_50_models.py` and `tools/run_real_defects_suite.py`, replaced by `tools/scan_public_models.py`.

### Changed in the tools
- `tools/run_50_defect_matrix.py` is now `tools/run_scenarios.py`. One scenario was removed because it described a problem its model doesn't have, and the others now match what real GPUs showed.

## [0.2.0] - 2026-11-01

### Added
- **Architecture-Aware KV Cache Memory Modeling (`estimate_per_gpu_vram_gb`)**:
  - **DeepSeek MLA (Multi-Head Latent Attention)**: Detects `kv_lora_rank` and `qk_rope_head_dim` to calculate the joint compressed latent KV cache memory (`kv_lora_rank + qk_rope_head_dim = 576` dims/token) instead of naive `2 * num_heads * head_dim = 32,768` dims/token, eliminating 98% false memory inflation on DeepSeek V2/V3/R1.
  - **Sliding-Window Attention (SWA)**: Binds KV cache sequence memory to `min(context_len, sliding_window)` for Mistral, Mixtral, and models with sliding-window attention.
  - **Gemma 2 Alternating Sliding Window**: Accurately accounts for alternating sliding layers and global attention layers.
  - **Hybrid SSM & Linear Attention**: Models layer splits for Mamba, Mamba-2, and Jamba (e.g. 4 attention layers out of 32 total), avoiding linear sequence memory inflation on recurrent SSM layers.
- **Rule `QPF013` (SafeTensors Span Integrity & Offset Corruption)**:
  - Validates that tensor byte spans within each shard are monotonically non-overlapping and strictly within physical file boundaries.
  - Detects byte length contradictions between declared header offsets and `numel * dtype` payload sizes.
- **Rule `QPF014` (Legacy Pickle Format & Supply-Chain Deserialization Hazard)**:
  - Detects dangerous legacy PyTorch pickle checkpoint weights (`pytorch_model.bin`, `*.pt`, `model.pt`) susceptible to arbitrary code execution (ACE) via `torch.load()`.
  - Flags lingering legacy pickle weights redundant alongside Safetensors.
  - Warns about unpinned custom modeling code requiring `trust_remote_code=True` (`auto_map`).
- **Expanded 50-Model Empirical Corpus Probe**: Tested across 50 diverse open-weights checkpoints on Hugging Face (Qwen, Llama, DeepSeek, Mistral, Gemma 2, Mamba, Jamba, Granite, Phi, NVFP4, AWQ, GPTQ, GGUF).

## [0.1.1] - 2026-11-01

### Added
- **NVIDIA ModelOpt and AutoGPTQ External Config Discovery**: Support for standalone `hf_quant_config.json` (NVIDIA ModelOpt FP8/NVFP4) and `quantize_config.json` (AutoGPTQ), eliminating false-positive `QPF010` warnings on official NVIDIA and community quantizations.
- **Two-Tier Fast Tokenizer Header Resolution**: Intelligently inspects `tokenizer_config.json` (`added_tokens_decoder`) and `config.json` (`vocab_size`) first, bypassing redundant 7–14 MiB `tokenizer.json` downloads and reducing remote scan times from ~40s to ~2s per checkpoint.
- **Automatic GGUF Repository Sibling Discovery**: Scanning bare Hugging Face GGUF repository IDs (e.g. `quantpreflight check Qwen/Qwen2.5-0.5B-Instruct-GGUF --engine llamacpp`) automatically discovers and selects the primary quantized GGUF variant (`Q4_K_M` / `Q4_0`).
- **Network Resilience & Connection Retry Loop**: Exponential backoff retry handling for transient TCP connection drops (`WinError 10054`) and remote socket resets across international CDN roundtrips.
- **20 Documented vLLM Issue Verification Suite**: Cataloged 20 real closed/documented vLLM deployment issues (`benchmark/vllm_20_issues.json`) and added regression tests (`tests/test_vllm_20_issues.py`).

## [0.1.0] - 2026-11-01

### Added
- Zero-GPU, zero-weight-download pre-flight scanner for Safetensors (`AWQ`, `GPTQ`, `Compressed-Tensors`, `FP8`, `NVFP4`, `MXFP4`, `AutoRound`, `BitsAndBytes`, `BitNet`) and `GGUF` (`v2`/`v3`, `F32`/`F16`/`BF16`, K-quants, I-quants, ternary).
- Parallel HTTP `Range` loader for Hugging Face repositories and direct URLs, preserving `Range` across cross-origin CDN redirects while stripping `Authorization` headers across hosts.
- Rules `QPF001` through `QPF012` covering bit-starved sensitive layers, `CVE-2026-115916` vocabulary/embedding out-of-bounds crashes, `eos_token_id` drift, GPU `sm_XX` hardware compatibility, Marlin/FP8 kernel hazards, `TP` divisibility, sharded alignment, shard index integrity, `MTP` projection mismatches, and per-GPU VRAM budgets.
- Declarative TOML profiles for 8 model architecture families, 13 GPUs, and 4 serving engines (`vLLM`, `SGLang`, `TensorRT-LLM`, `llama.cpp`).
- Human terminal, JSON, and GitHub Code Scanning SARIF 2.1.0 output formatters, plus `action.yml`, `.pre-commit-hooks.yaml`, and a 16-case labelled benchmark suite.
