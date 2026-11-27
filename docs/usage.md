# Usage

## Scan a model

Install it with `pip install quantpreflight`, then:

```bash
quantpreflight check <target> [options]
```

`<target>` can be a Hugging Face repo ID, a direct URL, a local folder, or a single `.safetensors` or `.gguf` file. You can pass several targets.

With `--engine llamacpp`, a repository that has both Safetensors and GGUF files is scanned as GGUF. Every part of a split GGUF file is read, and a missing part is reported.

## Get a setup that starts

After the findings, the report ends with a setup. It tries the GPU counts 1, 2, 4, and 8, closest to your `--tp` first, and the longest context that fits in memory. Then it says one of three things:

- **Ready as asked.** It also says how much context fits.
- **It starts with these changes.** For example, a `--tp` that splits the attention heads evenly, or a shorter context. vLLM refuses to start when the model's full context doesn't fit, so the command sets `--max-model-len` in that case.
- **No setting makes this start.** The format needs another GPU, or the engine can't read it. Run `suggest` to find a version that works.

Each setup that starts comes with the command for your engine: `vllm serve`, SGLang's `launch_server`, `trtllm-serve`, or `llama-server`. With `--format json`, it's in the `setup` field. Findings such as `QPF014` warn that loading a model is risky, so they don't count against a start, but they still set the exit code.

## Find a version that runs

```bash
quantpreflight suggest Qwen/Qwen2.5-7B-Instruct --gpu L4
```

`suggest` takes a model, or any quantized version of it, and finds the model it came from. It then reads Hugging Face's list of quantized versions, scans the most downloaded ones, and ranks them: the ones that start first, the highest precision first, then the most downloaded. `--limit` sets how many versions it scans, 15 by default. With `--engine llamacpp`, it scans GGUF repositories only, and otherwise it skips them.

It exits with `0` when at least one version starts, `1` when none does, and `2` when the model can't be found.

## Check on the machine you'll serve from

`--gpu auto` reads the GPUs from `nvidia-smi` and checks against the closest profile. If the card's memory differs from that profile, such as an A100 with 40 GB, it uses the card's memory. The setup then only tries GPU counts the machine has. `--engine-version auto` reads the version of the installed vLLM or SGLang:

```bash
quantpreflight check Qwen/Qwen2.5-7B-Instruct-AWQ --gpu auto --engine-version auto
```

AMD GPUs aren't detected yet, so pass `--gpu MI300X` there. In `.quantpreflight.toml`, `gpu = "auto"` works too.

## Options

| Option | What it does | Default |
| :--- | :--- | :--- |
| `--engine` | `vllm`, `sglang`, `trtllm`, or `llamacpp` | `vllm` |
| `--gpu` | Target GPU, alias, or architecture, or `auto` | None |
| `--tp` | GPUs per replica | `1` |
| `--engine-version` | Installed engine version, checked against advisories, or `auto` | None |
| `--context-len` | Context length for the cache estimate | Model limit, up to 4096 |
| `--concurrency` | Requests served at once, for the cache estimate | `1` |
| `--vram-gb` | Override VRAM per GPU, in GiB | From the GPU table |
| `--gpu-memory-utilization` | Share of VRAM the engine may use | `0.90` |
| `--marlin-input-dtype` | Simulate `VLLM_MARLIN_INPUT_DTYPE` | From the environment |
| `--revision` | Hugging Face branch, tag, or commit | `main` |
| `--token` | Hugging Face token | `HF_TOKEN` |
| `--ignore` | Rule IDs to skip, separated by commas | None |
| `--config` | Load a settings file | `.quantpreflight.toml` if present |
| `--fail-on` | Lowest severity that fails the run, or `none` | `high` |
| `--format` | `text`, `json`, or `sarif` | `text` |
| `--output` | Write the report to a file | Print it |

Exit codes: `0` means nothing reached the `--fail-on` level, `1` means something did, and `2` means the scan itself failed, for example on a corrupt file or a missing repository.

## Other commands

| Command | What it does |
| :--- | :--- |
| `quantpreflight suggest <model>` | Rank the quantized versions of a model that start. It takes the same target options as `check`, plus `--limit` |
| `quantpreflight explain QPF006` | Explain a rule |
| `quantpreflight list-rules [--format json]` | List every rule; JSON includes the rule ID, severity, title, and description |
| `quantpreflight list-gpus [--format json]` | List bundled GPU profiles; JSON includes every profile field |
| `quantpreflight validate-profiles` | Check the bundled TOML profiles |

You can pass a GPU alias or architecture such as `rtx4090` or `sm_89`.

## Severity

| Severity | What it means | What to do |
| :--- | :--- | :--- |
| **Critical** | It will crash, corrupt output, or expose your host. | Fix before you deploy. |
| **High** | Serious, and it may not start. | Fix soon. |
| **Medium** | It works, with a real cost. | Plan a fix. |
| **Low** | Good hygiene. | Fix when convenient. |

## Check your engine version

Pass the version you run, and it's compared with published vLLM and SGLang advisories from [OSV](https://osv.dev). They're bundled with the tool, so this needs no extra network access, and a weekly job refreshes them:

```bash
quantpreflight check Qwen/Qwen2.5-7B-Instruct-AWQ --engine-version 0.31.0
```

```text
[1] HIGH QPF015: vLLM 0.31.0 has 10 published advisories
    Fix:      Upgrade vLLM to 0.32.0 or newer.
```

`QPF002` catches token IDs the checkpoint itself carries. CVE-2026-115916 is an off-by-one in vLLM that a crafted request can trigger, so only `--engine-version` can flag that one.

## Estimate memory for real traffic

The estimate covers the weights, runtime overhead, and the KV cache for one request. Pass your own numbers to see the cost of real traffic:

```bash
quantpreflight check Qwen/Qwen2.5-72B-Instruct-AWQ --gpu H100 --tp 4 --context-len 8192 --concurrency 32
```

## Ignore a finding you've reviewed

Add `.quantpreflight.toml` to your repository root. It can also set your usual GPU:

```toml
gpu = "H100"
ignore = ["QPF008"]
```

Or pass `--ignore QPF003,QPF012` on the command line.

## Use it in CI

Add this as `.github/workflows/model-preflight.yml`:

```yaml
name: Model preflight
on: [push, pull_request]

permissions:
  contents: read
  security-events: write

jobs:
  preflight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: layer8labs8/quantpreflight@v0.4.0
        with:
          target: ./exported-checkpoint
          engine: vllm
          engine-version: "0.33.0"
          gpu: H100
          tp: "4"
          fail-on: high
          sarif-file: quantpreflight.sarif
      - uses: github/codeql-action/upload-sarif@v4
        if: always()
        with:
          sarif_file: quantpreflight.sarif
```

The upload step adds findings to the Security tab, which on private repositories can require GitHub Advanced Security. Without it, remove that step and `security-events: write`.

To run it as a [pre-commit](https://pre-commit.com) hook instead:

```yaml
repos:
  - repo: https://github.com/layer8labs8/quantpreflight
    rev: v0.4.0
    hooks:
      - id: quantpreflight
```
