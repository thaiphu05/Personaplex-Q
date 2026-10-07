# PersonaPlex LoRA Fine-Tuning, Live Interaction & Full-Duplex Benchmark

End-to-end pipeline for fine-tuning [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) with LoRA adapters, chatting with the model in real time (terminal CLI or Gradio web UI), and evaluating full-duplex conversational skills with the [Full-Duplex-Bench](https://arxiv.org/abs/2412.06251) protocol.

> 🇻🇳 *Pipeline huấn luyện LoRA cho PersonaPlex-7B, đàm thoại song công trực tiếp qua micro/loa hoặc web UI, và đánh giá Full-Duplex-Bench.*

## What this repo provides

| Feature | Description |
| :--- | :--- |
| **LoRA fine-tuning** | LoRA adapters over the PersonaPlex 7B transformer (temporal), optional depformer LoRA, 4-bit QLoRA. Single GPU or multi-GPU DDP via Accelerate. |
| **Stage-wise training** | Stage 0 Mimi audio codec adaptation → temporal-only → depth-only → joint with dual learning rates. |
| **Qwen backbone swap** | Replace the Helium 7B backbone with `Qwen/Qwen3-8B` or `Qwen/Qwen3.5-9B` (auto-detected LoRA surfaces). |
| **Live interaction** | Full-duplex terminal CLI (mic ↔ speakers) and Gradio web UI with temporary public `gradio.live` links. |
| **Full-Duplex-Bench** | Offline evaluation of pause handling, backchanneling, smooth turn-taking, and user interruption — no external ASR needed. |
| **Pre-flight tooling** | Dataset validation, frame inspection, Mimi codec fidelity tests, 1-step smoke tests, unit test suite. |

> 🇻🇳 *2 backbone: PersonaPlex 7B gốc hoặc Qwen3-8B / Qwen3.5-9B. Công cụ kiểm tra + huấn luyện + demo + benchmark trong 1 repo.*

---

## 1. Requirements & Environment Setup

### Requirements

- **Python 3.10 or 3.11**
- **PyTorch**: `>=2.2,<2.5` for the PersonaPlex backbone. The **Qwen backbone swap requires `torch>=2.6` + `transformers>=4.57`** (Qwen3.5 architecture).
- **ffmpeg** (+ `libsox-fmt-all` on Linux) for 24 kHz stereo audio processing.
- CUDA 12.1/12.4 GPU (Linux), or MPS/CPU (macOS).
- Do not install over your system Python — use conda or a venv.

> 🇻🇳 *Python 3.10/3.11, Torch 2.4.x (PersonaPlex) hoặc ≥2.6 (Qwen), ffmpeg bắt buộc.*

### Option A: Conda

```bash
conda env create -f environment.yml          # offline env, torch installed separately
# or, if you plan to download from Hugging Face:
conda env create -f environment.hf.yml       # includes huggingface-hub + aiohttp
conda activate personaplex-overfit
```

Then install PyTorch and the project:

```bash
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e ".[hub]"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

*(macOS Apple Silicon: `pip install torch==2.4.1 torchaudio==2.4.1` without `--index-url`.)*

> 🇻🇳 *`environment.hf.yml` = thêm HF Hub; `environment.yml` = offline.*

### Option B: Local venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e ".[hub]"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### Option C: One-command GPU server setup

```bash
bash scripts/setup_server_env.sh
```

Installs `ffmpeg`/`libsox-fmt-all` via apt, PyTorch 2.4.1, project requirements, HF CLI + `hf_transfer`, then prints a CUDA sanity check.

> 🇻🇳 *Script tự động cho GPU server Linux: apt → torch → deps → kiểm tra GPU.*

### Qwen backbone extra requirements

```bash
pip install "transformers>=4.57" "torch>=2.6"
```

Only needed when training with `backbone=qwen`.

---

## 2. Model & Dataset Download

Model checkpoint (3 files, ~17 GB total):

| File | Size | Role |
| :--- | :--- | :--- |
| `model.safetensors` | ~16.7 GB | PersonaPlex 7B transformer weights |
| `tokenizer-e351c8d8-checkpoint125.safetensors` | ~384 MB | Mimi audio codec (frozen) |
| `tokenizer_spm_32k_3.model` | ~552 KB | SentencePiece text tokenizer |

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
hf auth login   # requires accepting the terms at https://huggingface.co/nvidia/personaplex-7b-v1

# Way 1: direct hf download (fastest)
hf download nvidia/personaplex-7b-v1 \
  model.safetensors \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  tokenizer_spm_32k_3.model \
  --local-dir models/personaplex-7b-v1

hf download ngocbao220/personaplex-otospeech-prepared \
  --repo-type dataset \
  --local-dir prepared

# Way 2: repo tool
python -m tools.download_hf_assets \
  --assets-dir assets \
  --dataset-repo ngocbao220/personaplex-otospeech-prepared \
  --model-repo nvidia/personaplex-7b-v1 \
  --revision main

# Way 3: wrapper script (requires hf CLI + login)
bash scripts/download_hf_assets.sh --assets-dir assets
```

To publish your own prepared dataset back to the Hub:

```bash
python -m tools.publish_prepared --prepared-dir ../prepared --repo-id <your-account>/<dataset> --dry-run
```

> 🇻🇳 *Qwen backbone (`qwen_id: Qwen/Qwen3-8B`) tự tải qua transformers khi train — không cần tải tay.*

---

## 3. Pre-flight Checks

Run in order before training or demo.

### 3.1 Environment check

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MPS available (macOS):', torch.backends.mps.is_available())
for pkg in ['sphn', 'sounddevice', 'sentencepiece', 'safetensors', 'einops', 'accelerate', 'gradio']:
    __import__(pkg); print(f'{pkg}: OK')
"
```

### 3.2 Audio hardware (microphone & speakers)

```bash
python -m tools.interactive_cli --list-devices
```

Lists CoreAudio/ALSA input/output devices and exits.

### 3.3 Local checkpoint files

```bash
ls -lh ../models
```

Expect the 3 files from the table in §2.

### 3.4 Validate the training dataset

```bash
python -m tools.validate_dataset --config configs/config.yaml
```

Checks structure, stereo WAV (LEFT=agent, RIGHT=user), `words.json` alignment, voice prompts, and text prompts.

> ⚠️ **Note**: `configs/config.yaml` currently references `train: 104h` which no longer exists under `configs/train/` (available: `full`, `overfit`, `qwen`). If Hydra composition fails, edit that line to `train: full`, or use `configs/config_example.yaml` as a standalone config.

> 🇻🇳 *Tool hiện dùng `--config`; style cũ `data=overfit model=local` không còn áp dụng cho tools này.*

### 3.5 Inspect one sample's frame structure

```bash
python -m tools.inspect_sample --config configs/config.yaml --index 0 --device cuda
```

Prints Mimi frame structure, hybrid system prompt frames (voice + silence + text + silence), dialogue frames, and loss-masked token positions.

### 3.6 Unit tests

```bash
python -m unittest discover tests
```

### 3.7 1-step smoke test (forward + backward + LoRA gradient + reload check)

```bash
python -m personaplex_finetuning.train \
  --config configs/config.yaml \
  model=local data=otospeech data.prepared_dir=../prepared train=overfit \
  --smoke

# shortcut with the same behavior:
python -m tools.train_smoke --config configs/config.yaml model=local train=overfit
```

`--smoke` runs 1 optimizer step, verifies LoRA gradients are non-zero, base parameters stay frozen, saves the adapter, and reloads it to compare inference loss.

### 3.8 Mimi audio codec fidelity (esp. Vietnamese tones)

```bash
# single file
python -m tools.test_mimi --input path/to/sample.wav --model-root ../models

# whole directory of Vietnamese audio
python -m tools.test_mimi --input path/to/vietnamese_wavs/ \
  --num-codebooks 8 --num-samples 600 --output-dir outputs/mimi_vi_test
```

Scoring: `SI-SDR >= 12 dB` excellent · `8–12 dB` good · `< 8 dB` risk of lost tones/pitch → run Stage 0 Mimi fine-tuning first (§5.3).

> 🇻🇳 *Flags mới: `--num-samples`, `--seed`, `--save-samples`, `--save-all`.*

---

## 4. Training

Entry point: `python -m personaplex_finetuning.train`, config via `--config` plus `key=value` overrides.

### 4.1 Single GPU

```bash
# 10-sample overfit
python -m personaplex_finetuning.train \
  --config configs/config.yaml \
  model=local data=otospeech data.prepared_dir=../prepared train=overfit

# full 104h preset
python -m personaplex_finetuning.train \
  --config configs/config.yaml \
  model=local data=otospeech data.prepared_dir=../prepared train=full
```

Standalone (non-Hydra) config without preset groups:

```bash
python -m personaplex_finetuning.train --config configs/config_example.yaml train.max_steps=5000
```

### 4.2 Multi-GPU DDP

Launcher script (auto-sets `CUDA_VISIBLE_DEVICES` + process count):

```bash
# 2 GPUs, full preset
bash scripts/train_gpus.sh gpus=0,1 train=full model=server data=otospeech

# 4 GPUs, keep global batch = 1 GPU × accumulation:
bash scripts/train_gpus.sh gpus=0,1,2,3 \
  config=configs/config.yaml train=full train.gradient_accumulation_steps=2
```

Direct `accelerate launch`:

```bash
accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  -m personaplex_finetuning.train \
  --config configs/config.yaml train=full model=server data=otospeech
```

`train_gpus.sh` consumes `gpus=`, `n=/processes=`, `config=`/`--config`; everything else is passed through to the trainer.

### 4.3 Stage-wise training

```bash
# Stage 0 — adapt the Mimi codec (mono WAVs only, no transcript/alignment)
bash scripts/train_mimi.sh gpus=0 data_dir=path/to/vietnamese_wavs learning_rate=5e-5 max_steps=5000
# or directly:
python -m tools.train_mimi --data-dir path/to/vietnamese_wavs --learning-rate 5e-5 --max-steps 5000

# Stage 1 — temporal 7B only (depth transformer 100% frozen)
bash scripts/train_gpus.sh gpus=0,1 train=full model=server data=otospeech stage=temporal_only

# Stage 2 — depth transformer only (temporal 7B 100% frozen)
bash scripts/train_gpus.sh gpus=0,1 train=full model=server data=otospeech stage=depth_only

# Stage 3 — joint with dual learning rates
bash scripts/train_gpus.sh gpus=0,1 train=full model=server data=otospeech \
  stage=joint learning_rate=2e-5 depformer_lr=5e-6
```

### 4.4 Qwen backbone swap

```bash
# Qwen3-8B (LoRA on q_proj/v_proj)
python -m personaplex_finetuning.train --config configs/qwen3-8b.yaml

# Qwen3.5-9B (hybrid DeltaNet; LoRA on attn_qkv + ffn_gate/up/down)
python -m personaplex_finetuning.train --config configs/qwen35-9b.yaml
```

Or on the master config:

```bash
python -m personaplex_finetuning.train \
  --config configs/config.yaml \
  backbone=qwen qwen_id=Qwen/Qwen3-8B lora=qwen train=qwen
```

Requirements: `transformers>=4.57`, `torch>=2.6`. LoRA surfaces are auto-detected per family; override with `qwen_targets="q_proj,v_proj,k_proj,o_proj"`. The Qwen path uses three LR groups: `learning_rate` (backbone LoRA), `interface_lr` (depth I/O + adapters), `audio_embed_lr` (audio embedding table when `ft_embed=true`). QLoRA is not supported on the Qwen path.

> 🇻🇳 *Swap backbone không cần đổi code — chỉ đổi config. Qwen path 3 nhóm LR riêng.*

### 4.5 Resume & QLoRA

```bash
# resume (keep same GPU count and gradient_accumulation_steps)
bash scripts/train_gpus.sh gpus=4,5,6,7 train=full model=server \
  train.gradient_accumulation_steps=2 \
  --resume-from ../runs/full/train_YYYYMMDD_HHMMSS/checkpoints/checkpoint_000500

# 4-bit NF4 quantization (personaplex backbone only)
python -m personaplex_finetuning.train --config configs/config.yaml \
  lora=qlora train=overfit --qlora
```

Checkpoints restore adapter, optimizer, and scheduler; old adapter-only checkpoints still load with a fresh optimizer.

### 4.6 Override reference

The trainer accepts plain `key=value` overrides and maps them to config sections automatically:

| Override | Maps to | Notes |
| :--- | :--- | :--- |
| `model=`, `data=`, `train=`, `lora=` | config group | Preset: `local/server/qwen3-8b/qwen35-9b`, `otospeech/vietnamese`, `full/overfit/qwen`, `default/qlora/qwen` |
| `stage=` | `train.stage` | `joint`, `temporal_only`, `depth_only` |
| `freeze_depformer=true` / `freeze_tempformer=true` | `train.stage` | Shortcut for `temporal_only` / `depth_only` |
| `learning_rate=`, `max_steps=`, `output_dir=`, `warmup_steps=`, `eval_every_steps=`, `save_every_steps=`, `gradient_accumulation_steps=`, `gradient_checkpointing=`, `mixed_precision=` | `train.*` | |
| `depformer_lr=` / `tempformer_lr=` | depth / temporal LR | Dual-LR joint training |
| `interface_lr=`, `audio_embed_lr=` | `train.*` | Qwen path only |
| `rank=`, `alpha=`, `qlora=`, `quant_type=` | `lora.*` | |
| `qwen_rank=`, `qwen_alpha=`, `depformer_rank=`, `depformer_alpha=`, `ft_embed=`, `qwen_targets=` | `lora.*` | |
| `backbone=`, `qwen_id=`, `device=` | `model.*` | `device=` auto-prefixed to `model.device` |
| `window_seconds=`, `shuffle=` | `data.*` | |
| `data.prepared_dir=`, `data.val_manifest=`, `data.val_ratio=`, `data.static_chunking=`, `data.swap_roles_after_pass=`, `data.random_crop=`, `data.prompt_aug_prob=` | `data.*` | dotted overrides pass through to Hydra |
| `gpus=` / `gpu=` / `devices=` / `device_ids=` | ignored | Only meaningful for `scripts/train_gpus.sh` |

Full key reference for standalone configs: see `configs/config_example.yaml` (documents every parsed key, including loss weights which are hardcoded in `src/personaplex_finetuning/objective.py`).

### 4.7 Outputs

Each run writes `train_YYYYMMDD_HHMMSS/` under `output_dir`:

```
train_YYYYMMDD_HHMMSS/
├── checkpoints/checkpoint_NNNNNN/
│   ├── lora.safetensors        # adapter weights
│   ├── adapter.json            # rank/alpha/qlora metadata
│   └── training_state.pt       # optimizer + scheduler state
├── run.json                    # timing, peak GPU, best val loss, reload checks
└── tensorboard events
```

`data` presets `otospeech` / `vietnamese` split conversations into consecutive 30 s windows (last chunk zero-padded), shuffled per pass; with `swap_roles_after_pass` each chunk is trained once with LEFT as agent and once with RIGHT as agent. Every sample dir must contain `voice_prompt_left.wav`, `voice_prompt_right.wav`, `metadata.text_prompt_left`, and `metadata.text_prompt_right`; training fails fast before loading the model if any field is missing. Voice/text prompts are never shared between speakers.

---

## 5. Inference

### 5.1 Terminal CLI (mic ↔ speakers, full-duplex)

```bash
# recommended: config file
python -m tools.interactive_cli --config configs/demo.yaml

# or full flags
python -m tools.interactive_cli \
  --model-root ../models \
  --adapter ../runs/hf_overfit_10/checkpoints/checkpoint_000300 \
  --voice-prompt ../prepared/samples/conv_0001/voice_prompt_left.wav \
  --text-prompt "You enjoy having a good conversation. You are a helpful and friendly assistant." \
  --device cuda \
  --save-session-dir outputs/live_session

# launcher
bash scripts/run_cli_demo.sh --config configs/demo.yaml
```

Key flags:

| Flag | Meaning |
| :--- | :--- |
| `--config` | YAML/JSON config (see `configs/demo.yaml`) |
| `--model-root` | Local base checkpoint dir (required without config) |
| `--adapter` | LoRA checkpoint dir or `lora.safetensors`; omit for base model |
| `--personaplex-source` | PersonaPlex source root (defaults to bundled `src`) |
| `--voice-prompt` | Voice sample WAV/`.pt` |
| `--text-prompt` | Role/system prompt text or `.txt` path (auto-wrapped in `<system>` tags) |
| `--device` | `cuda` (default) or `cpu` |
| `--qlora` | 4-bit NF4 for low VRAM |
| `--lora-rank`, `--lora-alpha` | Needed only if `adapter.json` missing |
| `--greedy` / `--temp`, `--temp-text`, `--top-k`, `--top-k-text` | Decoding controls |
| `--input-device`, `--output-device` | Specific mic/speaker ID or name |
| `--input-wav` / `--output-wav` | File mode instead of microphone |
| `--save-session-dir` | Record `user.wav`, `agent.wav`, `dialogue_stereo.wav`, `transcript.txt` |
| `--list-devices` | List audio devices and exit |

### 5.2 Gradio web UI (with temporary public link)

Voice-preset dropdown, persona presets, hyperparameter tuning, browser microphone recording, live audio + transcript:

```bash
# config file (recommended)
python -m tools.interactive_web --config configs/demo.yaml

# launcher
bash scripts/run_web_demo.sh --config configs/demo.yaml

# with LoRA checkpoint + public share link
python -m tools.interactive_web \
  --model-root ../models \
  --adapter ../runs/hf_overfit_10/checkpoints/checkpoint_000300 \
  --share --port 8998
```

Access: local `http://localhost:8998`, or the temporary `https://xxxxxxxx.gradio.live` URL printed in the terminal (HTTPS, so microphone permission works; valid 72 h). `--no-share` disables the public link. Flags: `--config`, `--model-root`, `--adapter`, `--voice-prompt`, `--text-prompt`, `--device`, `--qlora`, `--host` (default `0.0.0.0`), `--port` (default `8998`), `--share` (default on), `--no-share`.

### 5.3 Inference smoke test (file-based)

Reloads a trained LoRA adapter, rebuilds the hybrid system prompt, and generates audio/text for one dataset sample:

```bash
python -m tools.inference_smoke \
  --config configs/config_example.yaml \
  --adapter runs/hf_overfit_10/checkpoints/checkpoint_000300/lora.safetensors \
  --index 0 \
  --start 42.5 \
  --output-dir outputs/smoke
```

Flags: `--config` (required), `--adapter` (required), `--index` (default `0`), `--start` (window start in seconds; must stay within the audio), `--input-path`/`--input-file` (optional external WAV/MP3 instead of sample audio), `--output-dir` (default `outputs/smoke`).

Outputs: `dialogue_original.wav` (source dialogue cropped to the inference window), `user.wav`, and base-model vs adapter-generated audio/text. `finetuned.txt` may be empty when free generation produces no valid text tokens — training loss measures teacher-forced prediction, not greedy inference transcripts.

---

## 6. Monitoring (TensorBoard)

```bash
tensorboard --logdir runs/hf_overfit_10 --host 0.0.0.0 --port 6006
```

Scalars: `loss/total`, `loss/text`, `loss/audio_semantic`, `loss/audio_nonsemantic`, gradient norm, LR, `samples_per_second`, GPU memory. Validation losses (`val/*`) when `eval_every_steps > 0`.

---

## 7. Full-Duplex-Bench

Offline benchmark implementing the Full-Duplex-Bench (FDB v1.0) protocol on the native 12.5 Hz frame-token stream — no external ASR model required. Four axes:

1. **Pause Handling** — patience during user hesitation (TOR ↓)
2. **Backchanneling** — natural short backchannels during long user turns (TOR ↓, Freq ↑, JSD ↓)
3. **Smooth Turn-Taking** — response speed after user stop (TOR ↑, Latency ↓)
4. **User Interruption** — yields and answers new content when interrupted (TOR ↑, Response Quality ↑, Latency ↓)

> 🇻🇳 *Benchmark offline, không cần ASR, chấm 4 trục song công theo chuẩn FDB v1.0.*

### 7.1 Quick compare demo (Base vs LoRA)

```bash
python -c "from personaplex_benchmark.cli import run_benchmark_cli; raise SystemExit(run_benchmark_cli())" \
  --compare_demo
```

*(The benchmark package has no `__main__` entry yet; the one-liner above is the current way to invoke it.)*

### 7.2 Benchmark on base checkpoint

```bash
# Linux CUDA
python -c "from personaplex_benchmark.cli import run_benchmark_cli; raise SystemExit(run_benchmark_cli())" \
  --model_root ../models \
  --model_name "PersonaPlex Base" \
  --fdb_dir benchmarks/datasets/fdb_v1/v1.0/extracted \
  --max_samples_per_task 10 \
  --output_dir benchmarks/results_fdb

# macOS Apple Silicon
PYTORCH_ENABLE_MPS_FALLBACK=1 python -c "from personaplex_benchmark.cli import run_benchmark_cli; raise SystemExit(run_benchmark_cli())" \
  --model_root ../models \
  --model_name "PersonaPlex Base" \
  --fdb_dir benchmarks/datasets/fdb_v1/v1.0/extracted \
  --max_samples_per_task 10 \
  --device mps \
  --output_dir benchmarks/results_fdb
```

### 7.3 Benchmark a fine-tuned LoRA checkpoint

```bash
python -c "from personaplex_benchmark.cli import run_benchmark_cli; raise SystemExit(run_benchmark_cli())" \
  --model_root ../models \
  --adapter ../checkpoints/lora_adapter.pt \
  --model_name "PersonaPlex LoRA" \
  --fdb_dir benchmarks/datasets/fdb_v1/v1.0/extracted \
  --max_samples_per_task 10 \
  --output_dir benchmarks/results_fdb
```

### 7.4 In-domain / prepared-data benchmark

```bash
python -c "from personaplex_benchmark.cli import run_benchmark_cli; raise SystemExit(run_benchmark_cli())" \
  --model_root ../models \
  --adapter ../checkpoints/lora_adapter.pt \
  --model_name "PersonaPlex-VI LoRA" \
  --prepared_dir ../prepared/samples \
  --max_samples_per_task 10 \
  --output_dir benchmarks/results_vi
```

### 7.5 Arguments

| Argument | Default | Meaning |
| :--- | :--- | :--- |
| `--model_root` | `models` | Base checkpoint dir |
| `--adapter` | `None` | LoRA weights (`lora.safetensors` or `.pt`); omit for base model |
| `--model_name` | `PersonaPlex Base` | Display name in report |
| `--fdb_dir` | `benchmarks/datasets/fdb_v1/v1.0/extracted` | Official FDB v1.0 dataset |
| `--prepared_dir` | `None` | Your prepared conversations (`words.json`, `conversation.wav`) |
| `--voice_prompt` | `prepared/samples/conv_0001/voice_prompt_left.wav` | Default voice prompt |
| `--gt_dist` | `benchmarks/datasets/fdb_v1/icc_gt_distribution.json` | ICC ground-truth distribution |
| `--output_dir` | `benchmarks/results` | Report destination |
| `--max_samples_per_task` | `10` | Samples per task group |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--mock` | off | Dry-run with a mock runner |
| `--compare_demo` | off | Base-vs-LoRA side-by-side demo |

### 7.6 Output

1. Colored Rich table in the terminal.
2. `{output_dir}/fdp-benchmark-result.md`
3. `{output_dir}/fdp-benchmark-result.json`

---

## 8. Repository Layout

```
personaplex-finetune-v2/
├── configs/                  # Hydra configs
│   ├── config.yaml           # master (model/data/lora/train groups)
│   ├── config_example.yaml   # standalone fully-commented config
│   ├── demo.yaml             # CLI/Web demo config
│   ├── qwen3-8b.yaml         # Qwen3-8B backbone swap master
│   ├── qwen35-9b.yaml        # Qwen3.5-9B backbone swap master
│   ├── model/  data/  lora/  train/   # preset groups
├── scripts/                  # launchers & setup
│   ├── setup_server_env.sh / setup_hf_server_env.sh
│   ├── download_hf_assets.sh
│   ├── train_gpus.sh  train_mimi.sh  train_overfit.sh
│   └── run_cli_demo.sh  run_web_demo.sh
├── src/
│   ├── moshi/                # vendored Moshi implementation
│   ├── personaplex_finetuning/   # config, data, LoRA, runtime, train, FSDP helpers
│   ├── personaplex_benchmark/    # Full-Duplex-Bench runner + metrics
│   └── tools/                # CLI tools (validate, inspect, demo, mimi, download)
├── tests/                    # unit test suite
├── pyproject.toml            # package: personaplex-finetuning
├── requirements.txt
├── environment.yml / environment.hf.yml
└── THIRD_PARTY_NOTICES.md
```

### Tools index

| Tool | Purpose |
| :--- | :--- |
| `tools.validate_dataset` | Validate prepared dataset structure |
| `tools.inspect_sample` | Inspect one sample's frame/loss-mask layout |
| `tools.train_smoke` | 1-step train smoke test shortcut |
| `tools.train_mimi` | Stage 0 Mimi codec fine-tuning |
| `tools.test_mimi` | Mimi encode→decode fidelity (SI-SDR/SNR) |
| `tools.compare_mimi_asr` | ASR comparison original vs Mimi-reconstructed audio |
| `tools.interactive_cli` | Terminal live duplex demo |
| `tools.interactive_web` | Gradio web demo |
| `tools.inference_smoke` | File-based LoRA inference check |
| `tools.download_hf_assets` | Download model + dataset from HF |
| `tools.publish_prepared` | Publish prepared dataset to HF |

### Tests

```bash
python -m unittest discover tests
```

Covers LoRA adapter resolution/loading, config parsing, objective loss weights, session frame stepping, sequence overflow, FSDP policies, Qwen backbones/tokenizer, benchmark metrics & runner, and the end-to-end pipeline smoke test.

---

## Notes

- **Loss weights** (`nonsemantic_audio_weight: 0.02`, `text_padding_weight: 0.3`, codebook 0 weight 1.0, user stream weight 0, system-prompt region masked) are hardcoded in `src/personaplex_finetuning/objective.py`, not configurable via YAML.
- **`configs/config.yaml`** currently references `train: 104h`; the available train presets are `full`, `overfit`, `qwen`. Edit the master config or override with `train=full` on the command line.
- QLoRA (`--qlora`, `lora.qlora: true`) applies to the PersonaPlex backbone only.
- Data presets in `configs/data/*.yaml` point at server paths — override with `data.prepared_dir=<your path>` for local runs.
- See `THIRD_PARTY_NOTICES.md` for bundled component licenses.
