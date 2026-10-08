"""PersonaPlex LoRA training path supporting Accelerate, DDP, BF16, and single-GPU execution."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from tqdm.auto import tqdm

from .config import Config, load_config
from .data import PreparedDataset, contiguous_chunks, sample_for_training_position
from .inference import _export_context, export_stereo, generate as generate_inference
from .lora import adapter_state_dict, inject_lora, load_adapter
from .objective import (
    normalize_text_padding_ids,
    stream_weights_torch,
    torch_weighted_cross_entropy_stats,
)
from .peft_adapter import (
    configure_qwen_trainable,
    detect_qwen_spec,
    inject_depformer_lora,
    inject_qwen_lora,
    load_qwen_adapter,
    qwen_adapter_state_dict,
)
from .runtime import QwenRuntimePaths, RuntimePaths, load_qwen_runtime, load_runtime
from .sequence import PersonaPlexTrainingExampleBuilder
from .train_utils import sample_position_for_step, write_val_scalars


def limit_cpu_threads(torch_module) -> int:
    """Keep training processes from consuming every CPU core."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    return 1


def seed_everything(seed: int, torch_module) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def create_run_dir(output_root: Path, smoke: bool) -> Path:
    """Create a unique, timestamped run directory below the configured root."""
    label = "smoke" if smoke else "train"
    run_dir = output_root / f"{label}_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_example(config: Config, sample, runtime, random_crop: bool = False, rng=None):
    if random_crop and getattr(config, "random_crop", False):
        effective_sample = sample.dynamic_sample(
            config.window_seconds, random_crop=True, prompt_aug_prob=getattr(config, "prompt_aug_prob", 0.0), rng=rng
        )
    else:
        effective_sample = sample
    builder = PersonaPlexTrainingExampleBuilder(
        runtime.codec, runtime.tokenizer, runtime.initial_tokens, runtime.zero_token,
        text_prompt_template=config.text_prompt_template,
    )
    # PersonaPlex/Qwen forward_train applies the native per-stream delays once.
    return builder.build(effective_sample)


def effective_global_batch_size(per_process_batch_size: int, num_processes: int, gradient_accumulation_steps: int) -> int:
    """Return the number of samples contributing to one optimizer update."""
    if min(per_process_batch_size, num_processes, gradient_accumulation_steps) < 1:
        raise ValueError("batch size, process count, and accumulation steps must all be positive")
    return per_process_batch_size * num_processes * gradient_accumulation_steps


def step_optimizer_if_ready(accelerator, optimizer, scheduler, trainable) -> float:
    """Commit an accumulated DDP update only at Accelerate's sync boundary."""
    if not accelerator.sync_gradients:
        return 0.0
    grad_norm = float(accelerator.clip_grad_norm_(trainable, 1.0))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def deterministic_crop_rng(seed: int, process_index: int, micro_step: int) -> random.Random:
    """Make dynamic crops reproducible across resume without saving Python RNG state."""
    return random.Random(seed + (process_index * 1_000_003) + micro_step)


def sample_index_for_rank(micro_step: int, process_index: int, num_processes: int, sample_count: int) -> int:
    """Select a disjoint rank-local sample before wrapping around the dataset."""
    if micro_step < 0 or process_index < 0 or process_index >= num_processes or sample_count < 1:
        raise ValueError("invalid distributed sample selection inputs")
    return (micro_step * num_processes + process_index) % sample_count


def write_rank_info(run_dir: Path, process_index: int, num_processes: int, device, sample_count: int, peak_gpu_bytes: int | None = None) -> Path:
    """Emit one non-contended runtime record per DDP rank for GPU verification."""
    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    physical_gpu = visible_devices[process_index] if process_index < len(visible_devices) else None
    info = {
        "rank": process_index,
        "world_size": num_processes,
        "device": str(device),
        "physical_gpu": physical_gpu,
        "first_sample_index": sample_index_for_rank(0, process_index, num_processes, sample_count),
    }
    if peak_gpu_bytes is not None:
        info["peak_gpu_bytes"] = peak_gpu_bytes
    path = run_dir / "ranks" / f"rank_{process_index:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return path


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    import math
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable activation checkpointing on StreamingTransformer layers without breaking streaming inference."""
    import torch.utils.checkpoint
    for module in model.modules():
        if module.__class__.__name__ == "StreamingTransformer":
            orig_forward = module.forward
            def make_checkpointed_forward(mod, orig_fwd):
                def checkpointed_forward(x: torch.Tensor, *args, **kwargs):
                    if not mod.training:
                        return orig_fwd(x, *args, **kwargs)
                    B, T, C = x.shape
                    state = mod._streaming_state
                    if state is None:
                        offset = torch.zeros(1, dtype=torch.long, device=x.device)
                    else:
                        offset = state.offset
                    if mod.positional_embedding in {"sin", "sin_rope"}:
                        from moshi.modules.transformer import create_sin_embedding
                        positions = torch.arange(T, device=x.device).view(1, -1, 1)
                        positions = positions + offset.view(-1, 1, 1)
                        pos_emb = create_sin_embedding(
                            positions, C, max_period=mod.max_period, dtype=x.dtype
                        )
                        x = x + mod.positional_scale * pos_emb
                    for layer in mod.layers:
                        x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                    if state is not None:
                        state.offset.add_(T)
                    return x
                return checkpointed_forward
            module.forward = make_checkpointed_forward(module, orig_forward)


def model_forward_train(model, codes: torch.Tensor):
    """LMModel exposes training through forward_train."""
    if hasattr(model, "forward_train"):
        return model.forward_train(codes)
    return model(codes)


def write_tensorboard_scalars(writer, record: dict[str, float | int], trainable_parameters: int, cpu_threads: int) -> None:
    step = int(record["step"])
    for name, value in (
        ("loss/total", record["loss/total"]),
        ("loss/text", record["loss/text"]),
        ("loss/audio_semantic", record["loss/audio_semantic"]),
        ("loss/audio_nonsemantic", record["loss/audio_nonsemantic"]),
        ("train/learning_rate", record["lr"]),
        ("train/gradient_norm", record["grad_norm"]),
        ("system/gpu_peak_bytes", record["gpu_peak_bytes"]),
        ("system/trainable_parameters", trainable_parameters),
        ("system/cpu_threads", cpu_threads),
    ):
        writer.add_scalar(name, value, step)


def tokenizer_text_padding_ids(tokenizer, include_end_padding: bool = False) -> tuple[int, ...]:
    """Text IDs treated as padding (down-weighted in the loss).

    EPAD marks a word onset; free-running generation can only start a word
    after the model itself emits EPAD, so by default EPAD stays a full-weight
    target. ``include_end_padding=True`` restores the legacy behavior.
    """
    ids = [int(tokenizer.padding_id)]
    end_padding_id = getattr(tokenizer, "end_padding_id", None)
    if include_end_padding and end_padding_id is not None and int(end_padding_id) not in ids:
        ids.append(int(end_padding_id))
    return normalize_text_padding_ids(ids)


def loss_components(
    model_output,
    codes,
    example,
    text_padding_id,
    torch_module,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 0.3,
    *,
    user_loss: bool = False,
):
    """Weighted losses where all audio groups share one mean denominator.

    Sharing the denominator keeps ``nonsemantic_audio_weight`` effective: the
    acoustic codebooks stay downweighted instead of cancelling inside their own
    group mean. Prompt/padding positions carry zero weight.
    """
    labels_tensor = torch_module.tensor(example.labels, dtype=torch_module.long, device=codes.device)
    mask_tensor = torch_module.tensor(example.loss_mask, dtype=torch_module.bool, device=codes.device)
    weights = stream_weights_torch(
        labels_tensor,
        mask_tensor,
        text_padding_id,
        text_padding_weight=text_padding_weight,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        user_loss=user_loss,
    )

    text_target = labels_tensor[0]
    text_mask = model_output.text_mask
    if text_mask.ndim == 3:
        text_mask = text_mask[0, 0]
    text_weight = weights[0] * text_mask.to(weights.dtype)
    text_stats = torch_weighted_cross_entropy_stats(
        model_output.text_logits.reshape(-1, model_output.text_logits.shape[-1]),
        text_target.reshape(-1),
        text_weight.reshape(-1),
    )
    text_loss = text_stats[0] / text_stats[1].clamp_min(1e-12)

    audio_target = labels_tensor[1:17]
    model_mask = model_output.mask
    if model_mask.ndim == 3:
        model_mask = model_mask[0]
    audio_weights = weights[1:17] * model_mask.to(weights.dtype)

    def audio_group(start: int, end: int):
        return torch_weighted_cross_entropy_stats(
            model_output.logits[:, start:end].reshape(-1, model_output.logits.shape[-1]),
            audio_target[start:end].reshape(-1),
            audio_weights[start:end].reshape(-1),
        )

    agent_semantic = audio_group(0, 1)
    agent_acoustic = audio_group(1, 8)
    user_semantic = audio_group(8, 9)
    user_acoustic = audio_group(9, 16)
    audio_denominator = sum(
        stats[1] for stats in (agent_semantic, agent_acoustic, user_semantic, user_acoustic)
    ).clamp_min(1e-12)

    components = {
        "text": text_loss,
        "audio_semantic": (agent_semantic[0] + user_semantic[0]) / audio_denominator,
        "audio_nonsemantic": (agent_acoustic[0] + user_acoustic[0]) / audio_denominator,
    }
    total = text_loss + (
        agent_semantic[0] + agent_acoustic[0] + user_semantic[0] + user_acoustic[0]
    ) / audio_denominator
    return total, components


def one_step(config: Config, runtime, example, optimizer=None):
    codes = torch.tensor(example.input_codes, dtype=torch.long, device=config.device).unsqueeze(0)
    output = model_forward_train(runtime.model, codes)
    total, components = loss_components(
        output, codes, example,
        tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding), torch,
        config.first_codebook_weight_multiplier, config.text_padding_weight,
        user_loss=config.user_loss,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        trainable = [p for p in runtime.model.parameters() if p.requires_grad]
        if not any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable):
            raise RuntimeError("LoRA gradients are all zero")
        if any(p.grad is not None for p in runtime.model.parameters() if not p.requires_grad):
            raise RuntimeError("frozen base parameter received a gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)
    return total, components, float(grad_norm)


def save_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    optimizer_step: int,
    gradient_accumulation_steps: int,
    num_processes: int,
) -> Path:
    """Persist enough state to continue the optimizer trajectory exactly."""
    path = checkpoint_dir / "training_state.pt"
    torch.save(
        {
            "format_version": 1,
            "optimizer_step": optimizer_step,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_processes": num_processes,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )
    return path


def load_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    gradient_accumulation_steps: int,
    num_processes: int,
) -> int | None:
    """Restore an optimizer checkpoint, rejecting incompatible DDP topology."""
    path = checkpoint_dir / "training_state.pt"
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("gradient_accumulation_steps") != gradient_accumulation_steps:
        raise RuntimeError("resume checkpoint gradient_accumulation_steps differs from this run")
    if state.get("num_processes") != num_processes:
        raise RuntimeError("resume checkpoint num_processes differs from this run")
    optimizer.load_state_dict(state["optimizer"])
    saved_scheduler = state.get("scheduler")
    if scheduler is None and saved_scheduler is not None:
        raise RuntimeError("resume checkpoint has scheduler state but this run has no scheduler")
    if scheduler is not None and saved_scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    return int(state["optimizer_step"])


def save_adapter(run_dir: Path, model, config: Config, step: int, optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1) -> Path:
    path = run_dir / "checkpoints" / f"checkpoint_{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    state = qwen_adapter_state_dict(model) if hasattr(model, "qwen") else adapter_state_dict(model)
    save_file(state, str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha,
             "gradient_accumulation_steps": gradient_accumulation_steps, "num_processes": num_processes},
            indent=2,
        )
    )
    if optimizer is not None:
        save_training_state(path, optimizer, scheduler, step, gradient_accumulation_steps, num_processes)
    return adapter


def save_best_adapter(
    run_dir: Path, model, config: Config, step: int, val_loss: float,
    optimizer=None, scheduler=None, gradient_accumulation_steps: int = 1, num_processes: int = 1,
) -> Path:
    path = run_dir / "checkpoints" / "best"
    path.mkdir(parents=True, exist_ok=True)
    adapter = path / "lora.safetensors"
    state = qwen_adapter_state_dict(model) if hasattr(model, "qwen") else adapter_state_dict(model)
    save_file(state, str(adapter))
    (path / "adapter.json").write_text(
        json.dumps(
            {"step": step, "val_loss": val_loss, "model_root": str(config.model_root), "rank": config.lora_rank, "alpha": config.lora_alpha,
             "gradient_accumulation_steps": gradient_accumulation_steps, "num_processes": num_processes},
            indent=2,
        )
    )
    if optimizer is not None:
        save_training_state(path, optimizer, scheduler, step, gradient_accumulation_steps, num_processes)
    return adapter


def evaluate_validation(config: Config, runtime, val_samples: list, accelerator: Accelerator) -> dict[str, float]:
    if not val_samples:
        return {}
    unwrapped = accelerator.unwrap_model(runtime.model)
    unwrapped.eval()
    totals = {"total": 0.0, "text": 0.0, "semantic": 0.0, "nonsemantic": 0.0}
    count = 0
    device = accelerator.device
    with torch.no_grad():
        for sample_index, sample in enumerate(val_samples):
            if sample_index % accelerator.num_processes != accelerator.process_index:
                continue
            example = build_example(config, sample, runtime, random_crop=False)
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)
            output = unwrapped(codes)
            total, comps = loss_components(
                output, codes, example,
                tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding), torch,
                config.first_codebook_weight_multiplier, config.text_padding_weight,
                user_loss=config.user_loss,
            )
            totals["total"] += float(total.detach())
            totals["text"] += float(comps["text"].detach())
            totals["semantic"] += float(comps["audio_semantic"].detach())
            totals["nonsemantic"] += float(comps["audio_nonsemantic"].detach())
            count += 1
    unwrapped.train()
    reduced = accelerator.reduce(
        torch.tensor([totals["total"], totals["text"], totals["semantic"], totals["nonsemantic"], count], device=device),
        reduction="sum",
    )
    total_count = float(reduced[4].item())
    if total_count == 0:
        raise RuntimeError("validation has no samples after rank partitioning")
    return {
        "val/loss_total": float(reduced[0].item()) / total_count,
        "val/loss_text": float(reduced[1].item()) / total_count,
        "val/loss_audio_semantic": float(reduced[2].item()) / total_count,
        "val/loss_audio_nonsemantic": float(reduced[3].item()) / total_count,
    }


def verify_reloaded_adapter(config: Config, sample, adapter: Path) -> float:
    """Load base + adapter in a fresh model object and return teacher-forced loss."""
    if config.backbone == "qwen":
        fresh = load_qwen_runtime(
            QwenRuntimePaths(config.model_root, config.personaplex_source, config.qwen_model_id),
            config.device,
        )
        targets_override = tuple(config.qwen_targets.split(",")) if config.qwen_targets else None
        inject_qwen_lora(
            fresh.model,
            rank=config.lora_qwen_rank,
            alpha=config.lora_qwen_alpha,
            targets=targets_override,
        )
        inject_depformer_lora(fresh.model, rank=config.lora_depformer_rank, alpha=config.lora_depformer_alpha)
        configure_qwen_trainable(fresh.model, ft_embed=config.ft_embed)
        load_qwen_adapter(fresh.model, adapter)
        fresh.model.eval()
        example = build_example(config, sample, fresh)
        with torch.no_grad():
            total, _, _ = one_step(config, fresh, example)
        return float(total)
    fresh = load_runtime(RuntimePaths(config.model_root, config.personaplex_source), config.device, config.qlora, config.quant_type)
    stage = getattr(config, "train_stage", "joint").lower()
    if stage in {"temporal_only", "freeze_depformer"}:
        prefixes = ("transformer",)
    elif stage in {"depth_only", "freeze_tempformer"}:
        prefixes = ("depformer",)
    else:
        prefixes = ("transformer", "depformer") if config.depformer_learning_rate is not None else ("transformer",)
    inject_lora(fresh.model, config.lora_rank, config.lora_alpha, prefixes=prefixes)
    load_adapter(fresh.model, adapter)
    fresh.model.eval()
    example = build_example(config, sample, fresh)
    with torch.no_grad():
        total, _, _ = one_step(config, fresh, example)
    return float(total)


def run(
    config: Config,
    smoke: bool = False,
    resume_from: str | None = None,
    reset_scheduler: bool = False,
) -> Path | None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install the project requirements before training") from exc

    if smoke and config.shuffle:
        raise ValueError("overfit/smoke configuration must set data.shuffle: false")

    accum_steps = max(1, config.gradient_accumulation_steps)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=accum_steps,
    )
    device = accelerator.device
    global_batch_size = effective_global_batch_size(1, accelerator.num_processes, accum_steps)
    cpu_threads = limit_cpu_threads(torch)
    set_seed(config.seed + accelerator.process_index)

    # Dataset loading
    if config.val_manifest:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = PreparedDataset(config.val_manifest, config.window_seconds).load()
    elif config.eval_every_steps > 0:
        train_samples, val_samples = PreparedDataset(config.manifest, config.window_seconds).split(
            val_ratio=config.val_ratio, seed=config.seed
        )
    else:
        train_samples = PreparedDataset(config.manifest, config.window_seconds).load()
        val_samples = []

    if config.static_chunking:
        train_samples = contiguous_chunks(train_samples, config.window_seconds)
        val_samples = contiguous_chunks(val_samples, config.window_seconds) if val_samples else []
        if config.swap_roles_after_pass and not smoke:
            # Fail before model loading, rather than halfway through the first right-speaker pass.
            for sample in train_samples:
                sample.swapped_roles()

    # Run directory creation (coordinated across ranks)
    run_dir = None
    if accelerator.is_main_process:
        run_dir = create_run_dir(config.output_dir, smoke)
        config_record = {
            "event": "configuration",
            "seed": config.seed,
            "backbone": config.backbone,
            "model_root": str(config.model_root),
            "personaplex_source": str(config.personaplex_source),
            "manifest": str(config.manifest),
            "output_dir": str(run_dir),
            "window_seconds": config.window_seconds,
            "max_steps": 1 if smoke else config.max_steps,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "qlora": config.qlora,
            "quant_type": config.quant_type if config.qlora else None,
            "device": str(device),
            "num_processes": accelerator.num_processes,
            "cpu_threads": cpu_threads,
            "gradient_accumulation_steps": accum_steps,
            "per_device_batch_size": 1,
            "global_batch_size": global_batch_size,
            "warmup_steps": config.warmup_steps,
            "eval_every_steps": config.eval_every_steps,
            "save_every_steps": config.save_every_steps,
            "random_crop": config.random_crop,
            "prompt_aug_prob": config.prompt_aug_prob,
            "static_chunking": config.static_chunking,
            "swap_roles_after_pass": config.swap_roles_after_pass,
            "gradient_checkpointing": config.gradient_checkpointing,
            "mixed_precision": config.mixed_precision,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "num_train_role_views": len(train_samples) * (2 if config.swap_roles_after_pass else 1),
        }
        (run_dir / "config.json").write_text(json.dumps(config_record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(config_record))
        print(f"[Main Process] Active across {accelerator.num_processes} process(es) on device {device}")

    if accelerator.num_processes > 1:
        run_dir_list = [str(run_dir) if accelerator.is_main_process else ""]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(run_dir_list, src=0)
        run_dir = Path(run_dir_list[0])

    if run_dir is not None:
        write_rank_info(run_dir, accelerator.process_index, accelerator.num_processes, device, len(train_samples))
    accelerator.wait_for_everyone()

    # Sequential model loading across ranks to avoid host CPU RAM spikes
    for i in range(accelerator.num_processes):
        if accelerator.process_index == i:
            if config.backbone == "qwen":
                runtime = load_qwen_runtime(
                    QwenRuntimePaths(config.model_root, config.personaplex_source, config.qwen_model_id),
                    str(device),
                )
            else:
                runtime = load_runtime(
                    RuntimePaths(config.model_root, config.personaplex_source),
                    str(device),
                    config.qlora,
                    config.quant_type,
                )
        accelerator.wait_for_everyone()

    # Drop fixed-window samples whose agent text cannot fit the frame budget:
    # silently truncating tokens corrupts the alignment target.
    if not config.random_crop:
        frame_rate = runtime.codec.frame_rate
        word_token_cache: dict[str, int] = {}

        def _text_fits(sample) -> bool:
            frames = max(0, int(round((sample.window_end_sec - sample.window_start_sec) * frame_rate)))
            required = 0
            for word in sample.words:
                if word.speaker != "agent" or not sample.window_start_sec <= word.start < sample.window_end_sec:
                    continue
                count = word_token_cache.get(word.word)
                if count is None:
                    count = len(runtime.tokenizer.encode(" " + word.word))
                    word_token_cache[word.word] = count
                required += count
            return required <= frames

        train_kept = [sample for sample in train_samples if _text_fits(sample)]
        val_kept = [sample for sample in val_samples if _text_fits(sample)]
        skipped_train = len(train_samples) - len(train_kept)
        skipped_val = len(val_samples) - len(val_kept)
        if not train_kept:
            raise RuntimeError("chunk filter removed every training sample; check window_seconds/tokenizer")
        train_samples = train_kept
        val_samples = val_kept
        if accelerator.is_main_process and (skipped_train or skipped_val):
            print(
                f"[Chunk filter] train kept={len(train_samples)} skipped={skipped_train}; "
                f"val kept={len(val_samples)} skipped={skipped_val} (text overflow)"
            )

    # Inject LoRA with Stage-Wise Freezing support
    if config.backbone == "qwen":
        spec = detect_qwen_spec(runtime.model)
        targets_override = tuple(config.qwen_targets.split(",")) if config.qwen_targets else None
        targets = inject_qwen_lora(
            runtime.model,
            rank=config.lora_qwen_rank,
            alpha=config.lora_qwen_alpha,
            targets=targets_override,
        )
        targets += inject_depformer_lora(
            runtime.model, rank=config.lora_depformer_rank, alpha=config.lora_depformer_alpha
        )
        configure_qwen_trainable(runtime.model, ft_embed=config.ft_embed)
        if accelerator.is_main_process:
            print(
                f"[Qwen Swap] family={spec.family} (config: {runtime.qwen_family}), "
                f"lora targets={spec.lora_targets if targets_override is None else targets_override}, "
                f"rank={config.lora_qwen_rank}, alpha={config.lora_qwen_alpha}"
            )
    else:
        stage = getattr(config, "train_stage", "joint").lower()
        if stage in {"temporal_only", "freeze_depformer"}:
            lora_prefixes = ("transformer",)
            if hasattr(runtime.model, "depformer"):
                runtime.model.depformer.requires_grad_(False)
            if accelerator.is_main_process:
                print("[Stage-Wise Training] Active Stage: TEMPORAL ONLY (Depth Transformer / Depformer is 100% frozen).")
        elif stage in {"depth_only", "freeze_tempformer"}:
            lora_prefixes = ("depformer",)
            if hasattr(runtime.model, "transformer"):
                runtime.model.transformer.requires_grad_(False)
            if accelerator.is_main_process:
                print("[Stage-Wise Training] Active Stage: DEPTH ONLY (Temporal 7B Transformer is 100% frozen).")
        else:
            lora_prefixes = ("transformer", "depformer") if config.depformer_learning_rate is not None else ("transformer",)
            if accelerator.is_main_process:
                print("[Stage-Wise Training] Active Stage: JOINT (Both Temporal & Depth active).")

        targets = inject_lora(runtime.model, config.lora_rank, config.lora_alpha, prefixes=lora_prefixes)

    # Optional gradient checkpointing
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(runtime.model)
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled on transformer layers.")

    # Load LoRA weights before DDP wrapping; optimizer state is restored after wrapping.
    resume_checkpoint_dir = None
    adapter_resume_step = 0
    if resume_from:
        resume_path = Path(resume_from)
        adapter_file = resume_path if resume_path.is_file() else resume_path / "lora.safetensors"
        if not adapter_file.is_file():
            raise FileNotFoundError(f"resume adapter does not exist: {adapter_file}")
        resume_checkpoint_dir = adapter_file.parent
        if accelerator.is_main_process:
            print(f"Resuming weights from {adapter_file}")
        if hasattr(runtime.model, "qwen"):
            load_qwen_adapter(runtime.model, adapter_file)
        else:
            load_adapter(runtime.model, adapter_file)
        meta_file = adapter_file.parent / "adapter.json"
        if meta_file.is_file():
            try:
                adapter_resume_step = int(json.loads(meta_file.read_text(encoding="utf-8")).get("step", 0))
                if accelerator.is_main_process:
                    print(f"Adapter checkpoint is at optimizer step {adapter_resume_step}")
            except Exception:
                pass

    # Alias LMModel.forward to forward_train for training execution
    if config.backbone != "qwen":
        from moshi.models.lm import LMModel
        LMModel.forward = LMModel.forward_train

    trainable = [p for p in runtime.model.parameters() if p.requires_grad]
    if accelerator.is_main_process:
        print(f"LoRA targets: {len(targets)}; trainable parameters: {sum(p.numel() for p in trainable):,}")

    temp_lr = config.learning_rate
    dep_lr = config.depformer_learning_rate
    if config.backbone == "qwen":
        dep_lr = dep_lr if dep_lr is not None else config.interface_lr
        groups: dict[float, list] = {}
        for name, parameter in runtime.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("qwen."):
                lr = temp_lr
            elif name.startswith("emb."):
                lr = config.audio_embed_lr
            else:
                lr = dep_lr
            groups.setdefault(lr, []).append(parameter)
        optimizer = torch.optim.AdamW(
            [{"params": params, "lr": lr} for lr, params in groups.items()], weight_decay=0.0
        )
        if accelerator.is_main_process:
            print(
                "Qwen optimizer groups: "
                + ", ".join(f"{lr:.1e}: {len(params)} params" for lr, params in sorted(groups.items()))
            )
    elif dep_lr is not None and dep_lr != temp_lr:
        temp_params = [p for n, p in runtime.model.named_parameters() if p.requires_grad and "depformer" not in n]
        dep_params = [p for n, p in runtime.model.named_parameters() if p.requires_grad and "depformer" in n]
        param_groups = []
        if temp_params:
            param_groups.append({"params": temp_params, "lr": temp_lr})
        if dep_params:
            param_groups.append({"params": dep_params, "lr": dep_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        if accelerator.is_main_process:
            print(f"Using Dual Learning Rates -> Temporal Transformer: {temp_lr:.2e}, Depth Transformer: {dep_lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=temp_lr, weight_decay=0.0)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    max_steps = 1 if smoke else config.max_steps
    scheduler = None
    if config.warmup_steps > 0 and not smoke:
        scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, max_steps)

    # Prepare model, optimizer, scheduler with Accelerator (DDP wrapping)
    runtime.model, optimizer = accelerator.prepare(runtime.model, optimizer)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    start_step = adapter_resume_step
    if resume_checkpoint_dir is not None:
        restored_step = load_training_state(
            resume_checkpoint_dir,
            optimizer,
            scheduler,
            accum_steps,
            accelerator.num_processes,
        )
        if restored_step is not None:
            start_step = restored_step
            if accelerator.is_main_process:
                print(f"Restored optimizer and scheduler state at optimizer step {start_step}")
        elif accelerator.is_main_process:
            print("Resume checkpoint has no training_state.pt; resuming adapter weights with a fresh optimizer.")
    if reset_scheduler and resume_checkpoint_dir is not None:
        if len(optimizer.param_groups) != len(base_lrs):
            raise RuntimeError("reset-scheduler needs the same optimizer group count as the checkpoint run")
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr
        inner = getattr(scheduler, "scheduler", scheduler)
        if inner is not None:
            inner.last_epoch = -1
        if accelerator.is_main_process:
            print(f"Reset LR schedule from config (base LRs {[f'{v:.1e}' for v in base_lrs]}); data continues at step {start_step}")
    if start_step > max_steps:
        raise RuntimeError(f"resume checkpoint step {start_step} exceeds train.max_steps={max_steps}")

    writer = None
    log_file = None
    if accelerator.is_main_process and run_dir is not None:
        log_path = run_dir / "metrics.jsonl"
        log_file = log_path.open("w", encoding="utf-8")
        tensorboard_dir = run_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_text("configuration", json.dumps(config_record, indent=2), 0)

    saved = None
    best_saved = None
    best_val_loss = float("inf")
    last_record = None
    reload_checks: list[dict[str, float | int]] = []
    infer_outputs: list[str] = []
    started = time.monotonic()

    try:
        start_micro_step = start_step * accum_steps
        max_micro_steps = max_steps * accum_steps
        progress = tqdm(total=max_steps, initial=start_step, desc="training (DDP)", unit="update", dynamic_ncols=True) if accelerator.is_main_process else None
        last_update_time = time.monotonic()

        for micro_step in range(start_micro_step, max_micro_steps):
            global_position = sample_position_for_step(
                micro_step, accelerator.process_index, accelerator.num_processes
            )
            if config.static_chunking:
                current_sample = sample_for_training_position(
                    train_samples,
                    global_position,
                    seed=config.seed,
                    shuffle=config.shuffle and not smoke,
                    swap_roles=config.swap_roles_after_pass and not smoke,
                )
            else:
                sample_idx = sample_index_for_rank(
                    micro_step, accelerator.process_index, accelerator.num_processes, len(train_samples)
                )
                current_sample = train_samples[sample_idx]
            example = build_example(
                config,
                current_sample,
                runtime,
                random_crop=config.random_crop and not config.static_chunking and not smoke,
                rng=deterministic_crop_rng(config.seed, accelerator.process_index, micro_step),
            )
            codes = torch.tensor(example.input_codes, dtype=torch.long, device=device).unsqueeze(0)

            with accelerator.accumulate(runtime.model):
                output = runtime.model(codes)
                total, components = loss_components(
                    output, codes, example,
                    tokenizer_text_padding_ids(runtime.tokenizer, config.epad_as_padding), torch,
                    config.first_codebook_weight_multiplier, config.text_padding_weight,
                    user_loss=config.user_loss,
                )
                accelerator.backward(total)

                grad_norm = step_optimizer_if_ready(accelerator, optimizer, scheduler, trainable)

            if not accelerator.sync_gradients:
                continue

            optimizer_step = (micro_step + 1) // accum_steps

            reduced_total = accelerator.reduce(total, reduction="mean")
            reduced_text = accelerator.reduce(components["text"], reduction="mean")
            reduced_sem = accelerator.reduce(components["audio_semantic"], reduction="mean")
            reduced_nonsem = accelerator.reduce(components["audio_nonsemantic"], reduction="mean")

            if accelerator.is_main_process:
                record = {
                    "step": optimizer_step,
                    "micro_step": micro_step + 1,
                    "loss/total": float(reduced_total.detach()),
                    "loss/text": float(reduced_text.detach()),
                    "loss/audio_semantic": float(reduced_sem.detach()),
                    "loss/audio_nonsemantic": float(reduced_nonsem.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "global_batch_size": global_batch_size,
                    "samples_per_second": global_batch_size / max(time.monotonic() - last_update_time, 1e-9),
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0,
                }
                last_update_time = time.monotonic()
                last_record = record
                if log_file:
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                if writer:
                    write_tensorboard_scalars(writer, record, sum(p.numel() for p in trainable), cpu_threads)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(loss=f"{record['loss/total']:.4f}", grad=f"{grad_norm:.3f}")
                if max_steps == 1:
                    tqdm.write(json.dumps({"event": "training_step", **record}))

            # Validation evaluation
            if val_samples and config.eval_every_steps > 0 and (optimizer_step % config.eval_every_steps == 0 or optimizer_step == max_steps):
                accelerator.wait_for_everyone()
                val_metrics = evaluate_validation(config, runtime, val_samples, accelerator)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    write_val_scalars(writer, val_metrics, optimizer_step)
                    val_loss = val_metrics.get("val/loss_total", float("inf"))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        unwrapped = accelerator.unwrap_model(runtime.model)
                        best_saved = save_best_adapter(
                            run_dir, unwrapped, config, optimizer_step, val_loss, optimizer, scheduler,
                            gradient_accumulation_steps=accum_steps, num_processes=accelerator.num_processes,
                        )
                        tqdm.write(json.dumps({"event": "new_best_val_loss", "step": optimizer_step, "val_loss": val_loss}))

            # Periodic saving & smoke reload verification
            save_interval = 1 if smoke else config.save_every_steps
            if optimizer_step % save_interval == 0 or optimizer_step == max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(runtime.model)
                    saved = save_adapter(
                        run_dir, unwrapped, config, optimizer_step, optimizer, scheduler,
                        gradient_accumulation_steps=accum_steps, num_processes=accelerator.num_processes,
                    )
                    if smoke:
                        reload_loss = verify_reloaded_adapter(config, train_samples[micro_step % len(train_samples)], saved)
                        if abs(reload_loss - record["loss/total"]) > 0.05:
                            raise RuntimeError(f"reloaded adapter loss drifted: {reload_loss} vs {record['loss/total']}")
                        reload_checks.append({"step": optimizer_step, "loss": reload_loss})
                accelerator.wait_for_everyone()

            # Free-running inference samples (main process only)
            if (
                config.infer_every_steps > 0
                and optimizer_step % config.infer_every_steps == 0
                and accelerator.is_main_process
            ):
                infer_dir = run_dir / "infer" / f"step_{optimizer_step:06d}"
                unwrapped = accelerator.unwrap_model(runtime.model)
                was_training = unwrapped.training
                infer_error = None
                unwrapped.eval()
                try:
                    for sample in (val_samples or train_samples)[: config.infer_samples]:
                        try:
                            sample_dir = infer_dir / sample.sample_id
                            sample_dir.mkdir(parents=True, exist_ok=True)
                            _export_context(sample, sample_dir)
                            generate_inference(
                                config, sample, sample_dir / "agent.wav", sample_dir / "agent.txt",
                                adapter=None, runtime=runtime, model=unwrapped, gen_cfg=config.generation,
                            )
                            export_stereo(sample_dir)
                            infer_outputs.append(str(sample_dir))
                        except Exception as exc:  # a failed sample must never kill training
                            infer_error = f"{sample.sample_id}: {exc}"
                            tqdm.write(json.dumps({"event": "inference_error", "step": optimizer_step, "error": infer_error}))
                finally:
                    unwrapped.train(was_training)
                tqdm.write(json.dumps({
                    "event": "inference_samples",
                    "step": optimizer_step,
                    "infer_error": infer_error,
                    "outputs": infer_outputs[-config.infer_samples:],
                }))

        if log_file:
            log_file.close()
        if progress is not None:
            progress.close()
    finally:
        if writer:
            writer.close()

    if run_dir is not None:
        rank_peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        write_rank_info(run_dir, accelerator.process_index, accelerator.num_processes, device, len(train_samples), rank_peak)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and run_dir is not None:
        peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
        run_info = {
            "seconds": time.monotonic() - started,
            "peak_gpu_bytes": peak,
            "checkpoint": str(saved) if saved else None,
            "best_checkpoint": str(best_saved) if best_saved else None,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "reload_checks": reload_checks,
            "infer_outputs": infer_outputs,
        }
        (run_dir / "run.json").write_text(json.dumps(run_info, indent=2))
        report = config.path.parent.parent / "reports" / "overfit_10.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# PersonaPlex Training Report\n\n"
            "Status: PASS\n\n"
            f"- Dataset: {config.manifest}\n- Model: {config.model_root}\n- LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}\n"
            f"- Processes (GPUs): {accelerator.num_processes}\n"
            f"- Steps: {max_steps}\n- Final loss: {last_record['loss/total'] if last_record else 'n/a'}\n"
            f"- Best val loss: {best_val_loss if best_val_loss < float('inf') else 'n/a'}\n"
            f"- Peak GPU bytes: {peak}\n- Checkpoint: {saved}\n"
            f"- Best checkpoint: {best_saved}\n"
            "- Inference outputs: run `python -m tools.inference_smoke` with this checkpoint.\n",
            encoding="utf-8",
        )

    accelerator.wait_for_everyone()
    return best_saved or saved


def main() -> int:
    parser = argparse.ArgumentParser(description="PersonaPlex Fine-Tuning with Accelerate and DDP")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON configuration file")
    parser.add_argument("--config-name", type=str, default=None, help="Name of config in configs/ directory (Hydra style)")
    parser.add_argument("--smoke", action="store_true", help="Run 1-step smoke test with verification")
    parser.add_argument("--qlora", action="store_true", default=None, help="Enable 4-bit QLoRA")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false", help="Disable QLoRA")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--reset-scheduler", action="store_true", help="Restart the LR schedule from config values when resuming (keeps weights and data position)")

    args, unknown = parser.parse_known_args()
    overrides = [arg for arg in unknown if "=" in arg]

    config_path = args.config
    if config_path is None:
        if args.config_name:
            config_path = Path("configs") / f"{args.config_name}.yaml"
        else:
            config_path = Path("configs/config.yaml")

    config = load_config(config_path, overrides=overrides)
    if args.qlora is not None:
        config = config.replace(qlora=args.qlora)

    run(
        config,
        smoke=args.smoke,
        resume_from=args.resume_from,
        reset_scheduler=args.reset_scheduler,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
