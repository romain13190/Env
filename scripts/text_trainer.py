#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time

from dataclasses import dataclass
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TrainerCallback
from typing import Any, Dict, List, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.alfworld_eval_ids import alfworld_eval_task_ids
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from core.config.config_handler import create_reward_funcs_file


ENV_SFT_DATA_PATH = "/workspace/data/alfworld_auto_task_0_2500.jsonl"

@dataclass(frozen=True)
class EnvSFTConfig:
    # Determinism mode:
    # - True: deterministic schedule + (by default) restrict training to validator eval task_ids
    # - False: non-deterministic schedule (shorter default epochs, no forced eval-id restriction)
    deterministic: bool = True

    # Training schedule
    max_epochs: float = 0.0
    deterministic_max_epochs: float = 30.0
    nondeterministic_max_epochs: float = 2.0

    # Hard-mining schedule (only used when deterministic=True)
    hard_mining_start_epoch: float = 10.0  # 0 disables hard mining
    hard_mining_every_epochs: float = 2.0  # 0 mines once, >0 mines periodically

    # Data / packing controls
    max_seq_len: int = 4096
    max_assistant_responses: int = 10
    min_prev_assistant_tokens: int = 200

    # Optim / LR schedule
    learning_rate: float = 8e-5
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.05  # LR floor ratio vs base LR

    def __post_init__(self):
        if self.max_seq_len <= 0:
            raise ValueError("EnvSFTConfig.max_seq_len must be > 0")
        if self.max_assistant_responses <= 0:
            raise ValueError("EnvSFTConfig.max_assistant_responses must be > 0")
        if self.min_prev_assistant_tokens < 0:
            raise ValueError("EnvSFTConfig.min_prev_assistant_tokens must be >= 0")
        if self.learning_rate <= 0:
            raise ValueError("EnvSFTConfig.learning_rate must be > 0")
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError("EnvSFTConfig.warmup_ratio must be in [0, 1]")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("EnvSFTConfig.min_lr_ratio must be in [0, 1]")
        if (self.lr_scheduler or "").strip().lower() not in {"cosine"}:
            raise ValueError("EnvSFTConfig.lr_scheduler must be one of: cosine")


# EnvTask SFT training defaults (edit here; no CLI knobs on purpose)
ENV_SFT_CFG = EnvSFTConfig()


def parse_thresholds(threshold_str: str) -> List[tuple[float, int]]:
    pairs: List[tuple[float, int]] = []
    for part in (threshold_str or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        bound_s, mult_s = part.split(":", 1)
        try:
            bound = float(bound_s)
            mult = int(float(mult_s))
            if bound > 0 and mult > 0:
                pairs.append((bound, mult))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        pairs = [(0.3, 5), (0.6, 3), (0.8, 2), (1.0, 1)]
    return pairs


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def copy_dataset_to_axolotl_directories(dataset_path):
    dataset_filename = os.path.basename(dataset_path)
    data_path, root_path = train_paths.get_axolotl_dataset_paths(dataset_filename)
    shutil.copy(dataset_path, data_path)
    shutil.copy(dataset_path, root_path)

    return data_path


def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, expected_repo_name=None, log_wandb=True):
    """Create the axolotl config file with appropriate settings."""

    print(f"Dataset type: {dataset_type}", flush=True)
    config_path = train_paths.get_axolotl_base_config_path(dataset_type)
    print(f"Config path: {config_path}", flush=True)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    model_path = str(train_paths.get_text_base_model_path(model))
    config["base_model"] = model_path
    config["mlflow_experiment_name"] = dataset
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = str(output_dir)

    if log_wandb:
        config["wandb_runid"] = f"{task_id}_{expected_repo_name}"
        config["wandb_name"] = f"{task_id}_{expected_repo_name}"
        config["wandb_mode"] = "offline"
        os.makedirs(train_cst.WANDB_LOGS_DIR, exist_ok=True)
    else:
        for key in list(config.keys()):
            if key.startswith("wandb"):
                config.pop(key)

    config = update_flash_attention(config, model)

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions],
            task_id,
            destination_dir=train_cst.AXOLOTL_DIRECTORIES["src"],
        )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"

            if "path" in ds:
                ds["path"] = train_cst.AXOLOTL_DIRECTORIES["data"]

            ds["data_files"] = [os.path.basename(dataset)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}
    if tokenizer.bos_token_id is None:
        if tokenizer.eos_token is not None:
            config["special_tokens"] = {"bos_token": tokenizer.eos_token}
        else:
            config["special_tokens"] = {"bos_token": ""}

    config_path = os.path.join(train_cst.AXOLOTL_DIRECTORIES["configs"], f"{task_id}.yml")
    save_config(config, config_path)
    return config_path


def run_training(config_path):
    print(f"Starting training with config: {config_path}", flush=True)
    """Run the training process using the specified config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    training_env = os.environ.copy()
    training_env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    training_env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    training_command = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(training_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


def _load_envtask_dataset(dataset_path: str, allowed_task_ids: Optional[set[int]] = None):
    print(f"[EnvTask-SFT] Loading dataset from {dataset_path}", flush=True)
    ds = load_dataset("json", data_files=dataset_path, split="train")
    if not allowed_task_ids:
        return ds

    allowed = set(int(x) for x in allowed_task_ids)

    def _keep(ex: Dict[str, Any]) -> bool:
        tid = ex.get("id")
        if tid is None:
            tid = (ex.get("request") or {}).get("task_id")
        try:
            return int(tid) in allowed
        except Exception:
            return False

    before = len(ds)
    ds = ds.filter(_keep)
    print(f"[EnvTask-SFT] Filtered dataset to {len(ds)}/{before} rows (eval-only)", flush=True)
    return ds


def _make_env_sft_training_args(
    output_dir: str,
    num_train_epochs: float,
    learning_rate: float,
    logging_steps: int,
    *,
    bf16: bool,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=float(num_train_epochs),
        learning_rate=float(learning_rate),
        logging_steps=int(logging_steps),
        logging_first_step=True,
        disable_tqdm=True,
        save_strategy="no",
        gradient_checkpointing=True,
        fp16=False,
        bf16=bool(bf16),
        report_to=[],
        remove_unused_columns=False,
    )


def _estimate_total_steps(dataset_len: int, args: TrainingArguments, num_epochs: float) -> int:
    if int(dataset_len) <= 0:
        return 1
    world_size = int(getattr(args, "world_size", 1) or 1)
    per_device = int(getattr(args, "per_device_train_batch_size", 1) or 1)
    grad_acc = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    steps_per_epoch = math.ceil((float(dataset_len) / max(1, per_device * max(1, world_size))) / max(1, grad_acc))
    return max(1, int(math.ceil(float(num_epochs) * float(steps_per_epoch))))


def _build_cosine_min_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int, float]:
    """
    Build a cosine LR schedule with:
    - linear warmup for `warmup_ratio` of total steps
    - a floor at `min_lr_ratio` of the base LR

    Important: `total_steps` is expected to be computed ONCE up-front (e.g. from the *raw* dataset size and
    target epochs) and then kept fixed, even if the training dataset changes size later (e.g. due to
    hard-mining oversampling). This prevents changing the LR curve mid-run.
    """
    total_steps = max(1, int(total_steps))
    warmup_ratio = max(0.0, float(warmup_ratio))
    warmup_steps = max(0, int(total_steps * warmup_ratio))
    min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))

    def cosine_with_min_lr_ratio(step: int) -> float:
        """
        Multiplicative factor for the optimizer base LR.

        - Warmup: 0 → 1 linearly over `warmup_steps`
        - Cosine decay: 1 → `min_lr_ratio` over the remaining steps
        - After `total_steps`: stay at `min_lr_ratio` (never decays to 0)
        """
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if step >= total_steps:
            return min_lr_ratio
        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_with_min_lr_ratio), warmup_steps, min_lr_ratio


def _format_messages_to_text(tok, messages, add_generation_prompt: bool = False) -> str:
    # Prefer the model's chat template when available; fall back to a simple serialization otherwise.
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass
    parts: List[str] = []
    for m in messages or []:
        role = (m.get("role") or "").strip().upper()
        content = (m.get("content") or "").strip()
        if role and content:
            parts.append(f"{role}:\n{content}")
    if add_generation_prompt:
        parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def _extract_messages(ex: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    # Supported formats:
    # 1) {"instruction": ..., "output": ...}
    # 2) {"response": {"extra": {"conversation": [...]}}} (alfworld logs)
    # 3) {"conversation": [...]} or {"messages": [...]}
    if isinstance(ex.get("conversation"), list):
        return ex.get("conversation")
    if isinstance(ex.get("messages"), list):
        return ex.get("messages")
    conv = (((ex.get("response") or {}).get("extra") or {}).get("conversation"))
    if isinstance(conv, list):
        return conv
    if "instruction" in ex or "output" in ex:
        instruction = (ex.get("instruction") or "").strip()
        output = (ex.get("output") or "").strip()
        if not instruction or not output:
            return None
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
    return None


def _encode_ids(tok, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> List[int]:
    text = _format_messages_to_text(tok, messages, add_generation_prompt=add_generation_prompt)
    enc = tok(text, add_special_tokens=False, truncation=False, return_tensors=None)
    ids = enc.get("input_ids") or []
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _build_episode_chunks(
    tok,
    ex: Dict[str, Any],
    max_length: int,
    max_assistant_responses: int,
    min_prev_assistant_tokens: int = 200,
) -> List[tuple[list[int], list[bool]]]:
    """
    Episode-level windows:
    - Treat a multi-turn episode as one logical sample.
    - If it exceeds max_length, split into multiple windows without ever cutting an assistant turn.
    - For windows after the first, try to include the previous assistant turn (overlap) when it has
      at least min_prev_assistant_tokens and fits.
    """
    messages = _extract_messages(ex)
    if not isinstance(messages, list) or not messages:
        return []

    assistant_indices = [i for i, m in enumerate(messages) if (m.get("role") or "").strip().lower() == "assistant"]
    if not assistant_indices:
        return []
    assistant_indices = assistant_indices[: max(0, int(max_assistant_responses))]

    # Fast path: single tokenization + offset mapping, compute assistant spans by character boundaries.
    spans: List[tuple[int, int]] = []
    full_ids: List[int] = []
    offsets: Optional[List[tuple[int, int]]] = None
    try:
        full_text = _format_messages_to_text(tok, messages, add_generation_prompt=False)
        enc = tok(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        ids = enc.get("input_ids") or []
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        full_ids = [int(x) for x in ids]
        off = enc.get("offset_mapping")
        if off and isinstance(off[0], list):
            off = off[0]
        offsets = [(int(a), int(b)) for (a, b) in (off or [])]
    except Exception:
        offsets = None

    if offsets and full_ids and len(offsets) == len(full_ids):

        def _char_to_token_start(pos: int) -> Optional[int]:
            # first token whose end > pos
            for i, (_s, e) in enumerate(offsets):
                if e > pos:
                    return i
            return None

        def _char_to_token_end(pos: int, start_from: int) -> int:
            # first token whose start >= pos
            for i in range(start_from, len(offsets)):
                s, _e = offsets[i]
                if s >= pos:
                    return i
            return len(offsets)

        for idx in assistant_indices:
            assistant_content = (messages[idx].get("content") or "").strip()
            if not assistant_content:
                continue
            if idx <= 1 and assistant_content.lower().startswith("ok"):
                continue

            prefix_text = _format_messages_to_text(tok, messages[:idx], add_generation_prompt=True)
            with_text = _format_messages_to_text(tok, messages[: idx + 1], add_generation_prompt=False)
            cs, ce = len(prefix_text), len(with_text)
            ts = _char_to_token_start(cs)
            if ts is None:
                continue
            te = _char_to_token_end(ce, ts)
            if 0 <= ts < te <= len(full_ids):
                spans.append((ts, te))
    else:
        # Fallback: multiple tokenizations (slower but robust across tokenizers)
        full_ids = _encode_ids(tok, messages, add_generation_prompt=False)
        if not full_ids:
            return []
        for idx in assistant_indices:
            assistant_content = (messages[idx].get("content") or "").strip()
            if not assistant_content:
                continue
            if idx <= 1 and assistant_content.lower().startswith("ok"):
                continue
            before_ids = _encode_ids(tok, messages[:idx], add_generation_prompt=True)
            with_ids = _encode_ids(tok, messages[: idx + 1], add_generation_prompt=False)
            s, e = len(before_ids), len(with_ids)
            if 0 <= s < e <= len(full_ids):
                spans.append((s, e))

    if not spans:
        return []

    def _mask_for_window(w_start: int, w_end: int) -> List[bool]:
        mask = [False] * (w_end - w_start)
        for s, e in spans:
            if s >= w_start and e <= w_end:
                for t in range(s - w_start, e - w_start):
                    mask[t] = True
        return mask

    if len(full_ids) <= max_length:
        return [(full_ids, _mask_for_window(0, len(full_ids)))]

    chunks: List[tuple[list[int], list[bool]]] = []
    i = 0
    while i < len(spans):
        # Required_start is where this window may begin without cutting assistant turns.
        i0 = i
        if i > 0:
            ps, pe = spans[i - 1]
            if (pe - ps) >= int(min_prev_assistant_tokens) and (spans[i][1] - ps) <= max_length:
                i0 = i - 1

        required_start = spans[i0][0]

        j = i
        while j + 1 < len(spans) and (spans[j + 1][1] - required_start) <= max_length:
            j += 1

        w_end = spans[j][1]
        w_start = max(0, w_end - max_length)
        if w_start > required_start:
            w_start = required_start

        # Guard (shouldn't happen unless a single assistant span exceeds max_length)
        if (w_end - w_start) > max_length:
            w_start = max(0, w_end - max_length)

        chunks.append((full_ids[w_start:w_end], _mask_for_window(w_start, w_end)))
        i = j + 1
    return chunks


class AssistantOnlyCollator:
    """
    Episode-level collator (multi-turn = one logical sample):
    - tokenizes the full episode
    - labels only assistant turns (everything else masked with -100)
    - if episode exceeds max_length, splits into multiple windows without ever cutting an assistant turn
      and tries to overlap the previous assistant turn (>=min_prev_assistant_tokens) between windows.
    """

    def __init__(self, tok, max_length: int, max_assistant_responses: int = 30, min_prev_assistant_tokens: int = 200):
        self.tok = tok
        self.max_length = int(max_length)
        self.max_assistant_responses = int(max_assistant_responses)
        self.min_prev_assistant_tokens = int(min_prev_assistant_tokens)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = int(self.tok.pad_token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_input_ids: List[List[int]] = []
        all_labels: List[List[int]] = []

        for ex in batch:
            sequences = _build_episode_chunks(
                self.tok,
                ex,
                self.max_length,
                self.max_assistant_responses,
                min_prev_assistant_tokens=self.min_prev_assistant_tokens,
            )
            for ids, mask in sequences:
                labels = [id_val if mask[idx] else -100 for idx, id_val in enumerate(ids)]
                all_input_ids.append(ids)
                all_labels.append(labels)

        if not all_input_ids:
            dummy = torch.tensor([[self.pad_id]], dtype=torch.long)
            return {
                "input_ids": dummy,
                "attention_mask": torch.ones_like(dummy),
                "labels": torch.full_like(dummy, -100),
            }

        max_len = max(len(x) for x in all_input_ids)
        bsz = len(all_input_ids)
        input_ids_tensor = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels_tensor = torch.full((bsz, max_len), -100, dtype=torch.long)

        for i, (ids, lbls) in enumerate(zip(all_input_ids, all_labels)):
            L = len(ids)
            input_ids_tensor[i, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :L] = 1
            labels_tensor[i, :L] = torch.tensor(lbls, dtype=torch.long)

        return {"input_ids": input_ids_tensor, "attention_mask": attention_mask, "labels": labels_tensor}


def _multiplier_for_prob(p: float, thresholds: List[tuple[float, int]]) -> int:
    for bound, mult in thresholds:
        if p < bound:
            return max(1, int(mult))
    return 1


def _hard_mining_run_batch(
    model,
    pad_id: int,
    seq_batch: List[tuple[list[int], list[bool]]],
    owners_batch: List[int],
    seq_min: Dict[int, float],
    device: torch.device,
):
    max_len = max(len(ids) for ids, _ in seq_batch)
    bsz = len(seq_batch)
    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    label_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    for i, (ids, mask) in enumerate(seq_batch):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :L] = 1
        label_mask[i, :L] = torch.tensor(mask, dtype=torch.bool, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = label_mask[:, 1:]
        # Avoid materializing full log_softmax over vocab (huge). Compute token log-prob via:
        # log p(y) = logit_y - logsumexp(logits)
        token_logits = shift_logits.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_norm = torch.logsumexp(shift_logits, dim=-1)
        token_logp = token_logits - log_norm

    for i in range(bsz):
        mask = shift_mask[i]
        if mask.any():
            min_prob = torch.exp(token_logp[i][mask].min()).item()
        else:
            min_prob = 1.0
        owner = owners_batch[i]
        seq_min[owner] = min(seq_min.get(owner, 1.0), min_prob)


def _hard_mine_dataset(
    ds,
    tokenizer,
    model,
    *,
    max_seq_len: int,
    max_assistant_responses: int,
    min_prev_assistant_tokens: int,
    eval_batch_size: int,
    thresholds: List[tuple[float, int]],
):
    print("[EnvTask-SFT] Hard mining: computing token log-probs...", flush=True)
    device = next(model.parameters()).device
    seq_min: Dict[int, float] = {}
    seqs: List[tuple[list[int], list[bool]]] = []
    owners: List[int] = []
    batch_size = max(1, int(eval_batch_size))
    pad_id = int(tokenizer.pad_token_id)

    for idx, ex in enumerate(ds):
        for ids, mask in _build_episode_chunks(
            tokenizer,
            ex,
            int(max_seq_len),
            int(max_assistant_responses),
            min_prev_assistant_tokens=int(min_prev_assistant_tokens),
        ):
            seqs.append((ids, mask))
            owners.append(idx)
            if len(seqs) >= batch_size:
                _hard_mining_run_batch(model, pad_id, seqs, owners, seq_min, device)
                seqs, owners = [], []
    if seqs:
        _hard_mining_run_batch(model, pad_id, seqs, owners, seq_min, device)

    multipliers: List[int] = []
    for i in range(len(ds)):
        prob = seq_min.get(i, 1.0)
        multipliers.append(_multiplier_for_prob(prob, thresholds))

    new_indices: List[int] = []
    for i, m in enumerate(multipliers):
        new_indices.extend([i] * max(1, m))
    mined_ds = ds.select(new_indices)
    print(
        f"[EnvTask-SFT] Hard mining done. Dataset size {len(ds)} → {len(mined_ds)} "
        f"(avg multiplier {sum(multipliers)/len(multipliers):.2f}).",
        flush=True,
    )
    return mined_ds


def run_phased_training(
    *,
    phase1_trainer: Trainer,
    raw_ds,
    model,
    tokenizer,
    output_dir: str,
    base_optimizers,
    max_seq_len: int,
    max_assistant_responses: int,
    min_prev_assistant_tokens: int,
    hard_mining_eval_batch_size: int,
    thresholds: List[tuple[float, int]],
    hard_mining_every_epochs: float,
    wall_clock_start: float,
    hours_to_complete: float,
    enable_hard_mining: bool,
    first_phase_epochs: float,
    target_epochs: float,
    epoch_callback: Optional[TrainerCallback],
    time_budget_callback_cls,
    learning_rate: float,
    bf16: bool,
) -> None:
    """
    Explicit phased training plan:
    - Phase 1: train on raw dataset (already done by caller)
    - Loop: hard-mine → train on mined dataset → update global_epoch
    """
    epochs_done = float(getattr(phase1_trainer.state, "epoch", 0.0) or first_phase_epochs)
    computed_stop = getattr(epoch_callback, "stop_epoch", None) if epoch_callback is not None else None
    target_stop_epoch = min(float(target_epochs), float(computed_stop if computed_stop is not None else target_epochs))

    def _save(tr: Trainer):
        print("[EnvTask-SFT] Saving model...", flush=True)
        tr.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[EnvTask-SFT] Done.", flush=True)

    if (not enable_hard_mining) or (target_stop_epoch <= float(first_phase_epochs)):
        _save(phase1_trainer)
        return

    if epochs_done < float(first_phase_epochs) - 1e-6:
        print("[EnvTask-SFT] Stopped during phase 1 (time budget). Saving.", flush=True)
        _save(phase1_trainer)
        return

    mining_period = float(hard_mining_every_epochs)
    global_epoch = float(first_phase_epochs)
    last_trainer = phase1_trainer

    while global_epoch < float(target_stop_epoch) - 1e-9:
        mining_start = time.time()
        mined_ds = _hard_mine_dataset(
            raw_ds,
            tokenizer,
            model,
            max_seq_len=max_seq_len,
            max_assistant_responses=max_assistant_responses,
            min_prev_assistant_tokens=min_prev_assistant_tokens,
            eval_batch_size=hard_mining_eval_batch_size,
            thresholds=thresholds,
        )
        mining_hours = max(0.0, (time.time() - mining_start) / 3600.0)

        elapsed_hours = max(0.0, (time.time() - wall_clock_start) / 3600.0)
        remaining_hours = max(0.0, hours_to_complete - elapsed_hours - (5.0 / 60.0))  # 5min safety margin
        if hours_to_complete > 0 and remaining_hours <= 0:
            print("[EnvTask-SFT] No remaining wall-clock budget after hard mining; saving.", flush=True)
            _save(last_trainer)
            return

        # Choose how many epochs to run before the next mining step (or until stop).
        phase_cap = float(target_stop_epoch) - global_epoch
        phase_epochs = phase_cap if mining_period <= 0 else min(phase_cap, max(1.0, mining_period))

        args_phase = _make_env_sft_training_args(
            output_dir,
            num_train_epochs=phase_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            bf16=bf16,
        )

        time_callback_phase = (
            time_budget_callback_cls(wall_clock_start, hours_to_complete, safety_minutes=5.0)
            if hours_to_complete > 0
            else None
        )
        callbacks_phase = [cb for cb in [time_callback_phase] if cb]

        phase_trainer = Trainer(
            model=model,
            args=args_phase,
            train_dataset=mined_ds,
            tokenizer=tokenizer,
            data_collator=AssistantOnlyCollator(
                tokenizer,
                max_length=max_seq_len,
                max_assistant_responses=max_assistant_responses,
                min_prev_assistant_tokens=min_prev_assistant_tokens,
            ),
            callbacks=callbacks_phase,
            optimizers=base_optimizers,
        )

        print(
            f"[EnvTask-SFT] Training on mined data ({phase_epochs} epochs, global_epoch={global_epoch:.2f}, mining={mining_hours:.3f}h)...",
            flush=True,
        )
        phase_trainer.train()
        last_trainer = phase_trainer

        phase_done = float(getattr(phase_trainer.state, "epoch", 0.0) or 0.0)
        if phase_done <= 0:
            print("[EnvTask-SFT] Phase produced zero progress (time budget). Saving.", flush=True)
            _save(last_trainer)
            return

        global_epoch += min(float(phase_epochs), phase_done)
        if mining_period <= 0:
            break

    _save(last_trainer)


async def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument(
        "--task-type", required=True, choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask", "EnvTask"], help="Type of task"
    )
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument(
        "--env-eval-only",
        action="store_true",
        help="(EnvTask only) Train only on the fixed 250 AlfWorld task_ids used during validator evaluation.",
    )
    parser.add_argument(
        "--hard-mining-start-epoch",
        type=float,
        default=ENV_SFT_CFG.hard_mining_start_epoch,
        help="(EnvTask deterministic only) Epoch to start hard mining; 0 disables it.",
    )
    parser.add_argument(
        "--hard-mining-every-epochs",
        type=float,
        default=ENV_SFT_CFG.hard_mining_every_epochs,
        help="(EnvTask deterministic only) Recompute log-probs & re-mine every N epochs after start; 0 = mine once.",
    )
    parser.add_argument(
        "--hard-mining-thresholds",
        type=str,
        default="0.3:10,0.6:5,0.8:2,1.0:1",
        help="(EnvTask deterministic only) Comma list of bound:multiplier, e.g. 0.3:10,0.6:5.",
    )
    parser.add_argument(
        "--hard-mining-eval-batch-size",
        type=int,
        default=2,
        help="(EnvTask deterministic only) Batch size for log-prob evaluation during hard mining.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=ENV_SFT_CFG.max_seq_len,
        help="(EnvTask only) Max sequence length for SFT examples. Lower this if you hit OOM.",
    )
    parser.add_argument(
        "--max-assistant-responses",
        type=int,
        default=ENV_SFT_CFG.max_assistant_responses,
        help="(EnvTask only) How many assistant turns to train on per episode. >1 increases memory a lot.",
    )
    parser.add_argument(
        "--max-epochs",
        type=float,
        default=ENV_SFT_CFG.max_epochs,
        help="(EnvTask only) Max total epochs; 0 uses default (deterministic vs non-deterministic default).",
    )
    args = parser.parse_args()

    # Default file format; may be overridden for EnvTask
    file_format = args.file_format

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)

        if args.task_type == TaskType.DPOTASK.value:
            dataset_type = DpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.INSTRUCTTEXTTASK.value:
            dataset_type = InstructTextDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.CHATTASK.value:
            dataset_type = ChatTemplateDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.GRPOTASK.value:
            dataset_type = GrpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.ENVIRONMENTTASK.value:
            dataset_type = EnvironmentDatasetType(**dataset_type_dict)
            file_format = FileFormat.JSON.value
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    # EnvTask → SFT-only path
    if args.task_type == TaskType.ENVIRONMENTTASK.value:
        dataset_path = ENV_SFT_DATA_PATH
        output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
        model_path = train_paths.get_text_base_model_path(args.model)
        # Hardcoded in script (see ENV_SFT_CFG). We intentionally do not use env vars for this.
        deterministic = bool(ENV_SFT_CFG.deterministic)
        thresholds = parse_thresholds(args.hard_mining_thresholds)
        allowed_ids = set(alfworld_eval_task_ids()) if (args.env_eval_only or deterministic) else None
        run_envtask_sft(
            dataset_path,
            model_path,
            output_dir,
            allowed_task_ids=allowed_ids,
            hours_to_complete=args.hours_to_complete,
            deterministic=deterministic,
            hard_mining_start_epoch=args.hard_mining_start_epoch,
            hard_mining_every_epochs=args.hard_mining_every_epochs,
            hard_mining_thresholds=thresholds,
            hard_mining_eval_batch_size=args.hard_mining_eval_batch_size,
            max_seq_len=args.max_seq_len,
            max_assistant_responses=args.max_assistant_responses,
            max_epochs=args.max_epochs,
        )
        patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)
        return

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    if args.task_type == TaskType.DPOTASK.value:
        adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True)
    elif args.task_type == TaskType.GRPOTASK.value:
        adapt_columns_for_grpo_dataset(dataset_path, dataset_type)
    
    dataset_path = copy_dataset_to_axolotl_directories(dataset_path)

    output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    config_path = create_config(
        args.task_id,
        args.model,
        dataset_path,
        dataset_type,
        file_format,
        output_dir,
        args.expected_repo_name,
        log_wandb=True,
    )

    run_training(config_path)

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


def run_envtask_sft(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    max_seq_len: int = ENV_SFT_CFG.max_seq_len,
    max_assistant_responses: int = ENV_SFT_CFG.max_assistant_responses,
    allowed_task_ids: Optional[set[int]] = None,
    hours_to_complete: float = 0.0,
    deterministic: bool = False,
    hard_mining_start_epoch: float = ENV_SFT_CFG.hard_mining_start_epoch,
    hard_mining_every_epochs: float = ENV_SFT_CFG.hard_mining_every_epochs,
    hard_mining_thresholds: Optional[List[tuple[float, int]]] = None,
    hard_mining_eval_batch_size: int = 2,
    max_epochs: float = ENV_SFT_CFG.max_epochs,
):
    wall_clock_start = time.time()
    print(f"[EnvTask-SFT] Loading model from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # Prefer flash attention if available, otherwise fall back to sdpa
    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        model.config.attn_implementation = "sdpa"
    print(f"[EnvTask-SFT] attn_implementation={getattr(model.config, 'attn_implementation', None)} dtype={dtype}", flush=True)

    raw_ds = _load_envtask_dataset(dataset_path, allowed_task_ids)
    min_prev_assistant_tokens = int(ENV_SFT_CFG.min_prev_assistant_tokens)

    class EpochBudgetCallback(TrainerCallback):
        def __init__(self, hours_budget: float):
            self.hours_budget = max(0.0, float(hours_budget))
            self.first_epoch_start: Optional[float] = None
            self.first_epoch_done = False
            self.stop_epoch: Optional[float] = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            if not self.first_epoch_done and self.first_epoch_start is None:
                self.first_epoch_start = time.time()
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if not self.first_epoch_done:
                self.first_epoch_done = True
                epoch_hours = 0.0
                if self.first_epoch_start is not None:
                    epoch_hours = max(0.0, (time.time() - self.first_epoch_start) / 3600.0)
                remaining_hours = max(0.0, self.hours_budget - (20.0 / 60.0) - epoch_hours)
                remaining_hours *= 0.9
                extra_epochs = math.floor(remaining_hours / epoch_hours) if epoch_hours > 0 else 0
                self.stop_epoch = 1 + max(0, extra_epochs)
                print(
                    f"[EnvTask-SFT] First epoch took {epoch_hours:.3f}h. Remaining budget(after margin): {remaining_hours:.3f}h. "
                    f"Will stop after epoch {self.stop_epoch}.",
                    flush=True,
                )

            if self.stop_epoch is not None and state.epoch >= self.stop_epoch:
                control.should_training_stop = True
            return control

    class TimeBudgetCallback(TrainerCallback):
        def __init__(self, wall_clock_start: float, hours_budget: float, safety_minutes: float = 5.0):
            self.wall_clock_start = float(wall_clock_start)
            self.hours_budget = max(0.0, float(hours_budget))
            self.safety = max(0.0, safety_minutes) / 60.0
            self.last_epoch_start: Optional[float] = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.last_epoch_start = time.time()
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            now = time.time()
            elapsed_hours = max(0.0, (now - self.wall_clock_start) / 3600.0)
            epoch_time = 0.0
            if self.last_epoch_start is not None:
                epoch_time = max(0.0, (now - self.last_epoch_start) / 3600.0)
            remaining = self.hours_budget - elapsed_hours - self.safety
            if remaining <= 0 or epoch_time >= remaining:
                control.should_training_stop = True
            return control

    thresholds = sorted(hard_mining_thresholds or [(0.3, 10), (0.6, 5), (0.8, 2), (1.0, 1)], key=lambda x: x[0])
    enable_hard_mining = deterministic and hard_mining_start_epoch > 0

    os.makedirs(output_dir, exist_ok=True)
    default_epochs = float(ENV_SFT_CFG.deterministic_max_epochs) if deterministic else float(ENV_SFT_CFG.nondeterministic_max_epochs)
    target_epochs = float(max_epochs) if float(max_epochs) > 0 else default_epochs
    first_phase_epochs = target_epochs if not enable_hard_mining else max(1.0, min(target_epochs, float(hard_mining_start_epoch)))
    print(
        f"[EnvTask-SFT] Schedule: deterministic={deterministic} enable_hard_mining={enable_hard_mining} "
        f"target_epochs={target_epochs} start_epoch={hard_mining_start_epoch} every_epochs={hard_mining_every_epochs} "
        f"phase1_epochs={first_phase_epochs}",
        flush=True,
    )

    lr = float(ENV_SFT_CFG.learning_rate)
    warmup_ratio = float(ENV_SFT_CFG.warmup_ratio)
    scheduler_name = str(ENV_SFT_CFG.lr_scheduler).strip().lower()
    min_lr_ratio = float(ENV_SFT_CFG.min_lr_ratio)
    bf16 = torch.cuda.is_available()
    args = _make_env_sft_training_args(
        output_dir,
        num_train_epochs=first_phase_epochs,
        learning_rate=lr,
        logging_steps=2,
        bf16=bf16,
    )

    epoch_callback = EpochBudgetCallback(hours_to_complete) if deterministic else None
    time_callback = TimeBudgetCallback(wall_clock_start, hours_to_complete, safety_minutes=5.0) if hours_to_complete > 0 else None
    callbacks = [cb for cb in [epoch_callback, time_callback] if cb]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=raw_ds,
        tokenizer=tokenizer,
        data_collator=AssistantOnlyCollator(
            tokenizer,
            max_length=max_seq_len,
            max_assistant_responses=max_assistant_responses,
            min_prev_assistant_tokens=min_prev_assistant_tokens,
        ),
        callbacks=callbacks,
    )

    # Build a scheduler once and reuse it across phases so it doesn't reset after hard-mining.
    trainer.create_optimizer()
    optimizer = trainer.optimizer
    if optimizer is None:
        raise RuntimeError("[EnvTask-SFT] Failed to create optimizer.")

    if scheduler_name != "cosine":
        raise ValueError(f"[EnvTask-SFT] Unsupported lr_scheduler='{scheduler_name}'. Supported: 'cosine'.")

    # NOTE: We intentionally compute total_steps ONCE from the raw dataset size + target epochs, and then keep it
    # fixed across later phases. Hard-mining oversamples the dataset (len changes), but we don't want the LR curve
    # to shift mid-run.
    total_steps = _estimate_total_steps(int(len(raw_ds)) if raw_ds is not None else 0, args, target_epochs)
    scheduler, warmup_steps, min_lr_ratio = _build_cosine_min_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )
    trainer.lr_scheduler = scheduler
    base_optimizers = (optimizer, scheduler)
    print(
        f"[EnvTask-SFT] LR schedule: {scheduler_name} warmup_steps={warmup_steps} total_steps={total_steps} "
        f"min_lr_ratio={min_lr_ratio} (min_lr={lr * min_lr_ratio:.3e}) "
        f"[total_steps fixed from raw_ds+target_epochs]",
        flush=True,
    )

    print(f"[EnvTask-SFT] Starting training (phase 1, {first_phase_epochs} epochs)...", flush=True)
    trainer.train()

    run_phased_training(
        phase1_trainer=trainer,
        raw_ds=raw_ds,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        base_optimizers=base_optimizers,
        max_seq_len=max_seq_len,
        max_assistant_responses=max_assistant_responses,
        min_prev_assistant_tokens=min_prev_assistant_tokens,
        hard_mining_eval_batch_size=hard_mining_eval_batch_size,
        thresholds=thresholds,
        hard_mining_every_epochs=hard_mining_every_epochs,
        wall_clock_start=wall_clock_start,
        hours_to_complete=hours_to_complete,
        enable_hard_mining=enable_hard_mining,
        first_phase_epochs=first_phase_epochs,
        target_epochs=target_epochs,
        epoch_callback=epoch_callback,
        time_budget_callback_cls=TimeBudgetCallback,
        learning_rate=lr,
        bf16=bf16,
    )
    return


if __name__ == "__main__":
    asyncio.run(main())
