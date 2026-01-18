#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
from collections import Counter
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time

import yaml
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import default_data_collator
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
from core.dataset_utils import adapt_columns_for_environment_dataset
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from core.config.config_handler import create_reward_funcs_file


ENV_SFT_DATA_PATH = "/workspace/data/alfworld_auto_task_0_2500.jsonl"

# EnvTask training defaults (edit here instead of passing CLI flags)
ENV_TASK_DEFAULTS = {
    # If max_epochs <= 0, we use deterministic_max_epochs / nondeterministic_max_epochs depending on TRAIN_DETERMINISTIC.
    "max_epochs": 0.0,
    "deterministic_max_epochs": 30.0,
    "nondeterministic_max_epochs": 2.0,
    # Hard-mining schedule (only used when deterministic=True)
    "hard_mining_start_epoch": 2.0,   # 0 disables hard mining
    "hard_mining_every_epochs": 3.0,  # 0 mines once, >0 mines periodically
    # Data / packing controls
    "max_seq_len": 4096,
    "max_assistant_responses": 10,
    "min_prev_assistant_tokens": 200,
}


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
        pairs = [(0.3, 10), (0.6, 5), (0.8, 2), (1.0, 1)]
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
        default=ENV_TASK_DEFAULTS["hard_mining_start_epoch"],
        help="(EnvTask deterministic only) Epoch to start hard mining; 0 disables it.",
    )
    parser.add_argument(
        "--hard-mining-every-epochs",
        type=float,
        default=ENV_TASK_DEFAULTS["hard_mining_every_epochs"],
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
        default=ENV_TASK_DEFAULTS["max_seq_len"],
        help="(EnvTask only) Max sequence length for SFT examples. Lower this if you hit OOM.",
    )
    parser.add_argument(
        "--max-assistant-responses",
        type=int,
        default=ENV_TASK_DEFAULTS["max_assistant_responses"],
        help="(EnvTask only) How many assistant turns to train on per episode. >1 increases memory a lot.",
    )
    parser.add_argument(
        "--max-epochs",
        type=float,
        default=ENV_TASK_DEFAULTS["max_epochs"],
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
        deterministic = (os.getenv("TRAIN_DETERMINISTIC") or "True").lower() in {"1", "true", "yes", "on"}
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
    max_seq_len: int = ENV_TASK_DEFAULTS["max_seq_len"],
    max_assistant_responses: int = ENV_TASK_DEFAULTS["max_assistant_responses"],
    allowed_task_ids: Optional[set[int]] = None,
    hours_to_complete: float = 0.0,
    deterministic: bool = False,
    hard_mining_start_epoch: float = ENV_TASK_DEFAULTS["hard_mining_start_epoch"],
    hard_mining_every_epochs: float = ENV_TASK_DEFAULTS["hard_mining_every_epochs"],
    hard_mining_thresholds: Optional[List[tuple[float, int]]] = None,
    hard_mining_eval_batch_size: int = 2,
    max_epochs: float = ENV_TASK_DEFAULTS["max_epochs"],
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

    print(f"[EnvTask-SFT] Loading dataset from {dataset_path}", flush=True)
    raw_ds = load_dataset("json", data_files=dataset_path, split="train")
    if allowed_task_ids:
        allowed = set(int(x) for x in allowed_task_ids)

        def _keep(ex: Dict[str, Any]) -> bool:
            tid = ex.get("id")
            if tid is None:
                tid = (ex.get("request") or {}).get("task_id")
            try:
                return int(tid) in allowed
            except Exception:
                return False

        before = len(raw_ds)
        raw_ds = raw_ds.filter(_keep)
        print(f"[EnvTask-SFT] Filtered dataset to {len(raw_ds)}/{before} rows (eval-only)", flush=True)

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

    class StdoutLogCallback(TrainerCallback):
        def __init__(self, prefix: str = "[EnvTask-SFT][train]"):
            self.prefix = prefix

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            # Keep it compact + stable
            keys = ("loss", "grad_norm", "learning_rate", "epoch", "step")
            payload = {k: logs[k] for k in keys if k in logs}
            if payload:
                print(f"{self.prefix} {payload}", flush=True)
            return control

    def _format_messages_to_text(tok, messages, add_generation_prompt: bool = False) -> str:
        # Prefer the model's chat template when available; fall back to a simple serialization otherwise.
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            except Exception:
                pass
        parts = []
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
        messages = _extract_messages(ex)  # type: ignore[arg-type]
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
                for i, (_s, e) in enumerate(offsets):  # type: ignore[arg-type]
                    if e > pos:
                        return i
                return None

            def _char_to_token_end(pos: int, start_from: int) -> int:
                # first token whose start >= pos
                for i in range(start_from, len(offsets)):  # type: ignore[arg-type]
                    s, _e = offsets[i]
                    if s >= pos:
                        return i
                return len(offsets)  # type: ignore[arg-type]

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
          and tries to overlap the previous assistant turn (>=200 tokens) between windows.
        """

        def __init__(self, tok, max_length: int, max_assistant_responses: int = 30):
            self.tok = tok
            self.max_length = int(max_length)
            self.max_assistant_responses = int(max_assistant_responses)
            if self.tok.pad_token_id is None:
                self.tok.pad_token = self.tok.eos_token
            self.pad_id = int(self.tok.pad_token_id)

        def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            all_input_ids: List[List[int]] = []
            all_labels: List[List[int]] = []

            for ex in batch:
                sequences = _build_episode_chunks(self.tok, ex, self.max_length, self.max_assistant_responses)
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

    thresholds = sorted(hard_mining_thresholds or [(0.3, 10), (0.6, 5), (0.8, 2), (1.0, 1)], key=lambda x: x[0])
    enable_hard_mining = deterministic and hard_mining_start_epoch > 0

    def _multiplier_for_prob(p: float) -> int:
        for bound, mult in thresholds:
            if p < bound:
                return max(1, int(mult))
        return 1

    def _hard_mine_dataset(ds) -> Any:
        print("[EnvTask-SFT] Hard mining: computing token log-probs...", flush=True)
        device = next(model.parameters()).device
        seq_min: Dict[int, float] = {}
        seqs: List[tuple[list[int], list[bool]]] = []
        owners: List[int] = []
        batch_size = max(1, int(hard_mining_eval_batch_size))
        t0 = time.time()
        scored = 0

        for idx, ex in enumerate(ds):
            for ids, mask in _build_episode_chunks(tokenizer, ex, max_seq_len, max_assistant_responses=max_assistant_responses):
                seqs.append((ids, mask))
                owners.append(idx)
                if len(seqs) >= batch_size:
                    _run_batch(seqs, owners, seq_min, device)
                    seqs, owners = [], []
                    scored += batch_size
                    if scored % max(50, batch_size) == 0:
                        print(f"[EnvTask-SFT] Hard mining progress: scored≈{scored} sequences in {time.time() - t0:.1f}s", flush=True)
        if seqs:
            _run_batch(seqs, owners, seq_min, device)

        multipliers: List[int] = []
        probs: List[float] = []
        for i in range(len(ds)):
            prob = seq_min.get(i, 1.0)
            probs.append(float(prob))
            multipliers.append(_multiplier_for_prob(prob))

        new_indices: List[int] = []
        for i, m in enumerate(multipliers):
            new_indices.extend([i] * max(1, m))
        mined_ds = ds.select(new_indices)
        hist = Counter(multipliers)
        hist_s = ", ".join(f"x{m}:{c}" for m, c in sorted(hist.items(), reverse=True))
        probs_sorted = sorted(probs)
        p50 = probs_sorted[len(probs_sorted) // 2] if probs_sorted else 0.0
        p10 = probs_sorted[max(0, int(0.10 * (len(probs_sorted) - 1)))] if probs_sorted else 0.0
        p90 = probs_sorted[min(len(probs_sorted) - 1, int(0.90 * (len(probs_sorted) - 1)))] if probs_sorted else 0.0
        print(
            f"[EnvTask-SFT] Hard mining done. Dataset size {len(ds)} → {len(mined_ds)} "
            f"(avg multiplier {sum(multipliers)/len(multipliers):.2f}).",
            flush=True,
        )
        print(
            f"[EnvTask-SFT] Hard mining stats: min_prob={min(probs):.3e} p10={p10:.3e} p50={p50:.3e} p90={p90:.3e} | multipliers: {hist_s}",
            flush=True,
        )
        return mined_ds

    def _run_batch(
        seq_batch: List[tuple[list[int], list[bool]]],
        owners_batch: List[int],
        seq_min: Dict[int, float],
        device: torch.device,
    ):
        max_len = max(len(ids) for ids, _ in seq_batch)
        bsz = len(seq_batch)
        pad_id = int(tokenizer.pad_token_id)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
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

    os.makedirs(output_dir, exist_ok=True)
    default_epochs = (
        float(ENV_TASK_DEFAULTS["deterministic_max_epochs"])
        if deterministic
        else float(ENV_TASK_DEFAULTS["nondeterministic_max_epochs"])
    )
    target_epochs = float(max_epochs) if float(max_epochs) > 0 else default_epochs
    first_phase_epochs = target_epochs if not enable_hard_mining else max(1.0, min(target_epochs, float(hard_mining_start_epoch)))
    print(
        f"[EnvTask-SFT] Schedule: deterministic={deterministic} enable_hard_mining={enable_hard_mining} "
        f"target_epochs={target_epochs} start_epoch={hard_mining_start_epoch} every_epochs={hard_mining_every_epochs} "
        f"phase1_epochs={first_phase_epochs}",
        flush=True,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=first_phase_epochs,
        learning_rate=4e-5,
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=True,
        save_strategy="no",
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )

    epoch_callback = EpochBudgetCallback(hours_to_complete) if deterministic else None
    time_callback = TimeBudgetCallback(wall_clock_start, hours_to_complete, safety_minutes=5.0) if deterministic else None
    callbacks = [cb for cb in [epoch_callback, time_callback, StdoutLogCallback()] if cb]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=raw_ds,
        tokenizer=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer, max_length=max_seq_len, max_assistant_responses=max_assistant_responses),
        callbacks=callbacks,
    )

    print(f"[EnvTask-SFT] Starting training (phase 1, {first_phase_epochs} epochs)...", flush=True)
    trainer.train()

    train_time_hours = max(0.0, (time.time() - wall_clock_start) / 3600.0)
    epochs_done = float(getattr(trainer.state, "epoch", 0.0) or first_phase_epochs)
    epoch_time_est = train_time_hours / max(epochs_done, 1e-6)
    computed_stop = epoch_callback.stop_epoch if epoch_callback and epoch_callback.stop_epoch is not None else target_epochs
    target_stop_epoch = min(float(target_epochs), float(computed_stop))
    if not enable_hard_mining or target_stop_epoch <= first_phase_epochs:
        print("[EnvTask-SFT] Saving model...", flush=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[EnvTask-SFT] Done.", flush=True)
        return

    if epochs_done < float(first_phase_epochs) - 1e-6:
        print("[EnvTask-SFT] Stopped during phase 1 (time budget). Saving.", flush=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[EnvTask-SFT] Done.", flush=True)
        return

    mining_period = float(hard_mining_every_epochs)
    global_epoch = float(first_phase_epochs)
    last_trainer = trainer

    while global_epoch < float(target_stop_epoch) - 1e-9:
        mining_start = time.time()
        mined_ds = _hard_mine_dataset(raw_ds)
        mining_hours = max(0.0, (time.time() - mining_start) / 3600.0)

        elapsed_hours = max(0.0, (time.time() - wall_clock_start) / 3600.0)
        remaining_hours = max(0.0, hours_to_complete - elapsed_hours - (5.0 / 60.0))  # 5min safety margin
        if deterministic and remaining_hours <= 0:
            print("[EnvTask-SFT] No remaining wall-clock budget after hard mining; saving.", flush=True)
            last_trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("[EnvTask-SFT] Done.", flush=True)
            return

        # Choose how many epochs to run before the next mining step (or until stop).
        phase_cap = float(target_stop_epoch) - global_epoch
        phase_epochs = phase_cap if mining_period <= 0 else min(phase_cap, max(1.0, mining_period))

        args_phase = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=phase_epochs,
            learning_rate=4e-5,
            logging_steps=10,
            logging_first_step=True,
            disable_tqdm=True,
            save_strategy="no",
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            report_to=[],
            remove_unused_columns=False,
        )

        time_callback_phase = TimeBudgetCallback(wall_clock_start, hours_to_complete, safety_minutes=5.0) if deterministic else None
        callbacks_phase = [cb for cb in [time_callback_phase, StdoutLogCallback(prefix="[EnvTask-SFT][train|mined]")] if cb]

        phase_trainer = Trainer(
            model=model,
            args=args_phase,
            train_dataset=mined_ds,
            tokenizer=tokenizer,
            data_collator=AssistantOnlyCollator(
                tokenizer, max_length=max_seq_len, max_assistant_responses=max_assistant_responses
            ),
            callbacks=callbacks_phase,
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
            last_trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("[EnvTask-SFT] Done.", flush=True)
            return
        global_epoch += min(float(phase_epochs), phase_done)

        if mining_period <= 0:
            break

    print("[EnvTask-SFT] Saving model...", flush=True)
    last_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[EnvTask-SFT] Done.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
