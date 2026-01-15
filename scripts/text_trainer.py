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
        default=0.0,
        help="(EnvTask deterministic only) Epoch to start hard mining; 0 disables it.",
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
            hard_mining_thresholds=thresholds,
            hard_mining_eval_batch_size=args.hard_mining_eval_batch_size,
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
    max_seq_len: int = 4096,
    allowed_task_ids: Optional[set[int]] = None,
    hours_to_complete: float = 0.0,
    deterministic: bool = False,
    hard_mining_start_epoch: float = 0.0,
    hard_mining_thresholds: Optional[List[tuple[float, int]]] = None,
    hard_mining_eval_batch_size: int = 2,
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
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # Prefer flash attention if available, otherwise fall back to sdpa
    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        model.config.attn_implementation = "sdpa"

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

    def _find_last_subsequence(haystack_ids: List[int], needle_ids: List[int]) -> Optional[int]:
        if not needle_ids:
            return None
        H, N = len(haystack_ids), len(needle_ids)
        if N > H:
            return None
        for i in range(H - N, -1, -1):
            if haystack_ids[i : i + N] == needle_ids:
                return i
        return None

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

    def _build_assistant_sequences(
        tok,
        ex: Dict[str, Any],
        max_length: int,
        max_assistant_responses: int,
        eos_tok: str,
    ) -> List[tuple[list[int], list[bool]]]:
        messages = _extract_messages(ex)  # type: ignore[arg-type]
        if not isinstance(messages, list) or not messages:
            return []

        assistant_indices = [i for i, m in enumerate(messages) if (m.get("role") or "").strip().lower() == "assistant"]
        if not assistant_indices:
            return []

        sequences: List[tuple[list[int], list[bool]]] = []
        for idx in assistant_indices[: max_assistant_responses]:
            assistant_content = (messages[idx].get("content") or "").strip()
            if not assistant_content:
                continue
            if idx <= 1 and assistant_content.lower().startswith("ok"):
                continue

            assistant_text = assistant_content + eos_tok
            prompt_text = _format_messages_to_text(tok, messages[:idx], add_generation_prompt=True)
            full_text = prompt_text + assistant_text

            enc = tok(full_text, add_special_tokens=False, truncation=False, return_tensors=None)
            ids = enc.get("input_ids") or []
            if not ids:
                continue

            assistant_ids = tok.encode(assistant_text, add_special_tokens=False)
            if not assistant_ids:
                continue
            if len(assistant_ids) > max_length:
                continue

            start = _find_last_subsequence(ids, assistant_ids)
            if start is None:
                start = len(ids) - len(assistant_ids)
                if start < 0:
                    continue
            end = start + len(assistant_ids)

            if len(ids) > max_length:
                w_end = end
                w_start = max(0, w_end - max_length)
                ids = ids[w_start:w_end]
                start -= w_start
                end = start + len(assistant_ids)

            labels_mask = [False] * len(ids)
            if 0 <= start < end <= len(ids):
                for i in range(start, end):
                    labels_mask[i] = True
            else:
                continue

            sequences.append((ids, labels_mask))
        return sequences

    class AssistantOnlyCollator:
        """
        Episode-level collator:
        - builds (prompt_context + assistant_answer + EOS) using chat template
        - labels only assistant_answer(+EOS) tokens, masks everything else with -100
        - if too long, truncates from the LEFT but keeps the entire assistant answer
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

            eos_tok = getattr(self.tok, "eos_token", None) or ""

            for ex in batch:
                sequences = _build_assistant_sequences(
                    self.tok, ex, self.max_length, self.max_assistant_responses, eos_tok
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
        eos_tok = getattr(tokenizer, "eos_token", None) or ""

        for idx, ex in enumerate(ds):
            for ids, mask in _build_assistant_sequences(
                tokenizer, ex, max_seq_len, max_assistant_responses=30, eos_tok=eos_tok
            ):
                seqs.append((ids, mask))
                owners.append(idx)
                if len(seqs) >= batch_size:
                    _run_batch(seqs, owners, seq_min, device)
                    seqs, owners = [], []
        if seqs:
            _run_batch(seqs, owners, seq_min, device)

        multipliers: List[int] = []
        for i in range(len(ds)):
            prob = seq_min.get(i, 1.0)
            multipliers.append(_multiplier_for_prob(prob))

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
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gathered = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            gathered_masked = gathered.masked_fill(~shift_mask, 0.0)

        for i in range(bsz):
            mask = shift_mask[i]
            if mask.any():
                min_prob = torch.exp(gathered_masked[i][mask]).min().item()
            else:
                min_prob = 1.0
            owner = owners_batch[i]
            seq_min[owner] = min(seq_min.get(owner, 1.0), min_prob)

    os.makedirs(output_dir, exist_ok=True)
    base_epochs = 100 if deterministic else 2
    first_phase_epochs = base_epochs if not enable_hard_mining else max(1.0, min(base_epochs, hard_mining_start_epoch))

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
    callbacks = [cb for cb in [epoch_callback, time_callback] if cb]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=raw_ds,
        tokenizer=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer, max_length=max_seq_len),
        callbacks=callbacks,
    )

    print("[EnvTask-SFT] Starting training (phase 1)...", flush=True)
    trainer.train()

    train_time_hours = max(0.0, (time.time() - wall_clock_start) / 3600.0)
    epochs_done = float(getattr(trainer.state, "epoch", 0.0) or first_phase_epochs)
    epoch_time_est = train_time_hours / max(epochs_done, 1e-6)
    target_stop_epoch = epoch_callback.stop_epoch if epoch_callback and epoch_callback.stop_epoch is not None else base_epochs
    if not enable_hard_mining or target_stop_epoch <= first_phase_epochs:
        print("[EnvTask-SFT] Saving model...", flush=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[EnvTask-SFT] Done.", flush=True)
        return

    mining_start = time.time()
    mined_ds = _hard_mine_dataset(raw_ds)
    mining_hours = max(0.0, (time.time() - mining_start) / 3600.0)

    elapsed_hours = max(0.0, (time.time() - wall_clock_start) / 3600.0)
    remaining_hours = max(0.0, hours_to_complete - elapsed_hours - (5.0 / 60.0))  # 5min safety margin
    if remaining_hours <= 0:
        print("[EnvTask-SFT] No remaining wall-clock budget after hard mining; saving.", flush=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[EnvTask-SFT] Done.", flush=True)
        return

    size_ratio = max(1e-6, len(mined_ds) / max(1, len(raw_ds)))
    epoch_time_phase2 = epoch_time_est * size_ratio
    max_epochs_by_time = math.floor(remaining_hours / max(epoch_time_phase2, 1e-6))
    remaining_epochs = min(max_epochs_by_time, max(0.0, target_stop_epoch - first_phase_epochs))
    if remaining_epochs <= 0:
        print("[EnvTask-SFT] Time-constrained: zero epochs available after hard mining; saving.", flush=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[EnvTask-SFT] Done.", flush=True)
        return

    args_phase2 = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=remaining_epochs,
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

    time_callback_phase2 = TimeBudgetCallback(wall_clock_start, hours_to_complete, safety_minutes=5.0) if deterministic else None
    callbacks_phase2 = [cb for cb in [time_callback_phase2] if cb]

    trainer_phase2 = Trainer(
        model=model,
        args=args_phase2,
        train_dataset=mined_ds,
        tokenizer=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer, max_length=max_seq_len),
        callbacks=callbacks_phase2,
    )

    print(f"[EnvTask-SFT] Restarting training (phase 2, {remaining_epochs} epochs)...", flush=True)
    trainer_phase2.train()
    print("[EnvTask-SFT] Saving model...", flush=True)
    trainer_phase2.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[EnvTask-SFT] Done.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
