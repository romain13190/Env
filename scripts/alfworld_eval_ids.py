#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Allow running as a standalone script (so `core` is importable)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from core.alfworld_eval_ids import alfworld_eval_task_ids


def main() -> None:
    p = argparse.ArgumentParser(description="Print the deterministic AlfWorld eval task_ids used by the validator.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--max-task-id", type=int, default=2500)
    p.add_argument("--json", action="store_true", help="Print as JSON array instead of newline-separated.")
    args = p.parse_args()

    ids = alfworld_eval_task_ids(seed=args.seed, n=args.n, max_task_id=args.max_task_id)
    if args.json:
        print(json.dumps(ids))
    else:
        print("\n".join(map(str, ids)))


if __name__ == "__main__":
    main()


