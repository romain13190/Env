import random
from typing import List


def alfworld_eval_task_ids(seed: int = 42, n: int = 250, max_task_id: int = 2500) -> List[int]:
    """
    Reproduce the validator's deterministic AlfWorld evaluation task IDs.

    Validator logic (as of this repo):
      random.seed(42)
      random.sample(range(1, 2500 + 1), 250)
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if max_task_id < 1:
        raise ValueError("max_task_id must be >= 1")
    if n > max_task_id:
        raise ValueError("n must be <= max_task_id when sampling without replacement")
    rng = random.Random(int(seed))
    return rng.sample(range(1, int(max_task_id) + 1), int(n))


