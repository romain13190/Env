#!/bin/bash
set -e
# Resolve the directory of this script to call text_trainer.py with a stable path.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/text_trainer.py" "$@"