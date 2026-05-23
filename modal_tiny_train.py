"""Run tiny/train.py on one Modal container with 8 H100 GPUs.

Usage:
    uv run modal run modal_tiny_train.py
    uv run modal run modal_tiny_train.py --train-args="--num-epochs 1 --run-name smoke"
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

import modal


APP_NAME = "slowrun-tiny-train"
REMOTE_ROOT = Path("/root/slowrun")
SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("SLOWRUN_REPO_ROOT", Path.cwd())).resolve()
if not (REPO_ROOT / "tiny" / "train.py").exists() and (
    SCRIPT_ROOT / "tiny" / "train.py"
).exists():
    REPO_ROOT = SCRIPT_ROOT
DATA_DIR = REPO_ROOT / "fineweb_data"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Expected local training data at {DATA_DIR}. Run prepare_data.py first."
    )

run_volume = modal.Volume.from_name("slowrun-tiny-runs", create_if_missing=True)
secrets = [modal.Secret.from_dotenv(REPO_ROOT)] if (REPO_ROOT / ".env").exists() else []

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .workdir(str(REMOTE_ROOT))
    .env({
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "NCCL_DEBUG": "WARN",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "SLOWRUN_REPO_ROOT": str(REMOTE_ROOT),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    })
    .add_local_file(REPO_ROOT / "requirements.txt", str(REMOTE_ROOT / "requirements.txt"))
    .add_local_dir(REPO_ROOT / "tiny", str(REMOTE_ROOT / "tiny"))
    .add_local_dir(DATA_DIR, str(REMOTE_ROOT / "fineweb_data"))
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=12 * 60 * 60,
    volumes={str(REMOTE_ROOT / "runs"): run_volume},
    secrets=secrets,
)
def run_tiny_train(train_args: str = "") -> int:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        "tiny/train.py",
        *shlex.split(train_args),
    ]

    env = os.environ.copy()
    if "WANDB_API_KEY" not in env and "WANDB_MODE" not in env:
        env["WANDB_MODE"] = "offline"

    print("Running:", shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REMOTE_ROOT, env=env, check=False)
    run_volume.commit()
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


@app.local_entrypoint()
def main(train_args: str = ""):
    run_tiny_train.remote(train_args)
