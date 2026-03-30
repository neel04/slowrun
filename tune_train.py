import argparse
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import optuna
import wandb

try:
    from optuna.integration.wandb import WeightsAndBiasesCallback
except ImportError:
    from optuna_integration.wandb import WeightsAndBiasesCallback


SEARCH_SPACE = {
    "n_layer": [10, 15, 20],
    "num_epochs": (1, 12),
    "hira_rank": [0, 16, 32, 64],
    "lr_multiplier": (0.05, 0.5),
    "warmup_ratio": (0.02, 0.5),
    "total_batch_size": [131072, 262144, 524288],
}

SEED_TRIALS = [
    {
        "num_epochs": 6,
        "hira_rank": 32,
        "lr_multiplier": 0.25,
        "warmup_ratio": 0.02,
        "dropout": 0.05,
        "weight_decay": 1.0,
        "total_batch_size": 524288,
    },
]

TPE_STARTUP_TRIALS = 8
PRUNE_PERCENTILE = 50.0
EPOCH_VAL_LOSS_RE = re.compile(
    r"Step\s+\d+\s+\|\s+Epoch\s+(\d+)\s+\|\s+Val BPB:\s+[0-9.eE+-]+\s+\|\s+Val Loss:\s+([0-9.eE+-]+)"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning wrapper for train.py using Optuna TPE."
    )
    parser.add_argument("--n-trials", type=int, default=48)
    parser.add_argument("--study-name", type=str, default="slowrun-tpe")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction",
        "--train_fraction",
        dest="train_fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument("--input_bin", type=str, default=None)
    parser.add_argument("--input_val_bin", type=str, default=None)
    parser.add_argument(
        "--n_layer", "--n_layers", dest="n_layer", type=int, default=None
    )
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("muon", "adamw", "search"),
        default="muon",
    )
    parser.add_argument(
        "--device-batch-size",
        "--device_batch_size",
        dest="device_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--total-batch-size",
        "--total_batch_size",
        dest="total_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-json", "--output_json", dest="output_json", type=str, default=""
    )
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--train-script", type=str, default="train.py")
    parser.add_argument("--print-trial-logs", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="slowrun")
    parser.add_argument(
        "--wandb-group",
        "--wandb_group",
        dest="wandb_group",
        type=str,
        default="hypertune",
    )
    parser.add_argument(
        "--compile-cache-dir", type=str, default="/tmp/slowrun_torchinductor"
    )
    parser.add_argument(
        "--error-log", type=str, default="/tmp/slowrun_tuner_errors.log"
    )
    return parser.parse_known_args()


def fixed_or_suggest[T](fixed_value: T | None, suggest_fn: Callable[[], T]) -> T:
    return fixed_value if fixed_value is not None else suggest_fn()


def get_fixed_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.optimizer != "search":
        overrides["optimizer"] = args.optimizer
    if args.n_layer is not None:
        overrides["n_layer"] = args.n_layer
    if args.total_batch_size is not None:
        overrides["total_batch_size"] = args.total_batch_size
    if args.device_batch_size is not None:
        overrides["device_batch_size"] = args.device_batch_size
    return overrides


def stream_subprocess(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    on_line: Callable[[str], bool] | None = None,
) -> tuple[int, str, bool]:
    lines: list[str] = []
    pruned = False
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            lines.append(line)
            if on_line is not None and on_line(line):
                pruned = True
                os.killpg(proc.pid, signal.SIGTERM)
                break
        if pruned:
            remainder = proc.stdout.read()
            if remainder:
                print(remainder, end="")
                log_file.write(remainder)
                lines.append(remainder)
        try:
            return_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            return_code = proc.wait()
    return return_code, "".join(lines), pruned


def build_command(
    args: argparse.Namespace,
    passthrough: list[str],
    trial: optuna.trial.Trial,
    result_path: Path,
) -> list[str]:
    optimizer = (
        trial.suggest_categorical("optimizer", ["muon", "adamw"])
        if args.optimizer == "search"
        else args.optimizer
    )

    n_layer = fixed_or_suggest(
        args.n_layer,
        lambda: trial.suggest_categorical("n_layer", SEARCH_SPACE["n_layer"]),
    )

    num_epochs = trial.suggest_int("num_epochs", *SEARCH_SPACE["num_epochs"])
    hira_rank = trial.suggest_categorical("hira_rank", SEARCH_SPACE["hira_rank"])
    lr_multiplier = trial.suggest_float(
        "lr_multiplier", *SEARCH_SPACE["lr_multiplier"], log=True
    )

    warmup_ratio = trial.suggest_float("warmup_ratio", *SEARCH_SPACE["warmup_ratio"])

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 4.0)

    total_batch_size = fixed_or_suggest(
        args.total_batch_size,
        lambda: trial.suggest_categorical(
            "total_batch_size", SEARCH_SPACE["total_batch_size"]
        ),
    )

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        args.train_script,
        "--n_layer",
        str(n_layer),
        "--optimizer",
        optimizer,
        "--hira-rank",
        str(hira_rank),
        "--dropout",
        str(dropout),
        "--lr_multiplier",
        str(lr_multiplier),
        "--warmup-ratio",
        str(warmup_ratio),
        "--weight-decay",
        str(weight_decay),
        "--num-epochs",
        str(num_epochs),
        "--patience",
        str(args.patience),
        "--train-fraction",
        str(args.train_fraction),
        "--total-batch-size",
        str(total_batch_size),
        "--save-result",
        str(result_path),
    ]
    if args.input_bin:
        cmd.extend(["--input_bin", args.input_bin])
    if args.input_val_bin:
        cmd.extend(["--input_val_bin", args.input_val_bin])
    if args.device_batch_size is not None:
        cmd.extend(["--device-batch-size", str(args.device_batch_size)])
    cmd.extend(passthrough)
    return cmd


def main() -> None:
    args, passthrough = parse_args()
    fixed_overrides = get_fixed_overrides(args)
    storage = args.storage or f"sqlite:///{Path('/tmp') / f'{args.study_name}.db'}"

    wandb_group = (
        args.wandb_group
        if args.wandb_group
        else f"optuna-{args.study_name}-{time.strftime('%Y%m%d_%H%M%S')}"
    )

    print(
        json.dumps(
            {
                "train_fraction": args.train_fraction,
                "seed": args.seed,
                "optimizer": args.optimizer,
                "storage": storage,
                "wandb_project": args.wandb_project,
                "wandb_group": wandb_group,
                "compile_cache_dir": args.compile_cache_dir,
                "error_log": args.error_log,
                "tpe_startup_trials": TPE_STARTUP_TRIALS,
                "prune_percentile": PRUNE_PERCENTILE,
                "search_space": SEARCH_SPACE,
            },
            indent=2,
        )
    )
    print("--- Fixed Overrides ---")
    print(json.dumps(fixed_overrides, indent=2) if fixed_overrides else "{}")
    if passthrough:
        print("--- Passthrough Args ---")
        print(json.dumps(passthrough, indent=2))

    error_log_path = Path(args.error_log)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    if error_log_path.exists():
        error_log_path.unlink()
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        n_startup_trials=TPE_STARTUP_TRIALS,
    )
    pruner = optuna.pruners.PercentilePruner(
        percentile=PRUNE_PERCENTILE,
        n_startup_trials=TPE_STARTUP_TRIALS,
        n_warmup_steps=1,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    for params in SEED_TRIALS:
        trial_params = {"n_layer": args.n_layer, **params}
        study.enqueue_trial(trial_params, skip_if_exists=True)

    wandb_callback = WeightsAndBiasesCallback(
        metric_name="best_val_loss",
        wandb_kwargs={
            "project": args.wandb_project,
            "group": wandb_group,
            "name": args.study_name,
        },
        as_multirun=True,
    )

    @wandb_callback.track_in_wandb()
    def objective(trial):
        with tempfile.TemporaryDirectory(
            prefix=f"optuna-trial-{trial.number:04d}-"
        ) as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            trial_log_path = Path(tmpdir) / "trial.log"
            cmd = build_command(args, passthrough, trial, result_path)
            env = os.environ.copy()
            env["WANDB_MODE"] = "disabled"
            env["WANDB_SILENT"] = "true"
            env.setdefault("PYTHONHASHSEED", str(args.seed))
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", args.compile_cache_dir)
            env.setdefault(
                "TRITON_CACHE_DIR", os.path.join(args.compile_cache_dir, "triton")
            )

            def on_line(line: str) -> bool:
                match = EPOCH_VAL_LOSS_RE.search(line)
                if match is None:
                    return False
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                trial.report(val_loss, step=epoch)
                wandb.log({"epoch": epoch, "epoch_val_loss": val_loss})
                return trial.should_prune()

            print(f"=== Trial {trial.number} ===")
            print(f"Command: {' '.join(cmd)}")
            print(f"Live log: {trial_log_path}")
            return_code, output, was_pruned = stream_subprocess(
                cmd, env, trial_log_path, on_line=on_line
            )
            if was_pruned:
                trial.set_user_attr("trial_log", str(trial_log_path))
                trial.set_user_attr("pruned", True)
                raise optuna.TrialPruned(
                    f"Pruned at epoch checkpoint; see {trial_log_path}"
                )
            if return_code != 0:
                trial.set_user_attr("returncode", return_code)
                trial.set_user_attr("output_tail", "\n".join(output.splitlines()[-50:]))
                trial.set_user_attr("trial_log", str(trial_log_path))
                with error_log_path.open("a") as f:
                    f.write(f"=== Trial {trial.number} failed ===\n")
                    f.write(f"Return code: {return_code}\n")
                    f.write(f"Command: {' '.join(cmd)}\n")
                    f.write(f"Trial log: {trial_log_path}\n")
                    f.write(output)
                    if not output.endswith("\n"):
                        f.write("\n")
                    f.write("\n")
                return float("inf")
            with result_path.open() as f:
                result = json.load(f)
            best_val_loss = float(result["best_val_loss"])
            trial.set_user_attr(
                "val_loss", float(result.get("val_loss", best_val_loss))
            )
            trial.set_user_attr("command", " ".join(cmd))
            trial.set_user_attr("trial_log", str(trial_log_path))
            wandb.config.update(
                {
                    "study_name": args.study_name,
                    "train_fraction": args.train_fraction,
                },
                allow_val_change=True,
            )
            wandb.log({"best_val_loss": best_val_loss, "trial_number": trial.number})
            return best_val_loss

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout if args.timeout > 0 else None,
        n_jobs=1,
        callbacks=[wandb_callback],
    )
    wandb.finish()

    summary = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "train_fraction": args.train_fraction,
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
