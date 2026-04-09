import argparse
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, TypeVar

import optuna
import wandb

try:
    from optuna.integration.wandb import WeightsAndBiasesCallback
except ImportError:
    from optuna_integration.wandb import WeightsAndBiasesCallback


T = TypeVar("T")

SOTA_ARCHITECTURE = {
    "n_layer": 30,
    "n_head": 14,
    "n_embd": 1792,
}

SOTA_RUN_DEFAULTS = {
    "num_epochs": 12,
    "matrix_lr": 0.04,
    "scalar_lr": 0.1,
    "lr_multiplier": 0.25,
    "weight_decay": 1.3,
    "dropout": 0.1,
    "warmdown_ratio": 0.2,
    "logit_cap": 10.0,
    "logit_avg": 3,
    "logit_avg_mode": "both",
    "swa_last_epochs": 3,
    "stoch_depth": 0.05,
    "total_batch_size": 524288,
}

SEARCH_SPACE = {
    "lr_multiplier": (0.15, 0.40),
    "weight_decay": (0.8, 2.0),
    "dropout": (0.05, 0.20),
    "warmdown_ratio": (0.12, 0.30),
    "stoch_depth": (0.0, 0.10),
    "swa_last_epochs": [2, 3, 4],
}

SEED_TRIALS = [
    {
        "lr_multiplier": 0.25,
        "weight_decay": 1.3,
        "dropout": 0.10,
        "warmdown_ratio": 0.20,
        "stoch_depth": 0.05,
        "swa_last_epochs": 3,
    },
    {
        "lr_multiplier": 0.20,
        "weight_decay": 1.0,
        "dropout": 0.08,
        "warmdown_ratio": 0.18,
        "stoch_depth": 0.05,
        "swa_last_epochs": 3,
    },
    {
        "lr_multiplier": 0.30,
        "weight_decay": 1.6,
        "dropout": 0.12,
        "warmdown_ratio": 0.22,
        "stoch_depth": 0.07,
        "swa_last_epochs": 4,
    },
]

TRAIN_DEFAULT_DEVICE_BATCH_SIZE = 4
TRAIN_SEQUENCE_LEN = 2048
TRAIN_NPROC_PER_NODE = 8
TPE_STARTUP_TRIALS = 12
PRUNE_PERCENTILE = 50.0
PRUNE_WARMUP_EPOCHS = 3
EPOCH_VAL_LOSS_RE = re.compile(
    r"Step\s+\d+\s+\|\s+Epoch\s+(\d+)\s+\|\s+Val BPB:\s+[0-9.eE+-]+\s+\|\s+Val Loss:\s+([0-9.eE+-]+)"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning wrapper for main train.py using Optuna TPE."
    )
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--study-name", type=str, default="slowrun-sota-tpe")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction",
        "--train_fraction",
        dest="train_fraction",
        type=float,
        default=1.0,
        help="Deprecated compatibility flag; ignored by main train.py.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Deprecated compatibility flag; ignored because main train.py uses a fixed optimizer stack.",
    )
    parser.add_argument("--input_bin", type=str, default=None)
    parser.add_argument("--input_val_bin", type=str, default=None)
    parser.add_argument(
        "--n_layer", "--n_layers", dest="n_layer", type=int, default=None
    )
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=-1)
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


def fixed_or_default(fixed_value: T | None, default_value: T) -> T:
    return default_value if fixed_value is None else fixed_value


def resolve_train_script(train_script: str) -> Path:
    path = Path(train_script)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def effective_device_batch_size(args: argparse.Namespace) -> int:
    return fixed_or_default(args.device_batch_size, TRAIN_DEFAULT_DEVICE_BATCH_SIZE)


def effective_n_layer(args: argparse.Namespace) -> int:
    return fixed_or_default(args.n_layer, SOTA_ARCHITECTURE["n_layer"])


def effective_num_epochs(args: argparse.Namespace) -> int:
    return fixed_or_default(args.num_epochs, SOTA_RUN_DEFAULTS["num_epochs"])


def effective_total_batch_size(args: argparse.Namespace) -> int:
    return fixed_or_default(args.total_batch_size, SOTA_RUN_DEFAULTS["total_batch_size"])


def tokens_per_fwdbwd(args: argparse.Namespace) -> int:
    return effective_device_batch_size(args) * TRAIN_SEQUENCE_LEN * TRAIN_NPROC_PER_NODE


def validate_args(args: argparse.Namespace) -> None:
    if args.n_layer is not None and args.n_layer < 4:
        raise ValueError("--n_layer must be at least 4")
    if args.num_epochs is not None and args.num_epochs <= 0:
        raise ValueError("--num-epochs must be >= 1")
    if effective_total_batch_size(args) % tokens_per_fwdbwd(args) != 0:
        raise ValueError(
            "--total_batch_size must be divisible by "
            f"device_batch_size * {TRAIN_SEQUENCE_LEN} * {TRAIN_NPROC_PER_NODE}"
        )
    train_script = resolve_train_script(args.train_script)
    if not train_script.exists():
        raise FileNotFoundError(f"train script not found: {train_script}")


def describe_search_space(args: argparse.Namespace) -> dict[str, object]:
    return {
        "fixed_architecture": {
            "n_layer": effective_n_layer(args),
            "n_head": SOTA_ARCHITECTURE["n_head"],
            "n_embd": SOTA_ARCHITECTURE["n_embd"],
        },
        "fixed_run_config": {
            "num_epochs": effective_num_epochs(args),
            "matrix_lr": SOTA_RUN_DEFAULTS["matrix_lr"],
            "scalar_lr": SOTA_RUN_DEFAULTS["scalar_lr"],
            "total_batch_size": effective_total_batch_size(args),
            "logit_cap": SOTA_RUN_DEFAULTS["logit_cap"],
            "logit_avg": SOTA_RUN_DEFAULTS["logit_avg"],
            "logit_avg_mode": SOTA_RUN_DEFAULTS["logit_avg_mode"],
        },
        "tuned": SEARCH_SPACE,
    }


def enqueue_seed_trials(study: optuna.Study) -> None:
    for params in SEED_TRIALS:
        study.enqueue_trial(params, skip_if_exists=True)


def stream_subprocess(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    *,
    print_logs: bool,
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
            if print_logs:
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
                if print_logs:
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
    n_layer = effective_n_layer(args)
    num_epochs = effective_num_epochs(args)
    total_batch_size = effective_total_batch_size(args)

    lr_multiplier = trial.suggest_float(
        "lr_multiplier", *SEARCH_SPACE["lr_multiplier"], log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", *SEARCH_SPACE["weight_decay"], log=True
    )
    dropout = trial.suggest_float("dropout", *SEARCH_SPACE["dropout"])
    warmdown_ratio = trial.suggest_float(
        "warmdown_ratio", *SEARCH_SPACE["warmdown_ratio"]
    )
    stoch_depth = trial.suggest_float("stoch_depth", *SEARCH_SPACE["stoch_depth"])
    swa_last_epochs = trial.suggest_categorical(
        "swa_last_epochs", SEARCH_SPACE["swa_last_epochs"]
    )
    swa_last_epochs = min(swa_last_epochs, num_epochs)
    logit_avg_dir = result_path.parent / "logit_avg_ckpts"

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={TRAIN_NPROC_PER_NODE}",
        str(resolve_train_script(args.train_script)),
        "--n_layer",
        str(n_layer),
        "--n_head",
        str(SOTA_ARCHITECTURE["n_head"]),
        "--n_embd",
        str(SOTA_ARCHITECTURE["n_embd"]),
        "--matrix-lr",
        str(SOTA_RUN_DEFAULTS["matrix_lr"]),
        "--scalar-lr",
        str(SOTA_RUN_DEFAULTS["scalar_lr"]),
        "--lr_multiplier",
        str(lr_multiplier),
        "--weight-decay",
        str(weight_decay),
        "--dropout",
        str(dropout),
        "--warmdown-ratio",
        str(warmdown_ratio),
        "--stoch-depth",
        str(stoch_depth),
        "--num-epochs",
        str(num_epochs),
        "--patience",
        str(args.patience),
        "--device-batch-size",
        str(effective_device_batch_size(args)),
        "--total-batch-size",
        str(total_batch_size),
        "--logit-cap",
        str(SOTA_RUN_DEFAULTS["logit_cap"]),
        "--logit-avg",
        str(SOTA_RUN_DEFAULTS["logit_avg"]),
        "--logit-avg-dir",
        str(logit_avg_dir),
        "--logit-avg-mode",
        str(SOTA_RUN_DEFAULTS["logit_avg_mode"]),
        "--swa-last-epochs",
        str(swa_last_epochs),
        "--save-result",
        str(result_path),
    ]
    if args.input_bin:
        cmd.extend(["--input_bin", args.input_bin])
    if args.input_val_bin:
        cmd.extend(["--input_val_bin", args.input_val_bin])
    cmd.extend(passthrough)
    return cmd


def main() -> None:
    args, passthrough = parse_args()
    validate_args(args)
    storage = args.storage or f"sqlite:///{Path('/tmp') / f'{args.study_name}.db'}"

    if args.optimizer not in (None, "", "main", "sota", "muon", "adamw", "search"):
        raise ValueError(
            f"Unsupported --optimizer value {args.optimizer!r}. "
            "main train.py uses a fixed optimizer stack, so this flag is ignored."
        )

    wandb_group = (
        args.wandb_group
        if args.wandb_group
        else f"optuna-{args.study_name}-{time.strftime('%Y%m%d_%H%M%S')}"
    )

    print(
        json.dumps(
            {
                "seed": args.seed,
                "storage": storage,
                "wandb_project": args.wandb_project,
                "wandb_group": wandb_group,
                "compile_cache_dir": args.compile_cache_dir,
                "error_log": args.error_log,
                "tpe_startup_trials": TPE_STARTUP_TRIALS,
                "prune_percentile": PRUNE_PERCENTILE,
                "prune_warmup_epochs": min(
                    PRUNE_WARMUP_EPOCHS, max(0, effective_num_epochs(args) - 1)
                ),
                "train_script": str(resolve_train_script(args.train_script)),
                "search_space": describe_search_space(args),
                "deprecated_flags": {
                    "optimizer": args.optimizer,
                    "train_fraction": args.train_fraction,
                },
            },
            indent=2,
        )
    )
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
        n_warmup_steps=min(PRUNE_WARMUP_EPOCHS, max(0, effective_num_epochs(args) - 1)),
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

    enqueue_seed_trials(study)

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
    def objective(trial: optuna.trial.Trial) -> float:
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
                cmd,
                env,
                trial_log_path,
                print_logs=args.print_trial_logs,
                on_line=on_line,
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
                    "sota_architecture": describe_search_space(args)["fixed_architecture"],
                    "sota_run_config": describe_search_space(args)["fixed_run_config"],
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

    if study.trials:
        best_value: float | None = study.best_value
        best_params: dict[str, object] | None = study.best_params
    else:
        best_value = None
        best_params = None

    summary = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_value": best_value,
        "best_params": best_params,
        "fixed_architecture": describe_search_space(args)["fixed_architecture"],
        "fixed_run_config": describe_search_space(args)["fixed_run_config"],
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
