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


COMMON_SEARCH_SPACE = {
    "n_layer": [10, 15, 20],
    "num_epochs": (1, 12),
    "hira_rank": [0, 16, 32, 64],
}

OPTIMIZER_SEARCH_SPACE = {
    "muon": {
        "lr_multiplier": (0.05, 0.5),
        "warmup_ratio": (0.02, 0.5),
        "dropout": (0.0, 0.5),
        "weight_decay": (0.0, 4.0),
        "weight_decay_log": False,
        "total_batch_size": [131072, 262144, 524288],
    },
    "adamw": {
        "lr_multiplier": (0.002, 0.04),
        "warmup_ratio": (0.02, 0.12),
        "dropout": (0.0, 0.2),
        "weight_decay": (0.003, 0.2),
        "weight_decay_log": True,
        "adam_beta1": [0.8, 0.85, 0.9, 0.95, 0.99],
        "adam_beta2": [0.95, 0.98, 0.99, 0.995, 0.999],
        "total_batch_size": [65536, 131072, 262144, 524288],
    },
}

SEED_TRIALS = {
    "muon": [
        {
            "num_epochs": 6,
            "hira_rank": 32,
            "lr_multiplier": 0.25,
            "warmup_ratio": 0.02,
            "dropout": 0.05,
            "weight_decay": 1.0,
            "total_batch_size": 524288,
        },
    ],
    "adamw": [
        {
            "num_epochs": 6,
            "hira_rank": 0,
            "lr_multiplier": 0.004,
            "warmup_ratio": 0.04,
            "dropout": 0.05,
            "weight_decay": 0.05,
            "adam_beta1": 0.8,
            "adam_beta2": 0.95,
            "total_batch_size": 262144,
        },
        {
            "num_epochs": 6,
            "hira_rank": 16,
            "lr_multiplier": 0.008,
            "warmup_ratio": 0.06,
            "dropout": 0.1,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.99,
            "total_batch_size": 131072,
        },
    ],
}

TRAIN_DEFAULT_DEVICE_BATCH_SIZE = 4
TRAIN_SEQUENCE_LEN = 2048
TRAIN_NPROC_PER_NODE = 8
TPE_STARTUP_TRIALS = 12
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


def validate_args(args: argparse.Namespace) -> None:
    if args.n_layer is not None and args.n_layer < 3:
        raise ValueError("--n_layer must be at least 3")
    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = Path(__file__).resolve().parent / train_script
    if not train_script.exists():
        raise FileNotFoundError(f"train script not found: {train_script}")


def effective_device_batch_size(args: argparse.Namespace) -> int:
    return (
        args.device_batch_size
        if args.device_batch_size is not None
        else TRAIN_DEFAULT_DEVICE_BATCH_SIZE
    )


def tokens_per_fwdbwd(args: argparse.Namespace) -> int:
    return effective_device_batch_size(args) * TRAIN_SEQUENCE_LEN * TRAIN_NPROC_PER_NODE


def valid_total_batch_sizes(args: argparse.Namespace, optimizer: str) -> list[int]:
    tok_per_step = tokens_per_fwdbwd(args)
    return [
        bs
        for bs in OPTIMIZER_SEARCH_SPACE[optimizer]["total_batch_size"]
        if bs % tok_per_step == 0
    ]


def param_name(
    args: argparse.Namespace, optimizer: str, raw_name: str, *, common: bool = False
) -> str:
    if common or args.optimizer != "search":
        return raw_name
    return f"{optimizer}_{raw_name}"


def describe_search_space(args: argparse.Namespace) -> dict[str, object]:
    if args.optimizer == "search":
        search_space = {
            "common": dict(COMMON_SEARCH_SPACE),
            "by_optimizer": {
                optimizer: dict(space)
                for optimizer, space in OPTIMIZER_SEARCH_SPACE.items()
            },
        }
        for optimizer in OPTIMIZER_SEARCH_SPACE:
            search_space["by_optimizer"][optimizer] = {
                **search_space["by_optimizer"][optimizer],
                "valid_total_batch_size": valid_total_batch_sizes(args, optimizer),
            }
        return search_space
    return {
        **COMMON_SEARCH_SPACE,
        **OPTIMIZER_SEARCH_SPACE[args.optimizer],
        "valid_total_batch_size": valid_total_batch_sizes(args, args.optimizer),
    }


def enqueue_seed_trials(
    study: optuna.Study, args: argparse.Namespace
) -> None:
    optimizers = ["muon", "adamw"] if args.optimizer == "search" else [args.optimizer]
    for optimizer in optimizers:
        valid_batch_sizes = set(valid_total_batch_sizes(args, optimizer))
        for params in SEED_TRIALS[optimizer]:
            if (
                args.total_batch_size is None
                and params["total_batch_size"] not in valid_batch_sizes
            ):
                continue
            trial_params = {}
            for key, value in params.items():
                trial_params[param_name(args, optimizer, key)] = value
            if args.optimizer == "search":
                trial_params["optimizer"] = optimizer
            if args.n_layer is not None:
                trial_params["n_layer"] = args.n_layer
            if args.total_batch_size is not None:
                trial_params[param_name(args, optimizer, "total_batch_size")] = (
                    args.total_batch_size
                )
            study.enqueue_trial(trial_params, skip_if_exists=True)


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
    optimizer_space = OPTIMIZER_SEARCH_SPACE[optimizer]

    n_layer = fixed_or_suggest(
        args.n_layer,
        lambda: trial.suggest_categorical("n_layer", COMMON_SEARCH_SPACE["n_layer"]),
    )

    num_epochs = trial.suggest_int("num_epochs", *COMMON_SEARCH_SPACE["num_epochs"])
    hira_rank = trial.suggest_categorical("hira_rank", COMMON_SEARCH_SPACE["hira_rank"])
    lr_multiplier = trial.suggest_float(
        param_name(args, optimizer, "lr_multiplier"),
        *optimizer_space["lr_multiplier"],
        log=True,
    )

    warmup_ratio = trial.suggest_float(
        param_name(args, optimizer, "warmup_ratio"),
        *optimizer_space["warmup_ratio"],
    )

    dropout = trial.suggest_float(
        param_name(args, optimizer, "dropout"), *optimizer_space["dropout"]
    )
    weight_decay = trial.suggest_float(
        param_name(args, optimizer, "weight_decay"),
        *optimizer_space["weight_decay"],
        log=optimizer_space["weight_decay_log"],
    )
    adam_beta1 = None
    adam_beta2 = None
    if optimizer == "adamw":
        adam_beta1 = trial.suggest_categorical(
            param_name(args, optimizer, "adam_beta1"),
            optimizer_space["adam_beta1"],
        )
        adam_beta2 = trial.suggest_categorical(
            param_name(args, optimizer, "adam_beta2"),
            optimizer_space["adam_beta2"],
        )

    total_batch_size = fixed_or_suggest(
        args.total_batch_size,
        lambda: trial.suggest_categorical(
            param_name(args, optimizer, "total_batch_size"),
            valid_total_batch_sizes(args, optimizer),
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
    if adam_beta1 is not None and adam_beta2 is not None:
        cmd.extend(["--adam-beta1", str(adam_beta1), "--adam-beta2", str(adam_beta2)])
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
    validate_args(args)
    fixed_overrides = get_fixed_overrides(args)
    storage = args.storage or f"sqlite:///{Path('/tmp') / f'{args.study_name}.db'}"

    if args.total_batch_size is not None and args.total_batch_size % tokens_per_fwdbwd(args) != 0:
        raise ValueError(
            "--total_batch_size must be divisible by "
            f"device_batch_size * {TRAIN_SEQUENCE_LEN} * {TRAIN_NPROC_PER_NODE}"
        )
    if args.total_batch_size is None:
        optimizers = ["muon", "adamw"] if args.optimizer == "search" else [args.optimizer]
        for optimizer in optimizers:
            if not valid_total_batch_sizes(args, optimizer):
                raise ValueError(
                    f"No valid total_batch_size choices remain for optimizer={optimizer} "
                    f"with device_batch_size={effective_device_batch_size(args)}"
                )

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
                "search_space": describe_search_space(args),
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
        n_warmup_steps=2,
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

    enqueue_seed_trials(study, args)

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
