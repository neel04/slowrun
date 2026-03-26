import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import optuna
import wandb

try:
    from optuna.integration.wandb import WeightsAndBiasesCallback
except ImportError:
    from optuna_integration.wandb import WeightsAndBiasesCallback


SEARCH_SPACE = {
    "n_layer": [5, 10, 15],
    "num_epochs": list(range(1, 11)),
    "hira_rank": [0, 16, 32, 64],
    "lr_multiplier": [0.0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
    "total_batch_size": [131072, 262144, 524288],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning wrapper for train.py using Optuna TPE."
    )
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--study-name", type=str, default="slowrun-tpe")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.2)
    parser.add_argument("--input_bin", type=str, default=None)
    parser.add_argument("--input_val_bin", type=str, default=None)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=("muon", "adamw", "search"),
        default="muon",
    )
    parser.add_argument("--device-batch-size", type=int, default=None)
    parser.add_argument("--total-batch-size", type=int, default=None)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--train-script", type=str, default="train.py")
    parser.add_argument("--print-trial-logs", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="slowrun")
    parser.add_argument("--wandb-group", type=str, default="hypertune")
    parser.add_argument("--compile-cache-dir", type=str, default="/tmp/slowrun_torchinductor")
    parser.add_argument("--error-log", type=str, default="/tmp/slowrun_tuner_errors.log")
    return parser.parse_known_args()


def build_command(args, passthrough, trial, result_path):
    optimizer = (
        trial.suggest_categorical("optimizer", ["muon", "adamw"])
        if args.optimizer == "search"
        else args.optimizer
    )
    n_layer = trial.suggest_categorical("n_layer", SEARCH_SPACE["n_layer"])
    num_epochs = trial.suggest_categorical("num_epochs", SEARCH_SPACE["num_epochs"])
    hira_rank = trial.suggest_categorical("hira_rank", SEARCH_SPACE["hira_rank"])
    lr_multiplier = trial.suggest_categorical(
        "lr_multiplier", SEARCH_SPACE["lr_multiplier"]
    )
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.5)
    total_batch_size = (
        args.total_batch_size
        if args.total_batch_size is not None
        else trial.suggest_categorical(
            "total_batch_size", SEARCH_SPACE["total_batch_size"]
        )
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


def main():
    args, passthrough = parse_args()
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
                "wandb_project": args.wandb_project,
                "wandb_group": wandb_group,
                "compile_cache_dir": args.compile_cache_dir,
                "error_log": args.error_log,
                "search_space": SEARCH_SPACE,
            },
            indent=2,
        )
    )
    error_log_path = Path(args.error_log)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    if error_log_path.exists():
        error_log_path.unlink()
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
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
            cmd = build_command(args, passthrough, trial, result_path)
            env = os.environ.copy()
            env["WANDB_MODE"] = "disabled"
            env["WANDB_SILENT"] = "true"
            env.setdefault("PYTHONHASHSEED", str(args.seed))
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", args.compile_cache_dir)
            env.setdefault("TRITON_CACHE_DIR", os.path.join(args.compile_cache_dir, "triton"))
            proc = subprocess.run(
                cmd,
                cwd=Path(__file__).resolve().parent,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = proc.stdout
            if args.print_trial_logs:
                print(output, end="" if output.endswith("\n") else "\n")
            else:
                tail = "\n".join(output.splitlines()[-20:])
                if tail:
                    print(f"Trial {trial.number} tail:\n{tail}")
            if proc.returncode != 0:
                trial.set_user_attr("returncode", proc.returncode)
                trial.set_user_attr("output_tail", "\n".join(output.splitlines()[-50:]))
                with error_log_path.open("a") as f:
                    f.write(f"=== Trial {trial.number} failed ===\n")
                    f.write(f"Return code: {proc.returncode}\n")
                    f.write(f"Command: {' '.join(cmd)}\n")
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
