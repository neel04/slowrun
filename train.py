"""
Train a language model on ~100M tokens with val loss evaluation.
Code is based on Nanochat (https://github.com/karpathy/nanochat), with modifications to support the slowrun setting.

Usage:
    torchrun --standalone --nproc_per_node=8 train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import gc
import importlib
import json
import math
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
import tiktoken
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import wandb

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--num-epochs", type=int, default=11)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run-name", type=str, default=None,
                    help="Run name under runs/ (default: random 6-char string)")
parser.add_argument("--scalar-lr", type=float, default=0.1)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--weight-decay", type=float, default=1.3)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=30)
parser.add_argument(
    "--num-iterations",
    type=int,
    default=3,
    help="Maximum recurrent iterations through the network",
)
parser.add_argument(
    "--iteration-schedule",
    type=str,
    default="constant",
    choices=["constant", "compute-matched"],
    help=(
        "Schedule recurrent iterations over training. "
        "'compute-matched' ramps from --min-iterations through each integer stage "
        "to --num-iterations so average layer-passes match --compute-equivalent-layers."
    ),
)
parser.add_argument(
    "--min-iterations",
    type=int,
    default=1,
    help="Initial recurrent iterations for --iteration-schedule compute-matched",
)
parser.add_argument(
    "--compute-equivalent-layers",
    type=float,
    default=30.0,
    help=(
        "Target average baseline-width-equivalent layer-passes per token for "
        "compute-matched schedule"
    ),
)
parser.add_argument(
    "--compute-reference-n-embd",
    type=float,
    default=1792.0,
    help="Reference model width used by --compute-equivalent-layers",
)
parser.add_argument(
    "--compute-width-power",
    type=float,
    default=2.0,
    help="Width scaling exponent for compute matching; dense matmuls are ~2.0",
)
parser.add_argument("--n_head", type=int, default=14)
parser.add_argument("--n_embd", type=int, default=1792)
parser.add_argument("--lr_multiplier", type=float, default=0.25)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument(
    "--dupe-start-epoch",
    type=int,
    default=None,
    help="Epoch to enable layer duplication (default: disabled)",
)
parser.add_argument(
    "--dupe-layers-start",
    type=int,
    default=15,
    help="First decoder layer to duplicate (inclusive)",
)
parser.add_argument(
    "--dupe-layers-end",
    type=int,
    default=21,
    help="Last decoder layer to duplicate (exclusive)",
)
parser.add_argument(
    "--dupe-loops",
    type=int,
    default=2,
    help="Number of extra replay passes through dupe layers",
)
parser.add_argument(
    "--warmdown-ratio",
    type=float,
    default=None,
    help="Override warmdown ratio (default 0.2)",
)
parser.add_argument(
    "--logit-cap",
    type=float,
    default=10.0,
    help="Logit soft-capping value (0=disabled)",
)
parser.add_argument(
    "--logit-avg",
    type=int,
    default=3,
    help="Number of late checkpoints for logit (probability) averaging (0=disabled)",
)
parser.add_argument(
    "--logit-avg-dir",
    type=str,
    default="logit_avg_ckpts",
    help="Directory to save/load epoch checkpoints for logit averaging",
)
parser.add_argument(
    "--logit-avg-mode",
    type=str,
    default="both",
    choices=["equal", "weighted", "both"],
    help="Weight scheme: equal, linear recency weighted, or compare both",
)
parser.add_argument(
    "--eval-logit-avg",
    action="store_true",
    help="Skip training and only run logit-avg eval on saved checkpoints",
)
parser.add_argument(
    "--swa-last-epochs",
    type=int,
    default=3,
    help="SWA: cosine-cycle LR in last N epochs for checkpoint diversity (0=off)",
)
parser.add_argument(
    "--stoch-depth",
    type=float,
    default=0.05,
    help="Stochastic depth max drop rate (linear schedule, 0=off)",
)
parser.add_argument(
    "--mtp-weight",
    type=float,
    default=0.3,
    help="Multi-token prediction weight (0=off)",
)
parser.add_argument(
    "--iha",
    action="store_true",
    default=True,
    help="Enable Interleaved Head Attention (cross-head Q/K/V mixing)",
)
parser.add_argument(
    "--no-iha", action="store_false", dest="iha", help="Disable IHA cross-head mixing"
)
parser.add_argument(
    "--iha-lr", type=float, default=0.02, help="LR for IHA mixing matrices"
)
parser.add_argument(
    "--no-doc-shuffle",
    action="store_true",
    help="Disable per-epoch document reshuffling (still shuffles batch order)",
)
parser.add_argument(
    "--hira-rank",
    "--param-relax-rank",
    dest="hira_rank",
    type=int,
    default=64,
    help="Rank for per-iteration ABBA parameter relaxation adapters (0=disabled)",
)
parser.add_argument(
    "--debug-timing-steps",
    type=int,
    default=0,
    help="Print per-section timings for the first N training steps",
)
args = parser.parse_args()

# Resolve output path
if args.output_json and not args.save_result:
    args.save_result = args.output_json

# =============================================================================
# Hardwired d12 (GPT-2 small) hyperparameters
# =============================================================================

# RLM
NUM_ITERATIONS = args.num_iterations

# Architecture (defaults = d12 GPT-2 small)
DEPTH = args.n_layer if args.n_layer is not None else 12
N_EMBD = args.n_embd if args.n_embd is not None else 768
N_HEAD = args.n_head if args.n_head is not None else 6
HEAD_DIM = N_EMBD // N_HEAD
if args.compute_reference_n_embd <= 0:
    raise ValueError("--compute-reference-n-embd must be > 0")
COMPUTE_WIDTH_SCALE = (N_EMBD / args.compute_reference_n_embd) ** args.compute_width_power
MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
EVAL_TOKENS = 10_000_000
DATA_DIR = "fineweb_data"
BOS_ID = 50256  # <|endoftext|>
RUNS_DIR = "runs"

# Base optimizer hyperparameters
BASE_MATRIX_LR = args.matrix_lr
BASE_SCALAR_LR = args.scalar_lr
BASE_EMBEDDING_LR = 0.15
BASE_UNEMBEDDING_LR = 0.002

# Apply LR multiplier if provided (scales all LRs uniformly)
_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
MATRIX_LR = BASE_MATRIX_LR * _lr_mult
UNEMBEDDING_LR = BASE_UNEMBEDDING_LR * _lr_mult
EMBEDDING_LR = BASE_EMBEDDING_LR * _lr_mult
SCALAR_LR = BASE_SCALAR_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = args.warmdown_ratio if args.warmdown_ratio is not None else 0.2
FINAL_LR_FRAC = 0.0
WARMDOWN_POWER = 0.5      # sqrt-shaped warmdown (stays ~41% higher at midpoint than linear)
WD_PRE_HOLD_FRAC = 0.40   # hold at base WD for first 40% of training, then decay to LOW by SWA start
WD_SWA_LOW_FACTOR = 0.65  # WD at start of each SWA epoch (LR is high → less regularization)
WD_SWA_HIGH_FACTOR = 1.50 # WD at end of each SWA epoch (LR has decayed → more regularization)
LOGIT_CAP = args.logit_cap
FINAL_EXTRA_EVAL_ITERATIONS = 1

# =============================================================================
# Utilities
# =============================================================================


def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return (
            True,
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    return False, 0, 0, 1


def print0(s="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)


@dataclass(frozen=True)
class IterationScheduleStage:
    start_frac: float
    end_frac: float
    iterations: int


@dataclass(frozen=True)
class IterationSchedule:
    stages: tuple
    avg_iterations: float
    target_effective_layers: float
    width_compute_scale: float


def build_iteration_schedule(
    schedule_name,
    min_iterations,
    max_iterations,
    n_layer,
    target_effective_layers,
    width_compute_scale,
    warmdown_ratio,
):
    if max_iterations < 1:
        raise ValueError("--num-iterations must be >= 1")
    if min_iterations < 1:
        raise ValueError("--min-iterations must be >= 1")
    if min_iterations > max_iterations:
        raise ValueError("--min-iterations must be <= --num-iterations")
    if n_layer <= 0:
        raise ValueError("--n_layer must be > 0")
    if width_compute_scale <= 0:
        raise ValueError("width_compute_scale must be > 0")

    if schedule_name == "constant":
        avg_iterations = float(max_iterations)
        return IterationSchedule(
            stages=(IterationScheduleStage(0.0, 1.0, max_iterations),),
            avg_iterations=avg_iterations,
            target_effective_layers=n_layer * avg_iterations * width_compute_scale,
            width_compute_scale=width_compute_scale,
        )

    target_iterations = target_effective_layers / (n_layer * width_compute_scale)
    if target_iterations < min_iterations or target_iterations > max_iterations:
        raise ValueError(
            "--compute-equivalent-layers requires an average of "
            f"{target_iterations:.3f} iterations for n_layer={n_layer} "
            f"and width_compute_scale={width_compute_scale:.3f}, outside "
            f"[--min-iterations={min_iterations}, --num-iterations={max_iterations}]"
        )

    if min_iterations == max_iterations:
        if abs(target_iterations - max_iterations) > 1e-9:
            raise ValueError(
                "Cannot compute-match with equal min/max iterations unless the "
                "target equals that iteration count"
            )
        return IterationSchedule(
            stages=(IterationScheduleStage(0.0, 1.0, max_iterations),),
            avg_iterations=float(max_iterations),
            target_effective_layers=target_effective_layers,
            width_compute_scale=width_compute_scale,
        )

    warmdown_duration = min(max(warmdown_ratio, 0.0), 1.0)
    extra_iterations = target_iterations - min_iterations
    threshold_count = max_iterations - min_iterations
    min_extra_with_max_before_warmdown = threshold_count * warmdown_duration
    if extra_iterations < min_extra_with_max_before_warmdown - 1e-9:
        warmdown_start_frac = 1.0 - warmdown_duration
        raise ValueError(
            "Compute-matched schedule cannot reach --num-iterations by warmdown "
            f"start ({warmdown_start_frac:.3f}) without exceeding the compute "
            f"target. Minimum average iterations would be "
            f"{min_iterations + min_extra_with_max_before_warmdown:.3f}, "
            f"but target is {target_iterations:.3f}."
        )

    suffix_durations = [warmdown_duration] * threshold_count
    remaining_extra = extra_iterations - min_extra_with_max_before_warmdown
    for i in range(threshold_count):
        capacity = 1.0 - suffix_durations[i]
        add = min(remaining_extra, capacity)
        suffix_durations[i] += add
        remaining_extra -= add
        if remaining_extra <= 1e-9:
            break
    if remaining_extra > 1e-9:
        raise ValueError(
            "Could not build compute-matched iteration schedule; target average "
            f"iterations {target_iterations:.3f} is too high"
        )

    durations = [1.0 - suffix_durations[0]]
    for i in range(1, threshold_count):
        durations.append(suffix_durations[i - 1] - suffix_durations[i])
    durations.append(suffix_durations[-1])

    stages = []
    start_frac = 0.0
    for offset, duration in enumerate(durations):
        if duration <= 1e-9:
            continue
        end_frac = min(1.0, start_frac + duration)
        stages.append(
            IterationScheduleStage(start_frac, end_frac, min_iterations + offset)
        )
        start_frac = end_frac
    if stages:
        stages[-1] = IterationScheduleStage(
            stages[-1].start_frac, 1.0, stages[-1].iterations
        )
    else:
        stages = [IterationScheduleStage(0.0, 1.0, max_iterations)]
    stages = tuple(stages)

    avg_iterations = sum(
        (stage.end_frac - stage.start_frac) * stage.iterations for stage in stages
    )
    return IterationSchedule(
        stages=stages,
        avg_iterations=avg_iterations,
        target_effective_layers=target_effective_layers,
        width_compute_scale=width_compute_scale,
    )


def get_scheduled_iterations(schedule, step, total_steps):
    if total_steps <= 0:
        return schedule.stages[-1].iterations
    frac = min(max(step / total_steps, 0.0), 1.0)
    for stage in schedule.stages:
        if frac < stage.end_frac or stage is schedule.stages[-1]:
            return stage.iterations
    return schedule.stages[-1].iterations


def format_iteration_schedule(schedule):
    return ", ".join(
        f"{stage.start_frac:.3f}-{stage.end_frac:.3f}: {stage.iterations}x"
        for stage in schedule.stages
    )


def iteration_schedule_counts(schedule):
    return tuple(dict.fromkeys(stage.iterations for stage in schedule.stages))


ITERATION_SCHEDULE = build_iteration_schedule(
    args.iteration_schedule,
    args.min_iterations,
    NUM_ITERATIONS,
    DEPTH,
    args.compute_equivalent_layers,
    COMPUTE_WIDTH_SCALE,
    WARMDOWN_RATIO,
)


class DummyWandb:
    def __init__(self):
        self.summary = {}

    def log(self, *a, **kw):
        pass

    def finish(self):
        pass


class TeeStream:
    """Save terminal output to file."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for stream in self.streams: stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams: stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()


def resolve_run_dir(run_name):
    if run_name:
        return run_name, os.path.join(RUNS_DIR, run_name)
    name = time.strftime('%Y%m%d_%H%M%S')
    return name, os.path.join(RUNS_DIR, name)


def load_state_dict_into_model(model, state_dict):
    """Load a state dict into model, handling dtype conversion."""
    for name, p in model.named_parameters():
        if name in state_dict:
            p.data.copy_(state_dict[name].to(p.device, dtype=p.dtype))


# =============================================================================
# Flash Attention (FA3 on Hopper)
# =============================================================================


def _load_fa3():
    if not torch.cuda.is_available():
        return None
    errors = []
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
    except Exception as exc:
        errors.append(f"cuda capability check failed: {exc}")
        return None
    try:
        return importlib.import_module("flash_attn_interface")
    except Exception as exc:
        errors.append(f"import flash_attn_interface failed: {exc}")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from kernels import get_kernel

        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception as exc:
        errors.append(
            f"kernels.get_kernel('varunneal/flash-attention-3') failed: {exc}"
        )
    _load_fa3.last_error = " | ".join(errors)
    return None


_fa3 = _load_fa3()


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention for training (FA3 only). q,k,v: (B, T, H, D)."""
    return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)


flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model
# =============================================================================


@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 50257
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.0
    stoch_depth: float = 0.05
    num_iterations: int = NUM_ITERATIONS
    hira_rank: int = 8
    use_iha: bool = False
    iha_mix_v: bool = True


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value Embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


def new_gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3.0))))


def megatron_uniform_(tensor):
    fan_in, fan_out = tensor.shape
    std = (0.33 / fan_in) ** 0.5
    lim = fan_out**-0.5
    return torch.nn.init.uniform_(tensor, -lim, lim).mul_(std)


class ABBA(nn.Module):
    """Per-iteration ABBA adapter for relaxing shared recurrent parameters."""

    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.scale = 1.0 / rank
        # Khatri-Rao is cheaper while rank^2 is below the dense dimension.
        # At higher ranks, the exact effective weight uses fewer activation FLOPs.
        self.use_dense_weight = rank * rank * (in_dim + out_dim) >= in_dim * out_dim
        self.B_1 = nn.Parameter(torch.empty(in_dim, rank))
        self.A_1 = nn.Parameter(torch.empty(rank, out_dim))
        self.B_2 = nn.Parameter(torch.empty(in_dim, rank))
        self.A_2 = nn.Parameter(torch.empty(rank, out_dim))

    @torch.no_grad()
    def reset_parameters(self):
        megatron_uniform_(self.B_1)
        megatron_uniform_(self.A_1)
        torch.nn.init.zeros_(self.B_2)
        megatron_uniform_(self.A_2)

    def _forward_rank_squared(self, x):
        a_kr = (
            (self.A_1.mT.unsqueeze(-1) * self.A_2.mT.unsqueeze(-2))
            .reshape(self.out_dim, self.rank * self.rank)
            .transpose(0, 1)
        ).to(dtype=x.dtype)
        b_kr = (
            (self.B_1.unsqueeze(-1) * self.B_2.unsqueeze(-2))
            .reshape(self.B_1.size(0), self.rank * self.rank)
            .to(dtype=x.dtype)
        )
        return (x @ b_kr) @ a_kr

    def _forward_dense_weight(self, x):
        # Equivalent to the rank^2 Khatri-Rao expansion:
        # b_kr @ a_kr == (B_1 @ A_1) * (B_2 @ A_2).
        weight = (self.B_1 @ self.A_1) * (self.B_2 @ self.A_2)
        return x @ weight.to(dtype=x.dtype)

    def forward(self, x):
        if self.use_dense_weight:
            out = self._forward_dense_weight(x)
        else:
            out = self._forward_rank_squared(x)
        return self.scale * out

    def compute_matmul_params(self):
        if self.use_dense_weight:
            return self.in_dim * self.out_dim
        return self.rank * self.rank * (self.in_dim + self.out_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Attention gate: per-head gating to enable context-based no-op
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)
        # IHA: cross-head mixing matrices (Interleaved Head Attention)
        # Mixing is fused into projection weights at forward time.
        # Cost: [H,H]@[H,d*C] matmul is negligible vs the [B*T,C]@[C,H*d] projection.
        self.use_iha = config.use_iha
        if self.use_iha:
            self.q_mix = nn.Parameter(torch.zeros(self.n_head, self.n_head))
            self.k_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))
            self.iha_mix_v = config.iha_mix_v
            if self.iha_mix_v:
                self.v_mix = nn.Parameter(torch.zeros(self.n_kv_head, self.n_kv_head))

    def _fuse_mix(self, weight, mix, H):
        """Fuse mixing matrix into projection weight: W_fused[h] = sum_m mix[h,m]*W[m]."""
        d = self.head_dim
        return (mix @ weight.view(H, d, -1).flatten(1)).view_as(weight)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()

        if self.use_iha:
            # Fuse mixing into weights then project — grad flows through mix params
            q = F.linear(x, self._fuse_mix(self.c_q.weight, self.q_mix, self.n_head))
            q = q.view(B, T, self.n_head, self.head_dim)
            k = F.linear(x, self._fuse_mix(self.c_k.weight, self.k_mix, self.n_kv_head))
            k = k.view(B, T, self.n_kv_head, self.head_dim)
            if self.iha_mix_v:
                v = F.linear(
                    x, self._fuse_mix(self.c_v.weight, self.v_mix, self.n_kv_head)
                )
                v = v.view(B, T, self.n_kv_head, self.head_dim)
            else:
                v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        else:
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        # Attention gate: per-head sigmoid gate
        y = y * torch.sigmoid(
            self.attn_gate(x[..., : self.attn_gate_channels])
        ).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx, enable_relax=True):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        if enable_relax and config.hira_rank > 0:
            self.attn_relax = nn.ModuleList([
                ABBA(config.n_embd, config.n_embd, config.hira_rank)
                for _ in range(config.num_iterations)
            ])
            self.mlp_relax = nn.ModuleList([
                ABBA(config.n_embd, config.n_embd, config.hira_rank)
                for _ in range(config.num_iterations)
            ])
        else:
            self.attn_relax = None
            self.mlp_relax = None
        # Stochastic depth: linear schedule from 0 at layer 0 to stoch_depth at last layer
        self.drop_prob = config.stoch_depth * (layer_idx / max(config.n_layer - 1, 1))

    def forward(self, x, ve, cos_sin, window_size, iteration_idx=None):
        x_in = x

        x_norm = norm(x)
        attn_out = self.attn(x_norm, ve, cos_sin, window_size)
        if self.attn_relax is not None and iteration_idx is not None:
            relax_idx = min(iteration_idx, len(self.attn_relax) - 1)
            attn_out = attn_out + new_gelu(self.attn_relax[relax_idx](x_norm))
        x = x + attn_out

        x_norm = norm(x)
        mlp_out = self.mlp(x_norm)
        if self.mlp_relax is not None and iteration_idx is not None:
            relax_idx = min(iteration_idx, len(self.mlp_relax) - 1)
            mlp_out = mlp_out + new_gelu(self.mlp_relax[relax_idx](x_norm))
        x = x + mlp_out

        # Stochastic depth: blend with identity when dropped (compile-friendly, no graph break)
        if self.training and self.drop_prob > 0:
            keep = (torch.rand((), device=x.device) >= self.drop_prob).to(x.dtype)
            x = x_in + keep * (x - x_in)
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({
            str(i): nn.Linear(config.n_embd, kv_dim, bias=False)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })

        # U-Net skip connections: encoder layer i → decoder layer (n_layer - 1 - i)
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=10000)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self._dupe_layers = None  # (start, end) or None

        self.mtp_weight = args.mtp_weight
        if self.mtp_weight > 0:
            self.mtp_proj = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
            self.mtp_block = Block(config, config.n_layer, enable_relax=False)

    def set_dupe_layers(self, start, end, loops=2):
        assert start >= self.encoder_layers, "dupe layers must be decoder-only"
        assert end <= self.config.n_layer
        self._dupe_layers = (start, end)
        self._dupe_loops = loops
        print0(
            f"Dupe layers {start}-{end - 1}: {loops} extra replays ({loops + 1} total passes)"
        )

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        s = 3**0.5 * self.config.n_embd**-0.5
        normal_std = self.config.n_embd**-0.5

        all_blocks = list(self.transformer.h)
        if self.mtp_weight > 0:
            all_blocks.append(self.mtp_block)
            torch.nn.init.uniform_(self.mtp_proj.weight, -s, s)
        for block in all_blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=normal_std)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=normal_std)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
            # IHA: initialize mixing matrices to identity (baseline-equivalent)
            if block.attn.use_iha:
                torch.nn.init.eye_(block.attn.q_mix)
                torch.nn.init.eye_(block.attn.k_mix)
                if block.attn.iha_mix_v:
                    torch.nn.init.eye_(block.attn.v_mix)
            if block.attn_relax is not None:
                for adapter in block.attn_relax:
                    adapter.reset_parameters()
                for adapter in block.mlp_relax:
                    adapter.reset_parameters()

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        self.skip_weights.fill_(1.0)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim, base=10000)
        self.cos = cos
        self.sin = sin

        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)  # final layer always full context
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _avg_causal_attended_keys(self, window, seq_len):
        if window < 0 or window >= seq_len - 1:
            return (seq_len + 1) / 2
        max_keys = min(window + 1, seq_len)
        return max_keys - max_keys * (max_keys - 1) / (2 * seq_len)

    def estimate_flops(self, num_iterations=None):
        active_num_iterations = (
            self.config.num_iterations if num_iterations is None else num_iterations
        )
        if not 1 <= active_num_iterations <= self.config.num_iterations:
            raise ValueError(
                f"num_iterations must be in [1, {self.config.num_iterations}], "
                f"got {active_num_iterations}"
            )
        nparams = sum(p.numel() for p in self.parameters())
        relax_modules = [
            adapter
            for block in self.transformer.h
            for relax in (block.attn_relax, block.mlp_relax)
            if relax is not None
            for adapter in relax
        ]
        active_relax_modules = [
            adapter
            for block in self.transformer.h
            for relax in (block.attn_relax, block.mlp_relax)
            if relax is not None
            for adapter in relax[:active_num_iterations]
        ]
        relax_params = sum(p.numel() for adapter in relax_modules for p in adapter.parameters())
        relax_compute_params = sum(
            adapter.compute_matmul_params() for adapter in active_relax_modules
        )
        shared_recurrent_params = (
            sum(p.numel() for p in self.transformer.h.parameters())
            - relax_params
            + sum(p.numel() for p in self.ve_projs.parameters())
        )
        # Exclude non-matmul params: embedding lookup + elementwise scalars
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.skip_weights.numel()
        )
        extra_nonshared_params = (
            nparams - nparams_exclude - shared_recurrent_params - relax_params
        )
        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        # Exact causal sliding-window attention FLOPs: 12 * h * q * E[keys attended per query]
        attn_flops = active_num_iterations * sum(
            12 * h * q * self._avg_causal_attended_keys(w[0], t)
            for w in self.window_sizes
        )
        effective_params = (
            extra_nonshared_params
            + active_num_iterations * shared_recurrent_params
            + relax_compute_params
        )
        return 6 * effective_params + attn_flops

    def setup_optimizer(self):
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate IHA mixing params (small H×H matrices) from large matrix params
        iha_params = []
        iha_param_ids = set()
        all_blocks = list(self.transformer.h)
        if self.mtp_weight > 0:
            all_blocks_for_iha = all_blocks + [self.mtp_block]
        else:
            all_blocks_for_iha = all_blocks
        for block in all_blocks_for_iha:
            if block.attn.use_iha:
                iha_params.append(block.attn.q_mix)
                iha_params.append(block.attn.k_mix)
                iha_param_ids.add(id(block.attn.q_mix))
                iha_param_ids.add(id(block.attn.k_mix))
                if block.attn.iha_mix_v:
                    iha_params.append(block.attn.v_mix)
                    iha_param_ids.add(id(block.attn.v_mix))
        all_h_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_h_params if id(p) not in iha_param_ids] + list(
            self.ve_projs.parameters()
        )
        if self.mtp_weight > 0:
            mtp_params = [
                p
                for p in list(self.mtp_block.parameters())
                + list(self.mtp_proj.parameters())
                if id(p) not in iha_param_ids
            ]
            matrix_params += mtp_params

        ve_params = []
        embed_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        skip_params = [self.skip_weights]

        # Build param groups
        param_groups = [
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=UNEMBEDDING_LR,
                betas=ADAM_BETAS,
                eps=1e-10,
                weight_decay=WEIGHT_DECAY,
            ),
            dict(
                kind="adamw",
                params=embed_params,
                lr=EMBEDDING_LR,
                betas=ADAM_BETAS,
                eps=1e-10,
                weight_decay=WEIGHT_DECAY,
            ),
            dict(
                kind="adamw",
                params=ve_params,
                lr=EMBEDDING_LR,
                betas=ADAM_BETAS,
                eps=1e-10,
                weight_decay=WEIGHT_DECAY,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=SCALAR_LR * 0.01,
                betas=ADAM_BETAS,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=x0_params,
                lr=SCALAR_LR,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=skip_params,
                lr=SCALAR_LR * 0.01,
                betas=ADAM_BETAS,
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        # IHA mixing matrices: use AdamW with dedicated or scalar-like LR
        if iha_params:
            iha_lr = args.iha_lr if args.iha_lr is not None else SCALAR_LR
            param_groups.append(
                dict(
                    kind="adamw",
                    params=iha_params,
                    lr=iha_lr,
                    betas=ADAM_BETAS,
                    eps=1e-10,
                    weight_decay=0.0,
                )
            )
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=MATRIX_LR,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=WEIGHT_DECAY,
                )
            )

        optimizer = DistMuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _run_decoder_layers(
        self, x, x0, encoder_outputs, start, end, cos_sin, iteration_idx
    ):
        """Run decoder layers [start, end), with U-Net skip connections."""
        skip_start = self.config.n_layer - self.encoder_layers
        for i in range(start, end):
            # Encoder layer j connects to decoder layer (n_layer - 1 - j)
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = x + self.skip_weights[i - skip_start] * encoder_outputs[j]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](
                x, ve, cos_sin, self.window_sizes[i], iteration_idx
            )
        return x

    def _run_network_once(self, x, cos_sin, iteration_idx):
        x0 = x
        encoder_outputs = []
        for i in range(self.encoder_layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](
                x, ve, cos_sin, self.window_sizes[i], iteration_idx
            )
            encoder_outputs.append(x)

        dupe = self._dupe_layers
        if dupe is None:
            x = self._run_decoder_layers(
                x,
                x0,
                encoder_outputs,
                self.encoder_layers,
                self.config.n_layer,
                cos_sin,
                iteration_idx,
            )
        else:
            x = self._run_decoder_layers(
                x,
                x0,
                encoder_outputs,
                self.encoder_layers,
                dupe[1],
                cos_sin,
                iteration_idx,
            )
            for _ in range(self._dupe_loops):
                x = self._run_decoder_layers(
                    x, x0, encoder_outputs, dupe[0], dupe[1], cos_sin, iteration_idx
                )
            x = self._run_decoder_layers(
                x,
                x0,
                encoder_outputs,
                dupe[1],
                self.config.n_layer,
                cos_sin,
                iteration_idx,
            )
        return x

    def forward(
        self,
        idx,
        targets=None,
        loss_reduction="mean",
        num_iterations=None,
        allow_extra_iterations=False,
    ):
        active_num_iterations = (
            self.config.num_iterations if num_iterations is None else num_iterations
        )
        max_num_iterations = self.config.num_iterations
        if allow_extra_iterations and not self.training:
            max_num_iterations += FINAL_EXTRA_EVAL_ITERATIONS
        if not 1 <= active_num_iterations <= max_num_iterations:
            raise ValueError(
                f"num_iterations must be in [1, {max_num_iterations}], "
                f"got {active_num_iterations}"
            )
        _, T = idx.size()
        x = norm(self.transformer.wte(idx))
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        for iteration in range(active_num_iterations):
            x = self._run_network_once(x, cos_sin, iteration)
            x = norm(x)

        logits = self.lm_head(x)[..., : self.config.vocab_size].float()
        logits = LOGIT_CAP * torch.tanh(logits / LOGIT_CAP) if LOGIT_CAP > 0 else logits

        if targets is None:
            return logits

        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        if loss_reduction != "mean":
            return lm_loss

        if self.mtp_weight <= 0:
            return lm_loss, {"lm_loss": lm_loss}

        mtp_emb = norm(self.transformer.wte(targets[:, :-1].clamp(min=0)))
        combined = self.mtp_proj(torch.cat([x[:, :-1], mtp_emb], dim=-1))
        mT = combined.size(1)
        mtp_out = norm(
            self.mtp_block(
                combined, None, (self.cos[:, :mT], self.sin[:, :mT]), (-1, -1)
            )
        )
        mtp_logits = self.lm_head(mtp_out)[..., : self.config.vocab_size].float()
        if LOGIT_CAP > 0:
            mtp_logits = LOGIT_CAP * torch.tanh(mtp_logits / LOGIT_CAP)

        mtp_loss = F.cross_entropy(
            mtp_logits.view(-1, mtp_logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=-1,
        )
        loss = lm_loss + self.mtp_weight * mtp_loss
        return loss, {"lm_loss": lm_loss, "mtp_loss": mtp_loss}


# =============================================================================
# Optimizer: MuonAdamW (Muon for matrices, AdamW for embeddings/scalars)
# =============================================================================

# Polar Express coefficients for orthogonalization
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t
):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    active_mask,
    ns_steps,
    red_dim,
):
    momentum = momentum_t.to(stacked_grads.dtype)
    active = active_mask.to(stacked_grads.dtype)
    momentum_update = (1 - momentum) * active
    momentum_buffer.mul_(1 - momentum_update).add_(stacked_grads * momentum_update)
    g = stacked_grads.lerp(momentum_buffer, momentum) * active
    # MuonEq-R row normalization
    g /= g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
    # Polar Express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_update = (1 - beta2) * active.to(second_momentum_buffer.dtype)
    second_momentum_buffer.mul_(1 - second_momentum_update).add_(
        v_mean.to(dtype=second_momentum_buffer.dtype) * second_momentum_update
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_((lr * g + lr * wd * stacked_params * mask) * active)


class DistMuonAdamW(torch.optim.Optimizer):
    """Distributed MuonAdamW with ZeRO-2 style sharding."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._scalar_device = None
        self._adamw_step_t = torch.empty(())
        self._adamw_lr_t = torch.empty(())
        self._adamw_beta1_t = torch.empty(())
        self._adamw_beta2_t = torch.empty(())
        self._adamw_eps_t = torch.empty(())
        self._adamw_wd_t = torch.empty(())
        self._muon_momentum_t = torch.empty(())
        self._muon_lr_t = torch.empty(())
        self._muon_wd_t = torch.empty(())
        self._muon_beta2_t = torch.empty(())
        self._group_buffers = {}

    def _ensure_scalar_tensors(self, device):
        if self._scalar_device == device:
            return
        self._scalar_device = device
        self._adamw_step_t = torch.empty((), device=device)
        self._adamw_lr_t = torch.empty((), device=device)
        self._adamw_beta1_t = torch.empty((), device=device)
        self._adamw_beta2_t = torch.empty((), device=device)
        self._adamw_eps_t = torch.empty((), device=device)
        self._adamw_wd_t = torch.empty((), device=device)
        self._muon_momentum_t = torch.empty((), device=device)
        self._muon_lr_t = torch.empty((), device=device)
        self._muon_wd_t = torch.empty((), device=device)
        self._muon_beta2_t = torch.empty((), device=device)

    def _buffers_for_group(self, group):
        return self._group_buffers.setdefault(id(group), {})

    def _reduce_adamw(self, group, world_size):
        infos = {}
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if p.numel() < 1024:
                future = dist.all_reduce(
                    grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                state = self.state[p]
                grad_slice = state.get("_grad_slice")
                if grad_slice is None:
                    grad_slice = torch.empty_like(grad[:rank_size])
                    state["_grad_slice"] = grad_slice
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=infos)

    def _reduce_muon(self, group, world_size):
        params = group["params"]
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        buffers = self._buffers_for_group(group)
        stacked_grads = buffers.get("stacked_grads")
        if stacked_grads is None:
            stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
            buffers["stacked_grads"] = stacked_grads
            buffers["stacked_grad_views"] = list(
                stacked_grads[: len(params)].unbind(0)
            )
        active_mask_shape = (padded,) + (1,) * len(shape)
        active_mask = buffers.get("active_mask")
        if active_mask is None:
            active_mask = torch.empty(active_mask_shape, dtype=torch.bool, device=device)
            buffers["active_mask"] = active_mask
        active_mask.zero_()
        for i, p in enumerate(params):
            if p.grad is None:
                buffers["stacked_grad_views"][i].zero_()
            else:
                buffers["stacked_grad_views"][i].copy_(p.grad)
                active_mask[i].fill_(True)
        if len(params) < padded:
            stacked_grads[len(params) :].zero_()
            active_mask[len(params) :].zero_()
        grad_chunk = buffers.get("grad_chunk")
        if grad_chunk is None:
            grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
            buffers["grad_chunk"] = grad_chunk
        future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()
        return dict(
            future=future,
            grad_chunk=grad_chunk,
            stacked_grads=stacked_grads,
            active_mask=active_mask,
            chunk_size=chunk_size,
        )

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p, pinfo in info["param_infos"].items():
            pinfo["future"].wait()
            state = self.state[p]
            if pinfo["is_small"]:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p_slice)
                state["exp_avg_sq"] = torch.zeros_like(p_slice)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p_slice,
                pinfo["grad_slice"],
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )
            if not pinfo["is_small"]:
                future = dist.all_gather_into_tensor(
                    p, p_slice, async_op=True
                ).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        info["future"].wait()
        params = group["params"]
        chunk_size = info["chunk_size"]
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                chunk_size, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            s = (
                (chunk_size, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (chunk_size, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        buffers = self._buffers_for_group(group)
        updated = buffers.get("updated")
        if updated is None:
            updated = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
            buffers["updated"] = updated
            buffers["updated_views"] = list(updated.unbind(0))
        if num_owned > 0:
            updated_owned = updated[:num_owned]
            torch._foreach_copy_(
                buffers["updated_views"][:num_owned],
                [params[start_idx + i] for i in range(num_owned)],
            )
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                info["grad_chunk"][:num_owned],
                updated_owned,
                state["momentum_buffer"][:num_owned],
                state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t,
                self._muon_lr_t,
                self._muon_wd_t,
                self._muon_beta2_t,
                info["active_mask"][start_idx : start_idx + num_owned],
                group["ns_steps"],
                red_dim,
            )
        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(
            stacked_params, updated, async_op=True
        ).get_future()
        gather_list.append(
            dict(future=future, stacked_params=stacked_params, params=params)
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["params"]:
                self._ensure_scalar_tensors(group["params"][0].device)
                break
        rank, world_size = dist.get_rank(), dist.get_world_size()
        reduce_infos = []
        for group in self.param_groups:
            if group["kind"] == "adamw":
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group["kind"] == "muon":
                reduce_infos.append(self._reduce_muon(group, world_size))
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group["kind"] == "adamw":
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group["kind"] == "muon":
                self._compute_muon(group, info, gather_list, rank)
        for info in gather_list:
            info["future"].wait()
            if info.get("params") is not None:
                torch._foreach_copy_(
                    info["params"],
                    list(info["stacked_params"][: len(info["params"])].unbind(0)),
                )


# =============================================================================
# Dataloader: BOS-aligned best-fit packing
# =============================================================================


class DataLoader:
    """Loads flat tokens , chunks into batches.

    doc_shuffle=False: applies the stored default sequence permutation (bitwise match
    with the old chunked pipeline), shuffles batch order each epoch.
    doc_shuffle=True: reshuffles documents each epoch, re-chunks, re-shuffles sequences.

    Always yields (x, y, epoch).
    """

    def __init__(self, filepath, B, T, device="cuda", doc_shuffle=False):
        data = torch.load(filepath, weights_only=True)
        all_tokens = data["tokens"].long()
        raw_doc_starts = data["doc_starts"].long()
        bos_id = int(data["bos_id"])
        assert bos_id == BOS_ID, f"data bos_id {bos_id} != expected {BOS_ID}"

        doc_ends = torch.cat([raw_doc_starts[1:], torch.tensor([all_tokens.numel()])])
        self.doc_tokens = [
            all_tokens[s:e]
            for s, e in zip(raw_doc_starts.tolist(), doc_ends.tolist())
        ]
        self.default_shuffle_seed = data["seq_shuffle_seed"]

        _, rank, _, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.B = B
        self.T = T
        self.seq_size = T + 1
        self.doc_shuffle = doc_shuffle
        self.epoch = 1
        self._build_batches()

    def _build_batches(self):
        tokens = torch.cat(self.doc_tokens)
        num_seqs = len(tokens) // self.seq_size
        all_seqs = tokens[:num_seqs * self.seq_size].view(num_seqs, self.seq_size)
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + 1000)
            all_seqs = all_seqs[torch.randperm(num_seqs, generator=g)]
        else:
            perm = np.random.RandomState(self.default_shuffle_seed).permutation(
                num_seqs
            )
            all_seqs = all_seqs[torch.from_numpy(perm)]
        seqs_per_step = self.B * self.world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        all_seqs = all_seqs[:usable].view(
            num_steps, self.world_size, self.B, self.seq_size
        )
        self.rank_data = all_seqs[:, self.rank].contiguous()
        self.num_steps = num_steps
        self.total_tokens = usable * self.T
        self.pos = 0

    def __iter__(self):
        return self

    def _next_epoch(self):
        self.epoch += 1
        print0(f"Starting epoch {self.epoch}")
        if self.doc_shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            perm = torch.randperm(len(self.doc_tokens), generator=g)
            self.doc_tokens = [self.doc_tokens[i] for i in perm.tolist()]
            self._build_batches()
        else:
            self.pos = 0
            g = torch.Generator()
            g.manual_seed(self.epoch)
            self.rank_data = self.rank_data[torch.randperm(self.num_steps, generator=g)]

    def __next__(self):
        if self.pos >= self.num_steps:
            self._next_epoch()
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch


# =============================================================================
# Loss evaluation
# =============================================================================


@torch.no_grad()
def evaluate_bpb(
    model,
    batches,
    steps,
    token_bytes,
    num_iterations=None,
    allow_extra_iterations=False,
):
    """Compute bits per byte and mean cross-entropy loss on a set of batches."""
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    model_kwargs = {"num_iterations": num_iterations}
    if allow_extra_iterations:
        model_kwargs["allow_extra_iterations"] = True
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        loss2d = model(
            x,
            y,
            loss_reduction="none",
            **model_kwargs,
        ).view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float("inf")
    loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return bpb, loss


@torch.no_grad()
def evaluate_bpb_logit_avg(
    eval_model,
    ckpt_paths,
    weights,
    steps,
    num_iterations=None,
    allow_extra_iterations=False,
):
    """Evaluate using probability averaging across checkpoints (proper ensemble).

    Loads each checkpoint from disk once, runs all val batches for it, then
    moves to the next — one CPU->GPU weight transfer per checkpoint, not per batch.
    Accumulates running scalar totals instead of per-token tensors.
    """
    dev = orig_model.get_device()
    V = orig_model.config.vocab_size

    # Pre-fetch all val batches to CPU (token ids, tiny ~10 MB)
    val_loader = build_val_loader()
    all_x, all_y = [], []
    for _ in range(steps):
        x, y, _ = next(val_loader)
        all_x.append(x.cpu())
        all_y.append(y.cpu())

    BT = all_y[0].numel()

    # Per-batch accumulated weighted target probs, kept on GPU
    # Shape: (steps, BT) — only target-token probs, not full vocab
    batch_target_probs = torch.zeros(steps, BT, dtype=torch.float32, device=dev)

    # Checkpoint-outer, batch-inner: each checkpoint loaded exactly once
    model_kwargs = {"num_iterations": num_iterations}
    if allow_extra_iterations:
        model_kwargs["allow_extra_iterations"] = True
    for path, w in zip(ckpt_paths, weights):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        load_state_dict_into_model(orig_model, ckpt)
        del ckpt
        for i, (x, y) in enumerate(zip(all_x, all_y)):
            y_flat = y.view(-1).to(dev)
            with autocast_ctx:
                logits = eval_model(
                    x.to(dev),
                    **model_kwargs,
                )
            probs = torch.softmax(logits.view(BT, V).float(), dim=-1)
            tgt = probs[torch.arange(BT, device=dev), y_flat.clamp_min(0)]
            batch_target_probs[i].add_(tgt, alpha=w)

    # Compute metrics from accumulated target probs using running totals
    total_nats = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=dev)
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=dev)

    for i, y in enumerate(all_y):
        y_flat = y.view(-1).to(dev)
        mask = y_flat != -1
        log_probs = batch_target_probs[i].clamp_min(1e-40).log()
        num_bytes_batch = token_bytes[y_flat.clamp_min(0)]

        total_nats += (log_probs.neg() * (num_bytes_batch > 0)).sum().double()
        total_bytes += num_bytes_batch.sum()
        total_loss += log_probs[mask].neg().sum().double()
        total_tokens += mask.sum()

    del batch_target_probs

    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    bpb = (
        total_nats.item() / (math.log(2) * total_bytes.item())
        if total_bytes.item() > 0
        else float("inf")
    )
    loss = (
        total_loss.item() / total_tokens.item()
        if total_tokens.item() > 0
        else float("inf")
    )
    return bpb, loss


# =============================================================================
# Training
# =============================================================================

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(42)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type = device.type

autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0


def debug_memory_stats():
    if device_type != "cuda":
        return ""
    stats = torch.cuda.memory_stats(device)
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    inactive = stats.get("inactive_split_bytes.all.current", 0) / 1e9
    retries = stats.get("num_alloc_retries", 0)
    return (
        f" | mem: alloc={allocated:.2f}G reserved={reserved:.2f}G"
        f" inactive={inactive:.2f}G alloc_retries={retries}"
    )


def precompile_iteration_stages(model, x, y, iteration_counts):
    if len(iteration_counts) <= 1:
        return
    print0(f"Precompiling recurrent iteration stages: {iteration_counts}")
    was_training = model.training
    for count in iteration_counts:
        model.train()
        model.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss, _ = model(x, y, num_iterations=count)
        loss.backward()
        model.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            with autocast_ctx:
                model(x, y, loss_reduction="none", num_iterations=count)
        synchronize()
    model.train(was_training)
    model.zero_grad(set_to_none=True)


# GPU info for MFU
gpu_peak_flops = float("inf")
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name:
        gpu_peak_flops = 989e12
    elif "a100" in gpu_name:
        gpu_peak_flops = 312e12
    elif "4090" in gpu_name:
        gpu_peak_flops = 165.2e12

# FA3 status
if _fa3 is not None:
    print0("Using Flash Attention 3 (Hopper GPU detected)")
else:
    raise RuntimeError(
        "Flash Attention 3 is required but was not importable on this Hopper run. "
        "Expected an official FA3 install providing `import flash_attn_interface` "
        "(see flash-attention/hopper `python setup.py install`). "
        f"Loader details: {getattr(_load_fa3, 'last_error', 'no additional details')}"
    )

# Run / logging paths
run_name, run_dir = resolve_run_dir(args.run_name)
if dist.is_initialized():
    shared = [run_name]
    dist.broadcast_object_list(shared, src=0)
    run_name = shared[0]
    run_dir = os.path.join(RUNS_DIR, run_name)
checkpoints_dir = os.path.join(run_dir, "checkpoints")
terminal_log_path = os.path.join(run_dir, "terminal.log")
stdout_orig = sys.stdout
stderr_orig = sys.stderr
artifacts_log_f = None
result_path = os.path.join(run_dir, "result.json")
if master_process:
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
    shutil.copy2(__file__, os.path.join(run_dir, "train.py"))
    artifacts_log_f = open(terminal_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, artifacts_log_f)
    sys.stderr = TeeStream(sys.stderr, artifacts_log_f)

# wandb
_wandb_kwargs = {"project": "slowrun", "name": run_name, "dir": os.path.join(run_dir, "wandb")}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = DummyWandb() if not master_process else wandb.init(**_wandb_kwargs)
if master_process:
    wandb_run.log_code(".")

# Print hyperparameters
print0("--- Hyperparameters ---")
print0(f"  n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}, head_dim={HEAD_DIM}")
print0(f"  seq_len={MAX_SEQ_LEN}, window_pattern={WINDOW_PATTERN}")
print0(f"  max_num_iterations={NUM_ITERATIONS}, hira_rank={args.hira_rank}")
print0(
    f"  iteration_schedule={args.iteration_schedule} "
    f"({format_iteration_schedule(ITERATION_SCHEDULE)}), "
    f"avg_layer_passes={DEPTH * ITERATION_SCHEDULE.avg_iterations:.3f}, "
    f"avg_compute_equiv_layers={DEPTH * ITERATION_SCHEDULE.avg_iterations * COMPUTE_WIDTH_SCALE:.3f}"
)
print0(
    f"  compute_reference_n_embd={args.compute_reference_n_embd}, "
    f"compute_width_power={args.compute_width_power}, "
    f"compute_width_scale={COMPUTE_WIDTH_SCALE:.3f}"
)
print0(f"  stoch_depth={args.stoch_depth}")
print0(
    f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={args.device_batch_size}"
)
print0(
    f"  matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embedding_lr={EMBEDDING_LR}, unembedding_lr={UNEMBEDDING_LR}"
)
print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}")
print0(
    f"  warmup_ratio={WARMUP_RATIO}, warmdown_ratio={WARMDOWN_RATIO}, final_lr_frac={FINAL_LR_FRAC}"
)
print0(f"  num_epochs={args.num_epochs}, patience={args.patience}")
print0(f"  dropout={args.dropout}, doc_shuffle={not args.no_doc_shuffle}")
print0(f"  run={run_name}")
print0(f"  run_dir={run_dir}")
if args.iha:
    print0(f"  iha=True, iha_lr={args.iha_lr}")
print0("-----------------------")

# Load GPT-2 tokenizer and compute token_bytes for BPB evaluation
encoder = tiktoken.get_encoding("gpt2")
vocab_size = encoder.n_vocab  # 50257
print0(f"Vocab size: {vocab_size:,}")

eot_id = encoder._special_tokens["<|endoftext|>"]
token_bytes_list = []
for i in range(vocab_size):
    if i == eot_id:
        token_bytes_list.append(0)
    else:
        token_bytes_list.append(len(encoder.decode_single_token_bytes(i)))
token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

# Build model
config = GPTConfig(
    vocab_size=vocab_size,
    dropout=args.dropout,
    stoch_depth=args.stoch_depth,
    num_iterations=NUM_ITERATIONS,
    hira_rank=args.hira_rank,
    use_iha=args.iha,
    iha_mix_v=args.iha,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
print0(f"HiRA parameter relaxation: {'enabled' if args.hira_rank > 0 else 'disabled'}")

param_counts = sum(p.numel() for p in model.parameters())
transformer_params = sum(p.numel() for p in model.transformer.h.parameters())
ve_params = sum(p.numel() for p in model.ve_projs.parameters())
lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
other_params = param_counts - transformer_params - ve_params - lm_head_params
max_flops_per_token = model.estimate_flops(NUM_ITERATIONS)
scheduled_iteration_counts = iteration_schedule_counts(ITERATION_SCHEDULE)
schedule_flops_per_token = {
    iteration_count: model.estimate_flops(iteration_count)
    for iteration_count in scheduled_iteration_counts
}
avg_flops_per_token = sum(
    (stage.end_frac - stage.start_frac)
    * schedule_flops_per_token[stage.iterations]
    for stage in ITERATION_SCHEDULE.stages
)
print0(
    f"Parameters: {param_counts:,} (transformer: {transformer_params:,}, value_embeds: {ve_params:,}, lm_head: {lm_head_params:,}, other: {other_params:,})"
)
print0(f"FLOPs per token at max iterations: {max_flops_per_token:e}")
print0(f"Schedule-average FLOPs per token: {avg_flops_per_token:e}")

# Compile
orig_model = model
model = torch.compile(model, dynamic=False)

# Optimizer
optimizer = model.setup_optimizer()

# Dataloaders
_train_path = (
    args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
)
_val_path = (
    args.input_val_bin
    if args.input_val_bin
    else os.path.join(DATA_DIR, "fineweb_val.pt")
)
train_loader = DataLoader(
    _train_path,
    args.device_batch_size,
    MAX_SEQ_LEN,
    device=device,
    doc_shuffle=not args.no_doc_shuffle,
)
build_val_loader = lambda: DataLoader(
    _val_path, args.device_batch_size, MAX_SEQ_LEN, device=device
)
TOKENS_PER_EPOCH = train_loader.total_tokens
x, y, current_epoch = next(train_loader)

# Training config
tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
num_iterations = round(
    TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE
)  # estimate for LR schedule
print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
print0(f"Training for {args.num_epochs} epoch(s) (~{num_iterations} steps estimated)")
print0(f"Recurrent iteration schedule: {format_iteration_schedule(ITERATION_SCHEDULE)}")
for stage in ITERATION_SCHEDULE.stages:
    print0(
        f"  stage {stage.start_frac:.3f}-{stage.end_frac:.3f} of epochs "
        f"(~steps {round(stage.start_frac * num_iterations)}-"
        f"{round(stage.end_frac * num_iterations)}): {stage.iterations} iteration(s)"
    )
print0(f"Eval set: {EVAL_TOKENS:,} tokens")


# Schedulers
def get_lr_multiplier(it):
    warmup = round(WARMUP_RATIO * num_iterations)
    warmdown = round(WARMDOWN_RATIO * num_iterations)
    if it < warmup:
        return (it + 1) / warmup
    elif it <= num_iterations - warmdown:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown
        shaped = progress ** WARMDOWN_POWER  # concave (stays higher longer) when POWER < 1
        return shaped + (1 - shaped) * FINAL_LR_FRAC


def get_muon_momentum(it):
    return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95


steps_per_epoch = num_iterations / args.num_epochs
_swa_start_step = (
    (num_iterations - args.swa_last_epochs * steps_per_epoch)
    if args.swa_last_epochs > 0
    else -1
)


def get_wd_multiplier(it):
    """Anti-phase WD: hold at 1.0 pre-SWA, decay to LOW by SWA start, then sawtooth LOW→HIGH per SWA epoch (anti-phase with the LR cosine cycle)."""
    if _swa_start_step >= 0 and it >= _swa_start_step:
        cycle_pos = (it - _swa_start_step) % steps_per_epoch
        frac = cycle_pos / steps_per_epoch
        return WD_SWA_LOW_FACTOR + (WD_SWA_HIGH_FACTOR - WD_SWA_LOW_FACTOR) * frac
    t = it / num_iterations
    if t < WD_PRE_HOLD_FRAC:
        return 1.0
    swa_start_frac = _swa_start_step / num_iterations if _swa_start_step > 0 else 1.0
    decay_frac = (t - WD_PRE_HOLD_FRAC) / max(swa_start_frac - WD_PRE_HOLD_FRAC, 1e-6)
    decay_frac = min(max(decay_frac, 0.0), 1.0)
    return 1.0 - (1.0 - WD_SWA_LOW_FACTOR) * decay_frac


# Training loop
step = 0
active_num_iterations = get_scheduled_iterations(
    ITERATION_SCHEDULE, step, num_iterations
)
min_val_bpb = float("inf")
min_val_loss = float("inf")
epochs_without_improvement = 0
smooth_train_loss = 0
total_training_time = 0
timed_steps = 0
timing_start_step = 4  # skip first compile + 3 warmup steps
eval_steps = EVAL_TOKENS // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size)
dupe_active = False

# Logit averaging setup
late_checkpoint_paths = []  # paths to saved epoch checkpoints for logit averaging
logit_avg_count = args.logit_avg
if logit_avg_count > 0 and master_process:
    os.makedirs(args.logit_avg_dir, exist_ok=True)
if logit_avg_count > 0:
    print0(
        f"Logit averaging: saving last {logit_avg_count} epoch checkpoints to {args.logit_avg_dir}/"
    )

if not args.eval_logit_avg:
    precompile_iteration_stages(model, x, y, scheduled_iteration_counts)

if args.eval_logit_avg:
    print0("--eval-logit-avg set: skipping training, loading checkpoints from disk.")
else:
    # Initial val evaluation
    model.eval()
    val_loader = build_val_loader()
    with autocast_ctx:
        val_bpb, val_loss = evaluate_bpb(
            model,
            val_loader,
            eval_steps,
            token_bytes,
            num_iterations=active_num_iterations,
        )
    print0(
        f"Step {step:05d} | Val BPB: {val_bpb:.6f} | "
        f"Val Loss: {val_loss:.6f} | iters: {active_num_iterations}"
    )
    wandb_run.log({
        "step": step,
        "val/bpb": val_bpb,
        "val/loss": val_loss,
        "val/num_iterations": active_num_iterations,
    })
    min_val_bpb = val_bpb
    min_val_loss = val_loss
    model.train()

while not args.eval_logit_avg and current_epoch <= args.num_epochs:
    next_num_iterations = get_scheduled_iterations(
        ITERATION_SCHEDULE, step, num_iterations
    )
    if next_num_iterations != active_num_iterations:
        active_num_iterations = next_num_iterations
        approx_epoch = step / steps_per_epoch if steps_per_epoch > 0 else 0.0
        print0(
            f"\n=== Recurrent iterations -> {active_num_iterations} "
            f"at step {step} ({100 * step / num_iterations:.2f}%, "
            f"epoch progress {approx_epoch:.2f}/{args.num_epochs}) ==="
        )
        timing_start_step = step + 4  # skip compile/specialization + 3 warmup steps
        gc.enable()
        gc.collect()

    if (
        not dupe_active
        and args.dupe_start_epoch is not None
        and current_epoch >= args.dupe_start_epoch
    ):
        print0(f"\n=== Enabling dupe-layers at epoch {current_epoch} ===")
        orig_model.set_dupe_layers(
            args.dupe_layers_start, args.dupe_layers_end, args.dupe_loops
        )
        model = torch.compile(orig_model, dynamic=False)
        # model = orig_model # replace compile with this line for eager mode
        dupe_active = True
        timing_start_step = step + 4  # skip dupe recompile + 3 warmup steps
        gc.enable()
        gc.collect()

    # Training step
    synchronize()
    t0 = time.time()
    debug_timing = args.debug_timing_steps > 0 and step < args.debug_timing_steps
    timings = {"fwd_bwd": 0.0, "data": 0.0} if debug_timing else None
    for micro_step in range(grad_accum_steps):
        if debug_timing:
            synchronize()
            section_t0 = time.time()
        with autocast_ctx:
            loss, metrics = model(x, y, num_iterations=active_num_iterations)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        if debug_timing:
            synchronize()
            timings["fwd_bwd"] += time.time() - section_t0
            section_t0 = time.time()
        x, y, epoch = next(train_loader)
        if debug_timing:
            synchronize()
            timings["data"] += time.time() - section_t0

    # Update LR/WD/momentum
    lrm = get_lr_multiplier(step)
    # SWA: cosine-cycle LR in final epochs for diverse checkpoints to average
    if _swa_start_step >= 0 and step >= _swa_start_step:
        cycle_pos = (step - _swa_start_step) % steps_per_epoch
        swa_base = max(lrm, 0.05)
        lrm = 0.05 + (swa_base - 0.05) * (1 + math.cos(math.pi * cycle_pos / steps_per_epoch)) / 2
    # WD schedule: pre-SWA decay from base to 0.65×base, then anti-phase sawtooth
    # during SWA's N epochs (0.65→1.50×base per epoch, anti-phase with LR cosine cycle).
    wdm = get_wd_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if "initial_wd" not in group:
            group["initial_wd"] = group.get("weight_decay", 0.0)
        group["weight_decay"] = group["initial_wd"] * wdm
        if group['kind'] == 'muon':
            group["momentum"] = get_muon_momentum(step)

    # Optimizer step
    if debug_timing:
        synchronize()
        section_t0 = time.time()
    optimizer.step()
    if debug_timing:
        synchronize()
        timings["optimizer"] = time.time() - section_t0
        section_t0 = time.time()
    model.zero_grad(set_to_none=True)
    if debug_timing:
        synchronize()
        timings["zero_grad"] = time.time() - section_t0

    train_loss_f = train_loss.item()
    synchronize()
    dt = time.time() - t0

    step += 1

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta**step)
    pct = 100 * step / num_iterations
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    active_flops_per_token = schedule_flops_per_token[active_num_iterations]
    mfu = (
        100
        * active_flops_per_token
        * TOTAL_BATCH_SIZE
        / dt
        / (gpu_peak_flops * ddp_world_size)
    )
    if step >= timing_start_step:
        total_training_time += dt
        timed_steps += 1
    eta_str = (
        f" | eta: {(num_iterations - step) * total_training_time / timed_steps / 60:.1f}m"
        if timed_steps > 0
        else ""
    )
    dupe_str = " [DUPE]" if dupe_active else ""
    print0(
        f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | "
        f"iters: {active_num_iterations} | dt: {dt * 1000:.2f}ms | "
        f"tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{dupe_str}{eta_str}"
    )
    timing_metrics = {}
    if timings is not None:
        section_sum = sum(timings.values())
        print0(
            "  timing | "
            + " ".join(f"{k}={v * 1000:.2f}ms" for k, v in timings.items())
            + f" accounted={section_sum / dt * 100:.1f}%"
            + debug_memory_stats()
        )
        timing_metrics = {f"timing/{k}_ms": v * 1000 for k, v in timings.items()}
    wandb_run.log({
        "step": step,
        "train/loss": debiased,
        "train/mfu": mfu,
        "train/num_iterations": active_num_iterations,
        "train/effective_layers": DEPTH * active_num_iterations,
        "train/flops_per_token": active_flops_per_token,
        **timing_metrics,
        **{f"train/{k}": v.item() for k, v in metrics.items()},
    })

    # Synchronize epoch across ranks (different ranks may exhaust data at different steps)
    if ddp:
        epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
        dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
        epoch = epoch_tensor.item()

    # Epoch boundary: evaluate when the dataloader advances to a new epoch
    if epoch != current_epoch:
        eval_num_iterations = get_scheduled_iterations(
            ITERATION_SCHEDULE, step, num_iterations
        )
        model.eval()
        val_loader = build_val_loader()
        with autocast_ctx:
            val_bpb, val_loss = evaluate_bpb(
                model,
                val_loader,
                eval_steps,
                token_bytes,
                num_iterations=eval_num_iterations,
            )
        print0(
            f"Step {step:05d} | Epoch {current_epoch} | "
            f"Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f} "
            f"| iters: {eval_num_iterations}"
        )
        wandb_run.log({
            "step": step,
            "epoch": current_epoch,
            "val/bpb": val_bpb,
            "val/loss": val_loss,
            "val/num_iterations": eval_num_iterations,
        })

        # Early stopping
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
            min_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience >= 0 and epochs_without_improvement >= args.patience:
                print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                break

        # Save checkpoint to disk for logit averaging
        if logit_avg_count > 0:
            ckpt_path = os.path.join(
                args.logit_avg_dir, f"epoch_{current_epoch:03d}.pt"
            )
            if master_process:
                ckpt = {
                    name: p.data.float().cpu()
                    for name, p in orig_model.named_parameters()
                }
                torch.save(ckpt, ckpt_path)
                del ckpt
            late_checkpoint_paths.append(ckpt_path)
            if len(late_checkpoint_paths) > logit_avg_count:
                old = late_checkpoint_paths.pop(0)
                if master_process and os.path.exists(old):
                    os.remove(old)
            print0(
                f"  Saved checkpoint {ckpt_path} ({len(late_checkpoint_paths)}/{logit_avg_count})"
            )

        model.train()
        current_epoch = epoch

    # GC management
    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()

# =============================================================================
# Post-training: evaluate checkpoint averages
# =============================================================================

final_extra_iter_result = None

# Evaluate logit (probability) average
if logit_avg_count > 0:
    # In eval-only mode, discover checkpoints from disk; otherwise use what was saved during training
    if args.eval_logit_avg:
        import glob as _glob

        all_disk = sorted(_glob.glob(os.path.join(args.logit_avg_dir, "epoch_*.pt")))
        ckpt_paths_for_logit = all_disk[-logit_avg_count:]
    else:
        ckpt_paths_for_logit = late_checkpoint_paths

    if len(ckpt_paths_for_logit) >= 2:
        n = len(ckpt_paths_for_logit)
        logit_avg_num_iterations = ITERATION_SCHEDULE.stages[-1].iterations
        print0(
            f"\n--- Evaluating logit avg ({n} checkpoints: "
            f"{[os.path.basename(p) for p in ckpt_paths_for_logit]}, "
            f"iters={logit_avg_num_iterations}) ---"
        )

        la_model = torch.compile(orig_model, dynamic=False)
        la_model.eval()

        def _run_mode(
            label, weights, eval_num_iterations, allow_extra_iterations=False
        ):
            print0(f"  [{label}] weights: {[f'{w:.3f}' for w in weights]}")
            bpb, loss = evaluate_bpb_logit_avg(
                la_model,
                ckpt_paths_for_logit,
                weights,
                eval_steps,
                num_iterations=eval_num_iterations,
                allow_extra_iterations=allow_extra_iterations,
            )
            print0(
                f"  [{label}] Val BPB: {bpb:.6f} | Val Loss: {loss:.6f} "
                f"| iters: {eval_num_iterations}"
            )
            wandb_run.log({
                f"logit_avg_{label}/bpb": bpb,
                f"logit_avg_{label}/loss": loss,
                f"logit_avg_{label}/num_iterations": eval_num_iterations,
            })
            return bpb, loss

        equal_w = [1.0 / n] * n
        raw_w = list(range(1, n + 1))
        weighted_w = [w / sum(raw_w) for w in raw_w]
        best_logit_avg = {
            "label": None,
            "weights": None,
            "bpb": float("inf"),
            "loss": float("inf"),
        }

        if args.logit_avg_mode in ("equal", "both"):
            eq_bpb, eq_loss = _run_mode("equal", equal_w, logit_avg_num_iterations)
            if eq_loss < best_logit_avg["loss"]:
                best_logit_avg.update(
                    label="equal", weights=list(equal_w), bpb=eq_bpb, loss=eq_loss
                )
            if eq_loss < min_val_loss:
                min_val_loss, min_val_bpb = eq_loss, eq_bpb
                print0("  ** New best! (logit avg equal weights)")

        if args.logit_avg_mode in ("weighted", "both"):
            wt_bpb, wt_loss = _run_mode(
                "weighted", weighted_w, logit_avg_num_iterations
            )
            if wt_loss < best_logit_avg["loss"]:
                best_logit_avg.update(
                    label="weighted",
                    weights=list(weighted_w),
                    bpb=wt_bpb,
                    loss=wt_loss,
                )
            if wt_loss < min_val_loss:
                min_val_loss, min_val_bpb = wt_loss, wt_bpb
                print0("  ** New best! (logit avg recency weights)")

        if best_logit_avg["weights"] is not None:
            extra_logit_avg_num_iterations = (
                logit_avg_num_iterations + FINAL_EXTRA_EVAL_ITERATIONS
            )
            extra_label = (
                f"{best_logit_avg['label']}_plus_{FINAL_EXTRA_EVAL_ITERATIONS}_iter"
            )
            print0(
                f"\n--- Re-evaluating best logit avg "
                f"({best_logit_avg['label']}, loss={best_logit_avg['loss']:.6f}) "
                f"at {extra_logit_avg_num_iterations} recurrent iterations "
                f"(+{FINAL_EXTRA_EVAL_ITERATIONS}); "
                "extra passes reuse the final trained HiRA adapter ---"
            )
            extra_bpb, extra_loss = _run_mode(
                extra_label,
                best_logit_avg["weights"],
                extra_logit_avg_num_iterations,
                allow_extra_iterations=True,
            )
            final_extra_iter_result = {
                "source": f"logit_avg_{best_logit_avg['label']}",
                "base_num_iterations": logit_avg_num_iterations,
                "extra_iterations": FINAL_EXTRA_EVAL_ITERATIONS,
                "num_iterations": extra_logit_avg_num_iterations,
                "bpb": extra_bpb,
                "loss": extra_loss,
            }


# Summary
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float("inf")
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
if final_extra_iter_result is not None:
    print0(
        "Final +3-iter eval "
        f"({final_extra_iter_result['source']}, "
        f"{final_extra_iter_result['num_iterations']} iters) "
        f"BPB: {final_extra_iter_result['bpb']:.6f} | "
        f"Loss: {final_extra_iter_result['loss']:.6f}"
    )

wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss
if final_extra_iter_result is not None:
    wandb_run.summary["final_extra_iter_eval_loss"] = final_extra_iter_result["loss"]
    wandb_run.summary["final_extra_iter_eval_bpb"] = final_extra_iter_result["bpb"]

_result_out = args.save_result or result_path
if master_process:
    result = {
        "matrix_lr": args.matrix_lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "max_num_iterations": NUM_ITERATIONS,
        "iteration_schedule": args.iteration_schedule,
        "iteration_schedule_stages": [
            {
                "start_frac": stage.start_frac,
                "end_frac": stage.end_frac,
                "iterations": stage.iterations,
            }
            for stage in ITERATION_SCHEDULE.stages
        ],
        "avg_effective_layers": DEPTH * ITERATION_SCHEDULE.avg_iterations,
        "avg_effective_layer_passes": DEPTH * ITERATION_SCHEDULE.avg_iterations,
        "avg_compute_equivalent_layers": (
            DEPTH * ITERATION_SCHEDULE.avg_iterations * COMPUTE_WIDTH_SCALE
        ),
        "compute_reference_n_embd": args.compute_reference_n_embd,
        "compute_width_power": args.compute_width_power,
        "compute_width_scale": COMPUTE_WIDTH_SCALE,
        "val_loss": val_loss,
        "best_val_loss": min_val_loss,
        "wandb_url": getattr(wandb_run, "url", None),
    }
    if final_extra_iter_result is not None:
        result["final_extra_iter_eval"] = final_extra_iter_result
    with open(_result_out, "w") as f:
        json.dump(result, f, indent=2)
    print0(f"Result saved to {_result_out}")

total_wall_time = time.time() - _script_start
print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time / 60:.2f}m)")

wandb_run.finish()

if dist.is_initialized():
    dist.destroy_process_group()

if artifacts_log_f is not None:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    artifacts_log_f.close()
