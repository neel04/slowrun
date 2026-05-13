"""Train the Tiny Track Slowrun language model with JAX + Equinox.

This is the TPU-oriented rewrite of the Tiny Track PyTorch trainer. It keeps
the tiny-track defaults and model details from main: pre-norm GPT blocks,
half-truncated RoPE, long-layer key offsets, sliding-window causal attention,
U-Net skips, value embeddings, attention gates, BPB validation, functional
Muon/AdamW optimizer groups, EMA evaluation, and recency-weight checkpoint
averaging.

Usage:
    python tiny/train.py
    python tiny/train.py --input_bin fineweb_data/fineweb_train.pt --input_val_bin fineweb_data/fineweb_val.pt

On TPU multi-host jobs, launch the same command on every host and pass
--jax-distributed before any JAX computation is created.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

try:
    import torch
except Exception:  # Torch is only needed to read the existing .pt data files.
    torch = None

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import tiktoken
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

import wandb

_script_start = time.time()


# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT model with JAX/Equinox")
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--num-epochs", type=int, default=16)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Run name under runs/ (default: timestamp)",
)
parser.add_argument("--scalar-lr", type=float, default=0.25)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--embedding-lr", type=float, default=0.15)
parser.add_argument("--unembedding-lr", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=0.8)
parser.add_argument("--wd-phase1-epoch", type=int, default=2)
parser.add_argument("--wd-phase2-epoch", type=int, default=8)
parser.add_argument("--wd-mid", type=float, default=0.1)
parser.add_argument("--wd-end", type=float, default=1.25)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=16)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_embd", type=int, default=1024)
parser.add_argument("--lr_multiplier", type=float, default=0.8)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--update-ema-every", type=int, default=10)
parser.add_argument("--ema-decay-per-epoch", type=float, default=0.15)
parser.add_argument(
    "--dupe-start-epoch",
    type=int,
    default=1_000_000,
    help="Epoch to enable optional layer duplication",
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
    default=0.6,
    help="Override warmdown ratio",
)
parser.add_argument(
    "--logit-cap",
    type=float,
    default=15.0,
    help="Logit soft-capping value (0=disabled)",
)
parser.add_argument(
    "--logit-avg",
    type=int,
    default=0,
    help="Optional number of late checkpoints for probability averaging (0=disabled)",
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
    default=4,
    help="SWA: cosine-cycle LR in last N epochs for checkpoint diversity (0=off)",
)
parser.add_argument(
    "--stoch-depth",
    type=float,
    default=0.0,
    help="Stochastic depth max drop rate (linear schedule, 0=off)",
)
parser.add_argument(
    "--mtp-weight",
    type=float,
    default=0.0,
    help="Optional multi-token prediction weight (0=off)",
)
parser.add_argument(
    "--iha",
    action="store_true",
    default=False,
    help="Optionally enable Interleaved Head Attention (cross-head Q/K/V mixing)",
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
    "--sequence-length",
    type=int,
    default=2048,
    help="Static sequence length. Keep 2048 for benchmark runs.",
)
parser.add_argument(
    "--eval-tokens", type=int, default=10_000_000, help="Validation token budget"
)
parser.add_argument(
    "--compute-dtype",
    type=str,
    default="bfloat16",
    choices=["bfloat16", "float32"],
    help="Activation/matmul dtype. bfloat16 is the TPU path.",
)
parser.add_argument(
    "--jax-distributed",
    action="store_true",
    help="Call jax.distributed.initialize() at startup for TPU multi-host jobs",
)
parser.add_argument(
    "--compile-cache-dir",
    type=str,
    default="",
    help="Optional JAX persistent compilation cache directory",
)
parser.add_argument(
    "--log-compiles", action="store_true", help="Log JAX compilations/cache misses"
)
parser.add_argument(
    "--disable-wandb", action="store_true", help="Disable WandB even on process 0"
)
args = parser.parse_args()

if args.output_json and not args.save_result:
    args.save_result = args.output_json

if args.compile_cache_dir:
    jax.config.update("jax_compilation_cache_dir", args.compile_cache_dir)
if args.log_compiles:
    jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_explain_cache_misses", True)
if args.jax_distributed:
    jax.distributed.initialize()


# =============================================================================
# Hyperparameters
# =============================================================================

DEPTH = args.n_layer
N_EMBD = args.n_embd
N_HEAD = args.n_head
assert N_EMBD % N_HEAD == 0, "n_embd must be divisible by n_head"
HEAD_DIM = N_EMBD // N_HEAD
assert HEAD_DIM % 2 == 0, "RoPE requires an even head_dim"
assert HEAD_DIM % 4 == 0, "Tiny half-truncated RoPE requires head_dim divisible by 4"
MAX_SEQ_LEN = args.sequence_length
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
EVAL_TOKENS = args.eval_tokens
DATA_DIR = "fineweb_data"
BOS_ID = 50256
RUNS_DIR = "runs"

BASE_MATRIX_LR = args.matrix_lr
BASE_SCALAR_LR = args.scalar_lr
BASE_EMBEDDING_LR = args.embedding_lr
BASE_UNEMBEDDING_LR = args.unembedding_lr

_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
MATRIX_LR = BASE_MATRIX_LR * _lr_mult
UNEMBEDDING_LR = BASE_UNEMBEDDING_LR * _lr_mult
EMBEDDING_LR = BASE_EMBEDDING_LR * _lr_mult
SCALAR_LR = BASE_SCALAR_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = args.warmdown_ratio
FINAL_LR_FRAC = 0.0


# =============================================================================
# Utilities
# =============================================================================


def is_process0() -> bool:
    return jax.process_index() == 0


def print0(s: str = "", **kwargs: Any) -> None:
    if is_process0():
        print(s, **kwargs)


class DummyWandb:
    def __init__(self):
        self.summary = {}
        self.url = None

    def log(self, *a, **kw):
        pass

    def log_code(self, *a, **kw):
        pass

    def finish(self):
        pass


class TeeStream:
    """Save terminal output to file."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self.streams
        )

    def fileno(self):
        return self.streams[0].fileno()


def resolve_run_dir(run_name: str | None) -> tuple[str, str]:
    if run_name:
        return run_name, os.path.join(RUNS_DIR, run_name)
    name = time.strftime("%Y%m%d_%H%M%S")
    return name, os.path.join(RUNS_DIR, name)


def dtype_from_name(name: str):
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"unknown dtype {name}")


def rms_norm(
    x: Float[Array, "... channels"], eps: float = 1e-6
) -> Float[Array, "... channels"]:
    y = x.astype(jnp.float32)
    y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(
    x: Float[Array, "... head_dim"],
    cos: Float[Array, "... half_head_dim"],
    sin: Float[Array, "... half_head_dim"],
) -> Float[Array, "... head_dim"]:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return jnp.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)


def precompute_rotary(
    seq_len: int, head_dim: int, dtype: Any, base: int = 10000
) -> tuple[
    Float[Array, "1 seq 1 half_head_dim"],
    Float[Array, "1 seq 1 half_head_dim"],
]:
    rotary_pairs = head_dim // 4
    inv_freq = 1.0 / (
        base
        ** (
            jnp.arange(0, rotary_pairs * 2, 2, dtype=jnp.float32)
            / (rotary_pairs * 2)
        )
    )
    inv_freq = jnp.concatenate(
        [
            inv_freq,
            jnp.zeros((head_dim // 2 - rotary_pairs,), dtype=jnp.float32),
        ]
    )
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs).astype(dtype)[None, :, None, :], jnp.sin(freqs).astype(dtype)[
        None, :, None, :
    ]


def dropout(
    x: Float[Array, "..."], p: float, key: PRNGKeyArray, deterministic: bool
) -> Float[Array, "..."]:
    if deterministic or p <= 0:
        return x
    keep_prob = 1.0 - p
    keep = jax.random.bernoulli(key, keep_prob, x.shape)
    return jnp.where(keep, x / keep_prob, 0).astype(x.dtype)


def cross_entropy(
    logits: Float[Array, "... vocab"],
    targets: Int[Array, "..."],
    reduction: str = "mean",
) -> Float[Array, "..."]:
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    safe_targets = jnp.maximum(targets, 0)
    nll = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1)[..., 0]
    mask = targets != -1
    nll = jnp.where(mask, nll, 0.0)
    if reduction == "none":
        return nll
    denom = jnp.maximum(mask.sum(), 1)
    return nll.sum() / denom


def tree_sum(tree: Any) -> int:
    leaves = jtu.tree_leaves(tree)
    return int(sum(x.size for x in leaves if isinstance(x, jax.Array)))


def tree_zeros_like(tree: Any) -> Any:
    return jtu.tree_map(
        lambda x: jnp.zeros_like(x) if isinstance(x, jax.Array) else None, tree
    )


def tree_add(a: Any, b: Any) -> Any:
    return jtu.tree_map(
        lambda x, y: None if x is None else x + y,
        a,
        b,
        is_leaf=lambda x: x is None,
    )


def tree_div(a: Any, scale: float) -> Any:
    return jtu.tree_map(
        lambda x: None if x is None else x / scale, a, is_leaf=lambda x: x is None
    )


# =============================================================================
# Model
# =============================================================================


@dataclass(frozen=True)
class GPTConfig:
    sequence_len: int
    vocab_size: int
    padded_vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    window_pattern: str
    dropout: float
    stoch_depth: float
    use_iha: bool
    iha_mix_v: bool
    mtp_weight: float
    logit_cap: float
    compute_dtype: str


class Linear(eqx.Module):
    weight: Float[Array, "out_features in_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        *,
        init: str,
        scale: float = 1.0,
    ):
        shape = (out_features, in_features)
        if init == "uniform":
            self.weight = jax.random.uniform(
                key, shape, minval=-scale, maxval=scale, dtype=jnp.float32
            )
        elif init == "normal":
            self.weight = scale * jax.random.normal(key, shape, dtype=jnp.float32)
        elif init == "zeros":
            self.weight = jnp.zeros(shape, dtype=jnp.float32)
        else:
            raise ValueError(f"unknown linear init {init}")

    def __call__(
        self, x: Float[Array, "... in_features"], dtype: Any
    ) -> Float[Array, "... out_features"]:
        return jnp.matmul(x.astype(dtype), self.weight.astype(dtype).T)


class Embedding(eqx.Module):
    weight: Float[Array, "num_embeddings embedding_dim"]

    def __init__(self, num_embeddings: int, embedding_dim: int, key: PRNGKeyArray):
        self.weight = jax.random.normal(
            key, (num_embeddings, embedding_dim), dtype=jnp.float32
        )

    def __call__(
        self, idx: Int[Array, "..."], dtype: Any
    ) -> Float[Array, "... embedding_dim"]:
        return jnp.take(self.weight, idx, axis=0).astype(dtype)


class CausalSelfAttention(eqx.Module):
    c_q: Linear
    c_k: Linear
    c_v: Linear
    c_proj: Linear
    ve_gate: Linear | None
    attn_gate: Linear
    q_mix: Float[Array, "head_out head_in"] | None
    k_mix: Float[Array, "head_out head_in"] | None
    v_mix: Float[Array, "head_out head_in"] | None
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    n_embd: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    use_key_offset: bool = eqx.field(static=True)
    ve_gate_channels: int = eqx.field(static=True, default=32)
    attn_gate_channels: int = eqx.field(static=True, default=12)

    def __init__(self, config: GPTConfig, layer_idx: int, key: PRNGKeyArray):
        keys = jax.random.split(key, 9)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        pattern = config.window_pattern.upper()
        char = pattern[layer_idx % len(pattern)]
        self.use_key_offset = char == "L" or layer_idx == config.n_layer - 1
        s = math.sqrt(3.0) * config.n_embd**-0.5
        self.c_q = Linear(
            config.n_embd, self.n_head * self.head_dim, keys[0], init="uniform", scale=s
        )
        self.c_k = Linear(
            config.n_embd,
            self.n_kv_head * self.head_dim,
            keys[1],
            init="uniform",
            scale=s,
        )
        self.c_v = Linear(
            config.n_embd,
            self.n_kv_head * self.head_dim,
            keys[2],
            init="uniform",
            scale=s,
        )
        self.c_proj = Linear(config.n_embd, config.n_embd, keys[3], init="zeros")
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, keys[4], init="zeros")
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        self.attn_gate = Linear(
            self.attn_gate_channels, self.n_head, keys[5], init="zeros"
        )
        if config.use_iha:
            self.q_mix = jnp.eye(self.n_head, dtype=jnp.float32)
            self.k_mix = jnp.eye(self.n_kv_head, dtype=jnp.float32)
            self.v_mix = (
                jnp.eye(self.n_kv_head, dtype=jnp.float32) if config.iha_mix_v else None
            )
        else:
            self.q_mix = None
            self.k_mix = None
            self.v_mix = None

    def _fuse_mix(
        self,
        weight: Float[Array, "heads_x_dim in_dim"],
        mix: Float[Array, "heads heads"],
        heads: int,
    ) -> Float[Array, "heads_x_dim in_dim"]:
        w = weight.reshape(heads, self.head_dim, -1)
        return jnp.einsum("hm,mdc->hdc", mix, w).reshape(weight.shape)

    def __call__(
        self,
        x: Float[Array, "batch seq channels"],
        ve: Float[Array, "batch seq kv_channels"] | None,
        cos_sin: tuple[
            Float[Array, "1 seq 1 half_head_dim"],
            Float[Array, "1 seq 1 half_head_dim"],
        ],
        window_size: int,
        key: PRNGKeyArray,
        deterministic: bool,
        config: GPTConfig,
    ) -> Float[Array, "batch seq channels"]:
        dtype = dtype_from_name(config.compute_dtype)
        B, T, _ = x.shape
        if self.q_mix is not None:
            q_mix = self.q_mix
            k_mix = self.k_mix
            assert k_mix is not None
            q_weight = self._fuse_mix(self.c_q.weight, q_mix, self.n_head)
            k_weight = self._fuse_mix(self.c_k.weight, k_mix, self.n_kv_head)
            q = jnp.matmul(x.astype(dtype), q_weight.astype(dtype).T).reshape(
                B, T, self.n_head, self.head_dim
            )
            k = jnp.matmul(x.astype(dtype), k_weight.astype(dtype).T).reshape(
                B, T, self.n_kv_head, self.head_dim
            )
            v_mix = self.v_mix
            if v_mix is not None:
                v_weight = self._fuse_mix(self.c_v.weight, v_mix, self.n_kv_head)
                v = jnp.matmul(x.astype(dtype), v_weight.astype(dtype).T).reshape(
                    B, T, self.n_kv_head, self.head_dim
                )
            else:
                v = self.c_v(x, dtype).reshape(B, T, self.n_kv_head, self.head_dim)
        else:
            q = self.c_q(x, dtype).reshape(B, T, self.n_head, self.head_dim)
            k = self.c_k(x, dtype).reshape(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x, dtype).reshape(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve_gate = self.ve_gate
            assert ve_gate is not None
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * jax.nn.sigmoid(ve_gate(x[..., : self.ve_gate_channels], dtype))
            v = v + gate[..., None].astype(dtype) * ve

        cos, sin = cos_sin
        q = rms_norm(apply_rotary_emb(q, cos, sin))
        k = rms_norm(apply_rotary_emb(k, cos, sin))
        if self.use_key_offset:
            stationary = jnp.concatenate(
                [k[:, :1, :, self.head_dim // 2 :], k[:, :-1, :, self.head_dim // 2 :]],
                axis=1,
            )
            k = k.at[:, :, :, self.head_dim // 2 :].set(stationary)
        local_window = (
            None if window_size < 0 or window_size >= T - 1 else (window_size, 0)
        )
        y = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            local_window_size=local_window,
        )
        gate = jax.nn.sigmoid(self.attn_gate(x[..., : self.attn_gate_channels], dtype))
        y = y * gate[..., None].astype(dtype)
        y = y.reshape(B, T, -1)
        y = self.c_proj(y, dtype)
        return dropout(y, config.dropout, key, deterministic)


class MLP(eqx.Module):
    c_gate: Linear
    c_fc: Linear
    c_proj: Linear

    def __init__(self, config: GPTConfig, key: PRNGKeyArray):
        k1, k2, k3 = jax.random.split(key, 3)
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        s = math.sqrt(3.0) * config.n_embd**-0.5
        self.c_gate = Linear(config.n_embd, hidden, k1, init="uniform", scale=s)
        self.c_fc = Linear(config.n_embd, hidden, k2, init="uniform", scale=s)
        self.c_proj = Linear(hidden, config.n_embd, k3, init="zeros")

    def __call__(
        self,
        x: Float[Array, "batch seq channels"],
        key: PRNGKeyArray,
        deterministic: bool,
        config: GPTConfig,
    ) -> Float[Array, "batch seq channels"]:
        dtype = dtype_from_name(config.compute_dtype)
        y = jax.nn.silu(self.c_gate(x, dtype)) * self.c_fc(x, dtype)
        return self.c_proj(y, dtype)


class Block(eqx.Module):
    attn: CausalSelfAttention
    mlp: MLP
    drop_prob: float = eqx.field(static=True)

    def __init__(self, config: GPTConfig, layer_idx: int, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key, 2)
        self.attn = CausalSelfAttention(config, layer_idx, k1)
        self.mlp = MLP(config, k2)
        self.drop_prob = config.stoch_depth * (layer_idx / max(config.n_layer - 1, 1))

    def __call__(
        self,
        x: Float[Array, "batch seq channels"],
        ve: Float[Array, "batch seq kv_channels"] | None,
        cos_sin: tuple[
            Float[Array, "1 seq 1 half_head_dim"],
            Float[Array, "1 seq 1 half_head_dim"],
        ],
        window_size: int,
        key: PRNGKeyArray,
        deterministic: bool,
        config: GPTConfig,
    ) -> Float[Array, "batch seq channels"]:
        k_attn, k_mlp, k_depth = jax.random.split(key, 3)
        x_in = x
        x = x + self.attn(
            rms_norm(x), ve, cos_sin, window_size, k_attn, deterministic, config
        )
        x = x + self.mlp(rms_norm(x), k_mlp, deterministic, config)
        if not deterministic and self.drop_prob > 0:
            keep = (jax.random.uniform(k_depth, ()) >= self.drop_prob).astype(x.dtype)
            x = x_in + keep * (x - x_in)
        return x


class GPT(eqx.Module):
    config: GPTConfig = eqx.field(static=True)
    window_sizes: tuple[int, ...] = eqx.field(static=True)
    encoder_layers: int = eqx.field(static=True)
    wte: Embedding
    blocks: tuple[Block, ...]
    lm_head: Linear
    resid_lambdas: Float[Array, " n_layer"]
    x0_lambdas: Float[Array, " n_layer"]
    ve_projs: tuple[Linear | None, ...]
    skip_weights: Float[Array, " encoder_layers"]
    mtp_proj: Linear | None
    mtp_block: Block | None

    def __init__(self, config: GPTConfig, key: PRNGKeyArray):
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.encoder_layers = config.n_layer // 2
        num_extra = (
            4 + config.n_layer + config.n_layer + (2 if config.mtp_weight > 0 else 0)
        )
        keys = iter(jax.random.split(key, num_extra))
        self.wte = Embedding(config.padded_vocab_size, config.n_embd, next(keys))
        self.blocks = tuple(Block(config, i, next(keys)) for i in range(config.n_layer))
        self.lm_head = Linear(
            config.n_embd,
            config.padded_vocab_size,
            next(keys),
            init="normal",
            scale=0.001,
        )
        self.resid_lambdas = jnp.full((config.n_layer,), 1.1, dtype=jnp.float32)
        self.x0_lambdas = jnp.full((config.n_layer,), 0.1, dtype=jnp.float32)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        s = math.sqrt(3.0) * config.n_embd**-0.5
        self.ve_projs = tuple(
            Linear(config.n_embd, kv_dim, next(keys), init="uniform", scale=s)
            if has_ve(i, config.n_layer)
            else None
            for i in range(config.n_layer)
        )
        self.skip_weights = jnp.ones((self.encoder_layers,), dtype=jnp.float32)
        if config.mtp_weight > 0:
            self.mtp_proj = Linear(
                2 * config.n_embd, config.n_embd, next(keys), init="uniform", scale=s
            )
            self.mtp_block = Block(config, config.n_layer, next(keys))
        else:
            self.mtp_proj = None
            self.mtp_block = None

    def _compute_window_sizes(self, config: GPTConfig) -> tuple[int, ...]:
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": long_w, "S": short_w}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = long_w
        return tuple(sizes)

    def _next_key(self, key: PRNGKeyArray) -> tuple[PRNGKeyArray, PRNGKeyArray]:
        key, subkey = jax.random.split(key)
        return key, subkey

    def _run_decoder_layers(
        self,
        x: Float[Array, "batch seq channels"],
        x0: Float[Array, "batch seq channels"],
        encoder_outputs: list[Float[Array, "batch seq channels"]],
        cos_sin: tuple[
            Float[Array, "1 seq 1 half_head_dim"],
            Float[Array, "1 seq 1 half_head_dim"],
        ],
        start: int,
        end: int,
        key: PRNGKeyArray,
        deterministic: bool,
    ) -> tuple[Float[Array, "batch seq channels"], PRNGKeyArray]:
        dtype = dtype_from_name(self.config.compute_dtype)
        for i in range(start, end):
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = (
                    x
                    + self.skip_weights[i - self.encoder_layers].astype(dtype)
                    * encoder_outputs[j]
                )
            x = (
                self.resid_lambdas[i].astype(dtype) * x
                + self.x0_lambdas[i].astype(dtype) * x0
            )
            ve_proj = self.ve_projs[i]
            ve = ve_proj(x0, dtype) if ve_proj is not None else None
            key, subkey = self._next_key(key)
            x = self.blocks[i](
                x, ve, cos_sin, self.window_sizes[i], subkey, deterministic, self.config
            )
        return x, key

    def __call__(
        self,
        idx: Int[Array, "batch seq"],
        targets: Int[Array, "batch seq"] | None = None,
        *,
        loss_reduction: str = "mean",
        key: PRNGKeyArray,
        deterministic: bool,
        dupe_enabled: bool,
        dupe_start: int,
        dupe_end: int,
        dupe_loops: int,
    ) -> Any:
        dtype = dtype_from_name(self.config.compute_dtype)
        _, T = idx.shape
        cos_sin = precompute_rotary(T, self.config.n_embd // self.config.n_head, dtype)
        x = rms_norm(self.wte(idx, dtype))
        x0 = x

        encoder_outputs: list[Float[Array, "batch seq channels"]] = []
        for i in range(self.encoder_layers):
            x = (
                self.resid_lambdas[i].astype(dtype) * x
                + self.x0_lambdas[i].astype(dtype) * x0
            )
            ve_proj = self.ve_projs[i]
            ve = ve_proj(x0, dtype) if ve_proj is not None else None
            key, subkey = self._next_key(key)
            x = self.blocks[i](
                x, ve, cos_sin, self.window_sizes[i], subkey, deterministic, self.config
            )
            encoder_outputs.append(x)

        if not dupe_enabled:
            x, key = self._run_decoder_layers(
                x,
                x0,
                encoder_outputs,
                cos_sin,
                self.encoder_layers,
                self.config.n_layer,
                key,
                deterministic,
            )
        else:
            x, key = self._run_decoder_layers(
                x,
                x0,
                encoder_outputs,
                cos_sin,
                self.encoder_layers,
                dupe_end,
                key,
                deterministic,
            )
            for _ in range(dupe_loops):
                x, key = self._run_decoder_layers(
                    x,
                    x0,
                    encoder_outputs,
                    cos_sin,
                    dupe_start,
                    dupe_end,
                    key,
                    deterministic,
                )
            x, key = self._run_decoder_layers(
                x,
                x0,
                encoder_outputs,
                cos_sin,
                dupe_end,
                self.config.n_layer,
                key,
                deterministic,
            )

        x = rms_norm(x)
        logits = self.lm_head(x, dtype)[..., : self.config.vocab_size].astype(
            jnp.float32
        )
        if self.config.logit_cap > 0:
            logits = self.config.logit_cap * jnp.tanh(logits / self.config.logit_cap)
        if targets is None:
            return logits

        lm_loss = cross_entropy(logits, targets, reduction=loss_reduction)
        if loss_reduction != "mean":
            return lm_loss
        if self.config.mtp_weight <= 0:
            zero = jnp.zeros((), dtype=jnp.float32)
            return lm_loss, {"lm_loss": lm_loss, "mtp_loss": zero}

        mtp_emb = rms_norm(self.wte(jnp.maximum(targets[:, :-1], 0), dtype))
        mtp_proj = self.mtp_proj
        mtp_block = self.mtp_block
        assert mtp_proj is not None
        assert mtp_block is not None
        combined = mtp_proj(jnp.concatenate([x[:, :-1], mtp_emb], axis=-1), dtype)
        mT = combined.shape[1]
        mtp_cos_sin = precompute_rotary(
            mT, self.config.n_embd // self.config.n_head, dtype
        )
        key, subkey = self._next_key(key)
        mtp_out = rms_norm(
            mtp_block(combined, None, mtp_cos_sin, -1, subkey, deterministic, self.config)
        )
        mtp_logits = self.lm_head(mtp_out, dtype)[..., : self.config.vocab_size].astype(
            jnp.float32
        )
        if self.config.logit_cap > 0:
            mtp_logits = self.config.logit_cap * jnp.tanh(
                mtp_logits / self.config.logit_cap
            )
        mtp_loss = cross_entropy(mtp_logits, targets[:, 1:], reduction="mean")
        loss = lm_loss + self.config.mtp_weight * mtp_loss
        return loss, {"lm_loss": lm_loss, "mtp_loss": mtp_loss}

    def _avg_causal_attended_keys(self, window: int, seq_len: int) -> float:
        if window < 0 or window >= seq_len - 1:
            return (seq_len + 1) / 2
        max_keys = min(window + 1, seq_len)
        return max_keys - max_keys * (max_keys - 1) / (2 * seq_len)

    def estimate_flops(self, nparams: int) -> float:
        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        attn_flops = sum(
            12 * h * q * self._avg_causal_attended_keys(w, t) for w in self.window_sizes
        )
        return 6 * nparams + attn_flops


# =============================================================================
# Optimizer: functional MuonAdamW
# =============================================================================

polar_express_coeffs = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


@dataclass(frozen=True)
class ParamSpec:
    kind: str
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    ns_steps: int = 5


@dataclass(frozen=True)
class LeafUpdate:
    param: Float[Array, "..."] | None
    adam_m: Float[Array, "..."] | None
    adam_v: Float[Array, "..."] | None
    muon_m: Float[Array, "..."] | None
    muon_v: Float[Array, "..."] | None


class OptimizerState(NamedTuple):
    step: Int[Array, ""]
    adam_m: PyTree[Float[Array, "..."] | None]
    adam_v: PyTree[Float[Array, "..."] | None]
    muon_m: PyTree[Float[Array, "..."] | None]
    muon_v: PyTree[Float[Array, "..."] | None]


class TrainState(NamedTuple):
    params: PyTree[Float[Array, "..."]]
    opt_state: OptimizerState
    key: PRNGKeyArray
    step: Int[Array, ""]


def _path_to_str(path: tuple[Any, ...]) -> str:
    parts = []
    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jtu.SequenceKey):
            parts.append(str(key.idx))
        elif isinstance(key, jtu.DictKey):
            parts.append(str(key.key))
        else:
            parts.append(str(key))
    return ".".join(parts)


def build_optimizer_spec(
    params: PyTree[Float[Array, "..."]],
) -> PyTree[ParamSpec | None]:
    def make_spec(path: tuple[Any, ...], p: Any):
        if p is None:
            return None
        name = _path_to_str(path)
        if name.endswith("lm_head.weight"):
            return ParamSpec(
                "adamw",
                UNEMBEDDING_LR,
                ADAM_BETAS[0],
                ADAM_BETAS[1],
                1e-10,
                WEIGHT_DECAY,
            )
        if name.endswith("wte.weight"):
            return ParamSpec(
                "adamw", EMBEDDING_LR, ADAM_BETAS[0], ADAM_BETAS[1], 1e-10, WEIGHT_DECAY
            )
        if name.endswith("resid_lambdas"):
            return ParamSpec(
                "adamw", SCALAR_LR * 0.01, ADAM_BETAS[0], ADAM_BETAS[1], 1e-10, 0.0
            )
        if name.endswith("x0_lambdas"):
            return ParamSpec("adamw", SCALAR_LR, 0.96, 0.95, 1e-10, 0.0)
        if name.endswith("skip_weights"):
            return ParamSpec(
                "adamw", SCALAR_LR * 0.01, ADAM_BETAS[0], ADAM_BETAS[1], 1e-10, 0.0
            )
        if name.endswith("attn_gate.weight"):
            return ParamSpec("adamw", SCALAR_LR, 0.9, 0.99, 1e-10, 0.0)
        if name.endswith(("q_mix", "k_mix", "v_mix")):
            return ParamSpec(
                "adamw", args.iha_lr, ADAM_BETAS[0], ADAM_BETAS[1], 1e-10, 0.0
            )
        if getattr(p, "ndim", 0) == 2:
            return ParamSpec("muon", MATRIX_LR, 0.95, 0.95, 1e-10, WEIGHT_DECAY)
        return ParamSpec("adamw", SCALAR_LR, ADAM_BETAS[0], ADAM_BETAS[1], 1e-10, 0.0)

    return jtu.tree_map_with_path(make_spec, params, is_leaf=lambda x: x is None)


def init_optimizer_state(
    params: PyTree[Float[Array, "..."]], specs: PyTree[ParamSpec | None]
) -> OptimizerState:
    def adam_state(p, spec):
        if p is None or spec is None or spec.kind != "adamw":
            return None
        return jnp.zeros_like(p)

    def muon_m_state(p, spec):
        if p is None or spec is None or spec.kind != "muon":
            return None
        return jnp.zeros_like(p)

    def muon_v_state(p, spec):
        if p is None or spec is None or spec.kind != "muon":
            return None
        rows, cols = p.shape[-2], p.shape[-1]
        shape = (rows, 1) if rows >= cols else (1, cols)
        return jnp.zeros(shape, dtype=jnp.float32)

    def is_leaf(x):
        return x is None or isinstance(x, ParamSpec)

    return OptimizerState(
        step=jnp.asarray(0, dtype=jnp.int32),
        adam_m=jtu.tree_map(adam_state, params, specs, is_leaf=is_leaf),
        adam_v=jtu.tree_map(adam_state, params, specs, is_leaf=is_leaf),
        muon_m=jtu.tree_map(muon_m_state, params, specs, is_leaf=is_leaf),
        muon_v=jtu.tree_map(muon_v_state, params, specs, is_leaf=is_leaf),
    )


def _adamw_update(
    p: Float[Array, "..."],
    g: Float[Array, "..."],
    m: Float[Array, "..."],
    v: Float[Array, "..."],
    step: Int[Array, ""],
    spec: ParamSpec,
    lr_mult: Float[Array, ""],
    wd_mult: Float[Array, ""],
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    lr = jnp.asarray(spec.lr, jnp.float32) * lr_mult
    wd = jnp.asarray(spec.weight_decay, jnp.float32) * wd_mult
    beta1 = jnp.asarray(spec.beta1, jnp.float32)
    beta2 = jnp.asarray(spec.beta2, jnp.float32)
    g = g.astype(jnp.float32)
    p32 = p.astype(jnp.float32) * (1.0 - lr * wd)
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * jnp.square(g)
    bias1 = 1.0 - beta1 ** step.astype(jnp.float32)
    bias2 = 1.0 - beta2 ** step.astype(jnp.float32)
    p32 = p32 - (lr / bias1) * m / (jnp.sqrt(v / bias2) + spec.eps)
    return p32.astype(p.dtype), m, v


def _muon_update(
    p: Float[Array, "..."],
    g: Float[Array, "..."],
    momentum_buffer: Float[Array, "..."],
    second_momentum_buffer: Float[Array, "..."],
    spec: ParamSpec,
    lr_mult: Float[Array, ""],
    wd_mult: Float[Array, ""],
    muon_momentum: Float[Array, ""],
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    g = g.astype(jnp.float32)
    momentum = muon_momentum.astype(jnp.float32)
    momentum_buffer = momentum * momentum_buffer + (1.0 - momentum) * g
    g = (1.0 - momentum) * g + momentum * momentum_buffer
    g = g / jnp.maximum(jnp.linalg.norm(g, axis=-1, keepdims=True), 1e-7)
    X = g.astype(jnp.bfloat16)
    X = X / (
        jnp.linalg.norm(X.astype(jnp.float32), axis=(-2, -1), keepdims=True).astype(
            X.dtype
        )
        * 1.02
        + 1e-6
    )

    if g.shape[-2] > g.shape[-1]:
        for a, b, c in polar_express_coeffs[: spec.ns_steps]:
            A = jnp.swapaxes(X, -1, -2) @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[: spec.ns_steps]:
            A = X @ jnp.swapaxes(X, -1, -2)
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X.astype(jnp.float32)

    red_dim = -1 if g.shape[-2] >= g.shape[-1] else -2
    red_dim_size = g.shape[red_dim]
    v_mean = jnp.mean(jnp.square(g), axis=red_dim, keepdims=True)
    v_norm = jnp.sqrt(jnp.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size)
    beta2 = jnp.asarray(spec.beta2, jnp.float32)
    second_momentum_buffer = beta2 * second_momentum_buffer + (1.0 - beta2) * v_mean
    step_size = jax.lax.rsqrt(jnp.maximum(second_momentum_buffer, 1e-10))
    scaled_sq_sum = (v_mean * red_dim_size) * jnp.square(step_size)
    v_norm_new = jnp.sqrt(jnp.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))
    g = g * final_scale

    lr = jnp.asarray(spec.lr, jnp.float32) * lr_mult
    lr = lr * math.sqrt(max(1.0, p.shape[-2] / p.shape[-1]))
    wd = jnp.asarray(spec.weight_decay, jnp.float32) * wd_mult
    p32 = p.astype(jnp.float32)
    mask = (g * p32) >= 0
    p32 = p32 - (lr * g + lr * wd * p32 * mask)
    return p32.astype(p.dtype), momentum_buffer, second_momentum_buffer


def optimizer_update(
    params: PyTree[Float[Array, "..."]],
    grads: PyTree[Float[Array, "..."] | None],
    state: OptimizerState,
    specs: PyTree[ParamSpec | None],
    lr_mult: Float[Array, ""],
    wd_mult: Float[Array, ""],
    muon_momentum: Float[Array, ""],
) -> tuple[PyTree[Float[Array, "..."]], OptimizerState]:
    step = state.step + 1

    def update_one(p, g, am, av, mm, mv, spec):
        if p is None or spec is None:
            return LeafUpdate(p, am, av, mm, mv)
        g = jnp.nan_to_num(g.astype(jnp.float32))
        if spec.kind == "adamw":
            p, am, av = _adamw_update(p, g, am, av, step, spec, lr_mult, wd_mult)
            return LeafUpdate(p, am, av, mm, mv)
        p, mm, mv = _muon_update(p, g, mm, mv, spec, lr_mult, wd_mult, muon_momentum)
        return LeafUpdate(p, am, av, mm, mv)

    def is_leaf(x):
        return x is None or isinstance(x, ParamSpec)

    updates = jtu.tree_map(
        update_one,
        params,
        grads,
        state.adam_m,
        state.adam_v,
        state.muon_m,
        state.muon_v,
        specs,
        is_leaf=is_leaf,
    )

    def upd_leaf(x):
        return isinstance(x, LeafUpdate)

    new_params = jtu.tree_map(lambda u: u.param, updates, is_leaf=upd_leaf)
    new_state = OptimizerState(
        step=step,
        adam_m=jtu.tree_map(lambda u: u.adam_m, updates, is_leaf=upd_leaf),
        adam_v=jtu.tree_map(lambda u: u.adam_v, updates, is_leaf=upd_leaf),
        muon_m=jtu.tree_map(lambda u: u.muon_m, updates, is_leaf=upd_leaf),
        muon_v=jtu.tree_map(lambda u: u.muon_v, updates, is_leaf=upd_leaf),
    )
    return new_params, new_state


# =============================================================================
# Dataloader: BOS-aligned best-fit packing
# =============================================================================


def load_token_file(filepath: str) -> dict[str, Any]:
    if filepath.endswith(".npz"):
        data = np.load(filepath)
        return {
            "tokens": np.asarray(data["tokens"]),
            "doc_starts": np.asarray(data["doc_starts"]),
            "bos_id": int(data["bos_id"]),
            "seq_shuffle_seed": int(data["seq_shuffle_seed"]),
        }
    if torch is None:
        raise RuntimeError(
            "Torch is required to read .pt data files. Use .npz data or install torch."
        )
    data = torch.load(filepath, map_location="cpu", weights_only=True)
    return {
        "tokens": data["tokens"].cpu().numpy(),
        "doc_starts": data["doc_starts"].cpu().numpy(),
        "bos_id": int(data["bos_id"]),
        "seq_shuffle_seed": int(data["seq_shuffle_seed"]),
    }


class DataLoader:
    """Loads flat tokens, chunks into fixed sequences, then shards by JAX process."""

    def __init__(
        self,
        filepath: str,
        per_process_batch: int,
        T: int,
        *,
        doc_shuffle: bool = False,
    ):
        data = load_token_file(filepath)
        all_tokens = np.asarray(data["tokens"], dtype=np.int32)
        raw_doc_starts = np.asarray(data["doc_starts"], dtype=np.int64)
        bos_id = int(data["bos_id"])
        assert bos_id == BOS_ID, f"data bos_id {bos_id} != expected {BOS_ID}"

        doc_ends = np.concatenate(
            [raw_doc_starts[1:], np.asarray([all_tokens.size], dtype=np.int64)]
        )
        self.doc_tokens = [
            all_tokens[s:e] for s, e in zip(raw_doc_starts.tolist(), doc_ends.tolist())
        ]
        self.default_shuffle_seed = int(data["seq_shuffle_seed"])
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()
        self.B = per_process_batch
        self.T = T
        self.seq_size = T + 1
        self.doc_shuffle = doc_shuffle
        self.epoch = 1
        self._build_batches()

    def _build_batches(self) -> None:
        tokens = np.concatenate(self.doc_tokens)
        num_seqs = len(tokens) // self.seq_size
        all_seqs = tokens[: num_seqs * self.seq_size].reshape(num_seqs, self.seq_size)
        if self.doc_shuffle:
            rng = np.random.RandomState(self.epoch + 1000)
            all_seqs = all_seqs[rng.permutation(num_seqs)]
        else:
            perm = np.random.RandomState(self.default_shuffle_seed).permutation(
                num_seqs
            )
            all_seqs = all_seqs[perm]
        seqs_per_step = self.B * self.process_count
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        if usable == 0:
            raise ValueError(
                f"{self.B=} and process_count={self.process_count} leave no full batches in {num_seqs} sequences"
            )
        all_seqs = all_seqs[:usable].reshape(
            num_steps, self.process_count, self.B, self.seq_size
        )
        self.rank_data = np.ascontiguousarray(all_seqs[:, self.process_index])
        self.num_steps = num_steps
        self.total_tokens = usable * self.T
        self.pos = 0

    def __iter__(self):
        return self

    def _next_epoch(self) -> None:
        self.epoch += 1
        print0(f"Starting epoch {self.epoch}")
        if self.doc_shuffle:
            rng = np.random.RandomState(self.epoch)
            perm = rng.permutation(len(self.doc_tokens))
            self.doc_tokens = [self.doc_tokens[i] for i in perm.tolist()]
            self._build_batches()
        else:
            self.pos = 0
            rng = np.random.RandomState(self.epoch)
            self.rank_data = np.ascontiguousarray(
                self.rank_data[rng.permutation(self.num_steps)]
            )

    def __next__(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.pos >= self.num_steps:
            self._next_epoch()
        batch = self.rank_data[self.pos]
        self.pos += 1
        return (
            np.ascontiguousarray(batch[:, :-1], dtype=np.int32),
            np.ascontiguousarray(batch[:, 1:], dtype=np.int32),
            self.epoch,
        )


def next_microbatches(
    loader: DataLoader, grad_accum_steps: int
) -> tuple[np.ndarray, np.ndarray, int]:
    xs, ys = [], []
    epoch = loader.epoch
    for _ in range(grad_accum_steps):
        x, y, epoch = next(loader)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys), epoch


# =============================================================================
# JIT step factories
# =============================================================================


def make_train_step(
    model_static: GPT,
    opt_specs: PyTree[ParamSpec | None],
    grad_accum_steps: int,
    microbatch_sharding: NamedSharding,
    dupe_enabled: bool,
):
    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(
        params: PyTree[Float[Array, "..."]],
        x: Int[Array, "batch seq"],
        y: Int[Array, "batch seq"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        model = eqx.combine(params, model_static)
        loss, metrics = model(
            x,
            y,
            key=key,
            deterministic=False,
            dupe_enabled=dupe_enabled,
            dupe_start=args.dupe_layers_start,
            dupe_end=args.dupe_layers_end,
            dupe_loops=args.dupe_loops,
        )
        return loss, metrics

    @eqx.filter_jit(donate="all")
    def train_step(
        state: TrainState,
        xs: Int[Array, "microbatch batch seq"],
        ys: Int[Array, "microbatch batch seq"],
        lr_mult: Float[Array, ""],
        wd_mult: Float[Array, ""],
        muon_momentum: Float[Array, ""],
    ) -> tuple[TrainState, dict[str, Float[Array, ""]]]:
        xs = jax.lax.with_sharding_constraint(xs, microbatch_sharding)
        ys = jax.lax.with_sharding_constraint(ys, microbatch_sharding)
        zero_grads = tree_zeros_like(state.params)
        zero = jnp.asarray(0.0, dtype=jnp.float32)

        def body(carry, batch):
            grad_accum, total_loss, total_lm, total_mtp, key = carry
            x, y = batch
            key, subkey = jax.random.split(key)
            (loss, metrics), grads = loss_fn(state.params, x, y, subkey)
            grad_accum = tree_add(grad_accum, grads)
            return (
                grad_accum,
                total_loss + loss,
                total_lm + metrics["lm_loss"],
                total_mtp + metrics["mtp_loss"],
                key,
            ), None

        (grads, total_loss, total_lm, total_mtp, key), _ = jax.lax.scan(
            body,
            (zero_grads, zero, zero, zero, state.key),
            (xs, ys),
            length=grad_accum_steps,
        )
        grads = tree_div(grads, float(grad_accum_steps))
        params, opt_state = optimizer_update(
            state.params,
            grads,
            state.opt_state,
            opt_specs,
            lr_mult.astype(jnp.float32),
            wd_mult.astype(jnp.float32),
            muon_momentum.astype(jnp.float32),
        )
        new_state = TrainState(params, opt_state, key, state.step + 1)
        metrics = {
            "loss": total_loss / grad_accum_steps,
            "lm_loss": total_lm / grad_accum_steps,
            "mtp_loss": total_mtp / grad_accum_steps,
        }
        return new_state, metrics

    return train_step


def make_eval_step(model_static: GPT, batch_sharding: NamedSharding, dupe_enabled: bool):
    @eqx.filter_jit
    def eval_step(
        params: PyTree[Float[Array, "..."]],
        x: Int[Array, "batch seq"],
        y: Int[Array, "batch seq"],
        token_bytes: Int[Array, " vocab"],
    ) -> tuple[Float[Array, ""], Int[Array, ""], Float[Array, ""], Int[Array, ""]]:
        x = jax.lax.with_sharding_constraint(x, batch_sharding)
        y = jax.lax.with_sharding_constraint(y, batch_sharding)
        model = eqx.combine(params, model_static)
        loss2d = model(
            x,
            y,
            loss_reduction="none",
            key=jax.random.key(0),
            deterministic=True,
            dupe_enabled=dupe_enabled,
            dupe_start=args.dupe_layers_start,
            dupe_end=args.dupe_layers_end,
            dupe_loops=args.dupe_loops,
        )
        mask = y != -1
        num_bytes = jnp.take(token_bytes, jnp.maximum(y, 0))
        total_loss = jnp.sum(jnp.where(mask, loss2d, 0.0), dtype=jnp.float32)
        total_tokens = jnp.sum(mask, dtype=jnp.int32)
        total_nats = jnp.sum(loss2d * (num_bytes > 0), dtype=jnp.float32)
        total_bytes = jnp.sum(num_bytes, dtype=jnp.int32)
        return total_nats, total_bytes, total_loss, total_tokens

    return eval_step


def make_target_prob_step(
    model_static: GPT, batch_sharding: NamedSharding, dupe_enabled: bool
):
    @eqx.filter_jit
    def target_prob_step(
        params: PyTree[Float[Array, "..."]],
        x: Int[Array, "batch seq"],
        y: Int[Array, "batch seq"],
    ) -> Float[Array, " tokens"]:
        x = jax.lax.with_sharding_constraint(x, batch_sharding)
        y = jax.lax.with_sharding_constraint(y, batch_sharding)
        model = eqx.combine(params, model_static)
        logits = model(
            x,
            key=jax.random.key(0),
            deterministic=True,
            dupe_enabled=dupe_enabled,
            dupe_start=args.dupe_layers_start,
            dupe_end=args.dupe_layers_end,
            dupe_loops=args.dupe_loops,
        )
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y = y.reshape(-1)
        probs = jax.nn.softmax(flat_logits.astype(jnp.float32), axis=-1)
        return probs[jnp.arange(flat_y.size), jnp.maximum(flat_y, 0)]

    return target_prob_step


# =============================================================================
# Training/evaluation helpers
# =============================================================================


def make_global_array(
    local: np.ndarray, sharding: NamedSharding, global_shape: tuple[int, ...]
) -> Array:
    return jax.make_array_from_process_local_data(
        sharding, local, global_shape=global_shape
    )


def evaluate_bpb(
    params: PyTree[Float[Array, "..."]],
    build_val_loader,
    steps: int,
    token_bytes: Int[Array, " vocab"],
    eval_step,
    batch_sharding: NamedSharding,
    global_batch_size: int,
) -> tuple[float, float]:
    val_loader = build_val_loader()
    totals: (
        tuple[Float[Array, ""], Int[Array, ""], Float[Array, ""], Int[Array, ""]]
        | None
    ) = None
    for _ in range(steps):
        x, y, _ = next(val_loader)
        xg = make_global_array(x, batch_sharding, (global_batch_size, MAX_SEQ_LEN))
        yg = make_global_array(y, batch_sharding, (global_batch_size, MAX_SEQ_LEN))
        out = eval_step(params, xg, yg, token_bytes)
        totals = out if totals is None else tuple(a + b for a, b in zip(totals, out))
    assert totals is not None
    total_nats, total_bytes, total_loss, total_tokens = [
        np.asarray(jax.device_get(x)) for x in totals
    ]
    total_nats = float(total_nats)
    total_bytes = int(total_bytes)
    total_loss = float(total_loss)
    total_tokens = int(total_tokens)
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float("inf")
    loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return bpb, loss


def evaluate_bpb_logit_avg(
    model_static: GPT,
    params_skeleton: PyTree[Float[Array, "..."]],
    ckpt_paths: list[str],
    weights: list[float],
    build_val_loader,
    steps: int,
    token_bytes: Int[Array, " vocab"],
    target_prob_step,
    batch_sharding: NamedSharding,
    global_batch_size: int,
) -> tuple[float, float]:
    val_loader = build_val_loader()
    batches = [next(val_loader)[:2] for _ in range(steps)]
    accum_probs: list[Float[Array, " tokens"] | None] = [None for _ in range(steps)]

    for path, w in zip(ckpt_paths, weights):
        params = eqx.tree_deserialise_leaves(path, params_skeleton)
        for i, (x, y) in enumerate(batches):
            xg = make_global_array(x, batch_sharding, (global_batch_size, MAX_SEQ_LEN))
            yg = make_global_array(y, batch_sharding, (global_batch_size, MAX_SEQ_LEN))
            p = target_prob_step(params, xg, yg)
            accum_probs[i] = p * w if accum_probs[i] is None else accum_probs[i] + p * w

    total_nats = jnp.asarray(0.0, dtype=jnp.float32)
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_bytes = jnp.asarray(0, dtype=jnp.int32)
    total_tokens = jnp.asarray(0, dtype=jnp.int32)
    for probs, (_, y) in zip(accum_probs, batches):
        assert probs is not None
        yg = make_global_array(y, batch_sharding, (global_batch_size, MAX_SEQ_LEN))
        flat_y = yg.reshape(-1)
        mask = flat_y != -1
        log_probs = jnp.log(jnp.maximum(probs, 1e-40))
        num_bytes = jnp.take(token_bytes, jnp.maximum(flat_y, 0))
        total_nats += jnp.sum(-log_probs * (num_bytes > 0), dtype=jnp.float32)
        total_bytes += jnp.sum(num_bytes, dtype=jnp.int32)
        total_loss += jnp.sum(jnp.where(mask, -log_probs, 0.0), dtype=jnp.float32)
        total_tokens += jnp.sum(mask, dtype=jnp.int32)

    total_nats, total_bytes, total_loss, total_tokens = [
        np.asarray(jax.device_get(x))
        for x in (total_nats, total_bytes, total_loss, total_tokens)
    ]
    bpb = (
        float(total_nats) / (math.log(2) * int(total_bytes))
        if int(total_bytes) > 0
        else float("inf")
    )
    loss = (
        float(total_loss) / int(total_tokens) if int(total_tokens) > 0 else float("inf")
    )
    return bpb, loss


def save_params(path: str, params: PyTree[Float[Array, "..."]]) -> None:
    if is_process0():
        eqx.tree_serialise_leaves(path, params)


@eqx.filter_jit
def update_ema_params(
    ema_params: PyTree[Float[Array, "..."] | None],
    params: PyTree[Float[Array, "..."] | None],
    beta: Float[Array, ""],
) -> PyTree[Float[Array, "..."] | None]:
    return jtu.tree_map(
        lambda ema, p: None if p is None else ema + (p - ema) * (1.0 - beta),
        ema_params,
        params,
        is_leaf=lambda x: x is None,
    )


def weighted_average_params(
    params_skeleton: PyTree[Float[Array, "..."] | None],
    ckpt_paths: list[str],
    weights: list[float],
) -> PyTree[Float[Array, "..."] | None]:
    avg_params = tree_zeros_like(params_skeleton)
    for path, weight in zip(ckpt_paths, weights):
        params = eqx.tree_deserialise_leaves(path, params_skeleton)
        avg_params = jtu.tree_map(
            lambda avg, p: None if p is None else avg + p * weight,
            avg_params,
            params,
            is_leaf=lambda x: x is None,
        )
    return avg_params


# =============================================================================
# Main
# =============================================================================

master_process = is_process0()

run_name, run_dir = resolve_run_dir(args.run_name)
checkpoints_dir = os.path.join(run_dir, "checkpoints")
artifact_model_path = os.path.join(run_dir, "model.eqx")
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

_wandb_kwargs: dict[str, Any] = {
    "project": "slowrun",
    "name": run_name,
    "dir": os.path.join(run_dir, "wandb"),
}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = (
    DummyWandb()
    if (not master_process or args.disable_wandb)
    else wandb.init(**_wandb_kwargs)
)
if master_process and not args.disable_wandb:
    wandb_run.log_code(".")

print0("--- Hyperparameters ---")
print0(f"  n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}, head_dim={HEAD_DIM}")
print0(
    f"  seq_len={MAX_SEQ_LEN}, window_pattern={WINDOW_PATTERN}, compute_dtype={args.compute_dtype}"
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
print0(
    f"  wd_schedule=hold@{args.weight_decay}->mid@{args.wd_mid}->end@{args.wd_end}"
)
print0(f"  num_epochs={args.num_epochs}, patience={args.patience}")
print0(f"  dropout={args.dropout}, doc_shuffle={not args.no_doc_shuffle}")
print0(f"  iha={args.iha}, iha_lr={args.iha_lr}")
print0(
    f"  jax_processes={jax.process_count()}, local_devices={jax.local_device_count()}, devices={jax.device_count()}"
)
print0(f"  run={run_name}")
print0(f"  run_dir={run_dir}")
print0("-----------------------")

encoder = tiktoken.get_encoding("gpt2")
vocab_size = encoder.n_vocab
padded_vocab = ((vocab_size + 63) // 64) * 64
print0(f"Vocab size: {vocab_size:,} (padded to {padded_vocab:,})")

eot_id = encoder._special_tokens["<|endoftext|>"]
token_bytes_list = []
for i in range(vocab_size):
    token_bytes_list.append(
        0 if i == eot_id else len(encoder.decode_single_token_bytes(i))
    )
token_bytes_np = np.asarray(token_bytes_list, dtype=np.int32)

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    padded_vocab_size=padded_vocab,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_kv_head=N_HEAD,
    n_embd=N_EMBD,
    window_pattern=WINDOW_PATTERN,
    dropout=args.dropout,
    stoch_depth=args.stoch_depth,
    use_iha=args.iha,
    iha_mix_v=args.iha,
    mtp_weight=args.mtp_weight,
    logit_cap=args.logit_cap,
    compute_dtype=args.compute_dtype,
)

dupe_can_activate = (
    not args.eval_logit_avg
) and args.dupe_start_epoch <= args.num_epochs
if dupe_can_activate:
    if args.dupe_layers_start < config.n_layer // 2:
        raise ValueError("dupe layers must be decoder-only")
    if args.dupe_layers_end > config.n_layer:
        raise ValueError("dupe_layers_end exceeds n_layer")

model_key, train_key = jax.random.split(jax.random.key(42))
model = GPT(config, model_key)
params, model_static = eqx.partition(model, eqx.is_inexact_array)
opt_specs = build_optimizer_spec(params)
opt_state = init_optimizer_state(params, opt_specs)
state = TrainState(params, opt_state, train_key, jnp.asarray(0, dtype=jnp.int32))

param_counts = tree_sum(params)
num_flops_per_token = model.estimate_flops(param_counts)
print0(f"Parameters: {param_counts:,}")
print0(f"FLOPs per token: {num_flops_per_token:e}")

mesh = Mesh(np.asarray(jax.devices()), ("data",))
state_sharding = NamedSharding(mesh, P())
batch_sharding = NamedSharding(mesh, P("data", None))
microbatch_sharding = NamedSharding(mesh, P(None, "data", None))
token_bytes = jax.device_put(jnp.asarray(token_bytes_np), state_sharding)
state = jax.device_put(state, state_sharding)

global_batch_size = args.device_batch_size * jax.device_count()
per_process_batch = args.device_batch_size * jax.local_device_count()
tokens_per_fwdbwd = global_batch_size * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, (
    f"total_batch_size={TOTAL_BATCH_SIZE} must divide device_batch_size * device_count * seq_len = "
    f"{tokens_per_fwdbwd}"
)
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

_train_path = (
    args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
)
_val_path = (
    args.input_val_bin
    if args.input_val_bin
    else os.path.join(DATA_DIR, "fineweb_val.pt")
)
train_loader = DataLoader(
    _train_path, per_process_batch, MAX_SEQ_LEN, doc_shuffle=not args.no_doc_shuffle
)


def build_val_loader():
    return DataLoader(_val_path, per_process_batch, MAX_SEQ_LEN, doc_shuffle=False)


TOKENS_PER_EPOCH = train_loader.total_tokens
num_iterations = max(1, round(TOKENS_PER_EPOCH * args.num_epochs / TOTAL_BATCH_SIZE))
steps_per_epoch = num_iterations / args.num_epochs
wd_phase1_end_step = round(args.wd_phase1_epoch / args.num_epochs * num_iterations)
wd_phase2_end_step = round(args.wd_phase2_epoch / args.num_epochs * num_iterations)
_swa_start_step = (
    (num_iterations - args.swa_last_epochs * steps_per_epoch)
    if args.swa_last_epochs > 0
    else -1
)
eval_steps = max(1, EVAL_TOKENS // (tokens_per_fwdbwd))

print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
print0(f"Training for {args.num_epochs} epoch(s) (~{num_iterations} steps estimated)")
print0(f"Eval set: {EVAL_TOKENS:,} tokens ({eval_steps} step(s))")


def get_lr_multiplier(it: int) -> float:
    warmup = round(WARMUP_RATIO * num_iterations)
    warmdown = round(WARMDOWN_RATIO * num_iterations)
    if warmup > 0 and it < warmup:
        return (it + 1) / warmup
    if warmdown <= 0 or it <= num_iterations - warmdown:
        return 1.0
    progress = max(num_iterations - it, 0) / warmdown
    return progress + (1 - progress) * FINAL_LR_FRAC


def get_muon_momentum(it: int) -> float:
    t = min(it / 300, 1)
    return (1 - t) * 0.85 + t * 0.95


def get_wd_multiplier(it: int) -> float:
    wd = float(
        np.interp(
            it,
            [0, wd_phase1_end_step, wd_phase2_end_step, num_iterations],
            [args.weight_decay, args.weight_decay, args.wd_mid, args.wd_end],
        )
    )
    return wd / args.weight_decay if args.weight_decay > 0 else 0.0


dupe_active = False
train_step = make_train_step(
    model_static, opt_specs, grad_accum_steps, microbatch_sharding, dupe_active
)
eval_step = make_eval_step(model_static, batch_sharding, dupe_active)

step = 0
current_epoch = train_loader.epoch
min_val_bpb = float("inf")
min_val_loss = float("inf")
val_loss = float("inf")
epochs_without_improvement = 0
smooth_train_loss = 0.0
total_training_time = 0.0
timed_steps = 0
timing_start_step = 4
param_ema_beta = (
    args.ema_decay_per_epoch ** (args.update_ema_every / steps_per_epoch)
    if args.update_ema_every > 0
    else 0.0
)
ema_params = tree_zeros_like(state.params) if args.update_ema_every > 0 else None

late_ckpt_paths: list[str] = []
late_logit_paths: list[str] = []
logit_avg_count = args.logit_avg
target_prob_step = (
    make_target_prob_step(model_static, batch_sharding, dupe_active)
    if logit_avg_count > 0
    else None
)
if logit_avg_count > 0 and master_process:
    os.makedirs(args.logit_avg_dir, exist_ok=True)
if logit_avg_count > 0:
    print0(
        f"Logit averaging: saving last {logit_avg_count} epoch checkpoints to {args.logit_avg_dir}/"
    )

if args.eval_logit_avg:
    print0("--eval-logit-avg set: skipping training, loading checkpoints from disk.")
else:
    val_bpb, val_loss = evaluate_bpb(
        state.params,
        build_val_loader,
        eval_steps,
        token_bytes,
        eval_step,
        batch_sharding,
        global_batch_size,
    )
    print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
    wandb_run.log({"step": step, "val/bpb": val_bpb, "val/loss": val_loss})
    min_val_bpb = val_bpb
    min_val_loss = val_loss

while not args.eval_logit_avg and current_epoch <= args.num_epochs:
    if not dupe_active and current_epoch >= args.dupe_start_epoch:
        print0(f"\n=== Enabling dupe-layers at epoch {current_epoch} ===")
        dupe_active = True
        train_step = make_train_step(
            model_static, opt_specs, grad_accum_steps, microbatch_sharding, dupe_active
        )
        eval_step = make_eval_step(model_static, batch_sharding, dupe_active)
        if target_prob_step is not None:
            target_prob_step = make_target_prob_step(
                model_static, batch_sharding, dupe_active
            )
        timing_start_step = step + 4
        gc.collect()

    t0 = time.time()
    local_xs, local_ys, epoch = next_microbatches(train_loader, grad_accum_steps)
    xs = make_global_array(
        local_xs,
        microbatch_sharding,
        (grad_accum_steps, global_batch_size, MAX_SEQ_LEN),
    )
    ys = make_global_array(
        local_ys,
        microbatch_sharding,
        (grad_accum_steps, global_batch_size, MAX_SEQ_LEN),
    )

    lrm = get_lr_multiplier(step)
    if _swa_start_step >= 0 and step >= _swa_start_step:
        cycle_pos = (step - _swa_start_step) % steps_per_epoch
        swa_base = max(lrm, 0.05)
        lrm = (
            0.05
            + (swa_base - 0.05)
            * (1 + math.cos(math.pi * cycle_pos / steps_per_epoch))
            / 2
        )
    wdm = get_wd_multiplier(step)
    state, metrics = train_step(
        state,
        xs,
        ys,
        jnp.asarray(lrm, dtype=jnp.float32),
        jnp.asarray(wdm, dtype=jnp.float32),
        jnp.asarray(get_muon_momentum(step), dtype=jnp.float32),
    )
    metrics = jax.device_get(metrics)
    train_loss_f = float(metrics["loss"])
    if ema_params is not None and step % args.update_ema_every == 0:
        ema_params = update_ema_params(
            ema_params, state.params, jnp.asarray(param_ema_beta, dtype=jnp.float32)
        )
    state.step.block_until_ready()
    dt = time.time() - t0
    step += 1

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta**step)
    pct = 100 * step / num_iterations
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
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
        f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | dt: {dt * 1000:.2f}ms | "
        f"tok/sec: {tok_per_sec:,}{dupe_str}{eta_str}"
    )
    wandb_run.log(
        {
            "step": step,
            "train/loss": debiased,
            "train/lm_loss": float(metrics["lm_loss"]),
            "train/mtp_loss": float(metrics["mtp_loss"]),
            "train/lr_mult": lrm,
            "train/wd_mult": wdm,
        }
    )

    if epoch != current_epoch:
        val_bpb, val_loss = evaluate_bpb(
            state.params,
            build_val_loader,
            eval_steps,
            token_bytes,
            eval_step,
            batch_sharding,
            global_batch_size,
        )
        print0(
            f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}"
        )
        wandb_run.log(
            {
                "step": step,
                "epoch": current_epoch,
                "val/bpb": val_bpb,
                "val/loss": val_loss,
            }
        )
        if args.swa_last_epochs > 0:
            ckpt_path = os.path.join(checkpoints_dir, f"epoch_{current_epoch:03d}.eqx")
            save_params(ckpt_path, state.params)
            late_ckpt_paths.append(ckpt_path)
            if len(late_ckpt_paths) > args.swa_last_epochs:
                old = late_ckpt_paths.pop(0)
                if master_process and os.path.exists(old):
                    os.remove(old)

        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
            min_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience >= 0 and epochs_without_improvement >= args.patience:
                print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                break

        if logit_avg_count > 0:
            ckpt_path = os.path.join(
                args.logit_avg_dir, f"epoch_{current_epoch:03d}.eqx"
            )
            save_params(ckpt_path, state.params)
            late_logit_paths.append(ckpt_path)
            if len(late_logit_paths) > logit_avg_count:
                old = late_logit_paths.pop(0)
                if master_process and os.path.exists(old):
                    os.remove(old)
            print0(
                f"  Saved checkpoint {ckpt_path} ({len(late_logit_paths)}/{logit_avg_count})"
            )

        current_epoch = epoch

    if step == 1:
        gc.collect()


# =============================================================================
# Post-training: evaluate EMA and checkpoint averages
# =============================================================================

artifact_params = state.params

if ema_params is not None:
    ema_updates = step // args.update_ema_every
    if ema_updates > 0:
        correction = 1.0 / (1.0 - param_ema_beta**ema_updates)
        ema_eval_params = jtu.tree_map(
            lambda p: None if p is None else p * correction,
            ema_params,
            is_leaf=lambda x: x is None,
        )
        ema_bpb, ema_loss = evaluate_bpb(
            ema_eval_params,
            build_val_loader,
            eval_steps,
            token_bytes,
            eval_step,
            batch_sharding,
            global_batch_size,
        )
        print0(f"EMA Val BPB: {ema_bpb:.6f} | EMA Val Loss: {ema_loss:.6f}")
        wandb_run.log({"step": step, "val/ema_bpb": ema_bpb, "val/ema_loss": ema_loss})
        val_bpb = ema_bpb
        val_loss = ema_loss
        artifact_params = ema_eval_params
        if ema_bpb < min_val_bpb:
            min_val_bpb = ema_bpb
            min_val_loss = ema_loss

if len(late_ckpt_paths) >= 2:
    n = len(late_ckpt_paths)
    raw_w = list(range(1, n + 1))
    weights = [w / sum(raw_w) for w in raw_w]
    avg_params = weighted_average_params(state.params, late_ckpt_paths, weights)
    avg_bpb, avg_loss = evaluate_bpb(
        avg_params,
        build_val_loader,
        eval_steps,
        token_bytes,
        eval_step,
        batch_sharding,
        global_batch_size,
    )
    print0(f"Ckpt avg Val BPB: {avg_bpb:.6f} | Val Loss: {avg_loss:.6f}")
    wandb_run.log({"ckpt_avg/bpb": avg_bpb, "ckpt_avg/loss": avg_loss})
    artifact_params = avg_params
    if avg_loss < min_val_loss:
        min_val_loss, min_val_bpb = avg_loss, avg_bpb

if logit_avg_count > 0:
    if args.eval_logit_avg:
        all_disk = sorted(glob.glob(os.path.join(args.logit_avg_dir, "epoch_*.eqx")))
        ckpt_paths_for_logit = all_disk[-logit_avg_count:]
    else:
        ckpt_paths_for_logit = late_logit_paths

    if len(ckpt_paths_for_logit) >= 2:
        assert target_prob_step is not None
        n = len(ckpt_paths_for_logit)
        print0(
            f"\n--- Evaluating logit avg ({n} checkpoints: {[os.path.basename(p) for p in ckpt_paths_for_logit]}) ---"
        )

        def _run_mode(label: str, weights: list[float]) -> tuple[float, float]:
            print0(f"  [{label}] weights: {[f'{w:.3f}' for w in weights]}")
            bpb, loss = evaluate_bpb_logit_avg(
                model_static,
                state.params,
                ckpt_paths_for_logit,
                weights,
                build_val_loader,
                eval_steps,
                token_bytes,
                target_prob_step,
                batch_sharding,
                global_batch_size,
            )
            print0(f"  [{label}] Val BPB: {bpb:.6f} | Val Loss: {loss:.6f}")
            wandb_run.log(
                {f"logit_avg_{label}/bpb": bpb, f"logit_avg_{label}/loss": loss}
            )
            return bpb, loss

        equal_w = [1.0 / n] * n
        raw_w = list(range(1, n + 1))
        weighted_w = [w / sum(raw_w) for w in raw_w]

        if args.logit_avg_mode in ("equal", "both"):
            eq_bpb, eq_loss = _run_mode("equal", equal_w)
            if eq_loss < min_val_loss:
                min_val_loss, min_val_bpb = eq_loss, eq_bpb
                print0("  ** New best! (logit avg equal weights)")

        if args.logit_avg_mode in ("weighted", "both"):
            wt_bpb, wt_loss = _run_mode("weighted", weighted_w)
            if wt_loss < min_val_loss:
                min_val_loss, min_val_bpb = wt_loss, wt_bpb
                print0("  ** New best! (logit avg recency weights)")


print0(f"Total training time: {total_training_time / 60:.2f}m")
final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float("inf")
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss

_result_out = args.save_result or result_path
if master_process:
    result = {
        "matrix_lr": args.matrix_lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "val_loss": val_loss,
        "best_val_loss": min_val_loss,
        "wandb_url": getattr(wandb_run, "url", None),
        "jax_devices": jax.device_count(),
        "compute_dtype": args.compute_dtype,
    }
    with open(_result_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print0(f"Result saved to {_result_out}")
    print0(f"Saving model to {artifact_model_path}")
    save_params(artifact_model_path, artifact_params)

total_wall_time = time.time() - _script_start
print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time / 60:.2f}m)")

wandb_run.finish()
if artifacts_log_f is not None:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    artifacts_log_f.close()
