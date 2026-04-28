
# Hybrid Attention

Hybrid attention track led by [@ChinmayK0607](https://x.com/ChinmayKak).

Hybrid attention mixes full softmax layers with linear-attention layers inside the same 30-layer trainer. The trainer supports both GDN and KDA on the same alternating-layer layout, plus hybrid-specific optimizer and dataloader controls used in recent experiments.

On Hopper hosts, full GDN training should use:

```bash
FLA_TILELANG=1
````

## Current Leaderboard

| Rank | Backend     | Best / Final Val Loss | Training Time | Peak Memory    | Notes                                               |
| ---- | ----------- | --------------------- | ------------- | -------------- | --------------------------------------------------- |
| 1    | GDN         | `3.230877 / 3.243014` | `80.38` min   | `54820.06` MiB | Current quality-best baseline with `--muon-eq-r`    |
| 2    | GDN no-conv | `3.234275 / 3.246534` | `71.93` min   | `53583.05` MiB | Current speed-quality frontier with `--gdn-no-conv` |
| 3    | KDA         | `3.239565 / 3.255612` | `89.28` min   | `57735.46` MiB | Slower quality reference line                       |
| 4    | GDN         | `3.234646 / 3.247445` | `80.37` min   | `54820.06` MiB | TileLang GDN control before MuonEq-R                |

## Records

| PR / Run                              | Record                                              | Time                                   |
| ------------------------------------- | --------------------------------------------------- | -------------------------------------- |
| PR `#49` (`1f0fe74`)                  | `3.246` val loss                                    | about `81` minutes                     |
| PR `#58` (`5cb9428`)                  | `3.241282` val loss                                 | `72.33` min training, `76.91` min wall |
| KDA / FlashKDA extension              | `3.239565` best val loss, `3.255612` final val loss | `89.28` min training, `94.42` min wall |
| Current GDN quality-best              | `3.230877` best val loss, `3.243014` final val loss | `80.38` min training                   |
| Current GDN no-conv speed-quality run | `3.234275` best val loss, `3.246534` final val loss | `71.93` min training                   |

## Usage

Current quality-best GDN default:

```bash
FLA_TILELANG=1 torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type gdn \
  --muon-eq-r
```

Current faster GDN alternative:

```bash
FLA_TILELANG=1 torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type gdn \
  --muon-eq-r \
  --gdn-no-conv
```

Current KDA / FlashKDA reference run:

```bash
torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type kda \
  --muon-eq-r
```

Older KDA / FlashKDA runs may also use:

```bash
FLA_FLASH_KDA=1 torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type kda
```

## Trainer Options

* `--linear-attn-type {gdn,kda}` switches the linear-attention block used on the selected hybrid layers.
* `--gdn-no-conv` disables the GDN short-convolution path and is the current best runtime-saving knob.
* `--muon-eq-r` enables row-normalized Muon updates and is part of the current quality-best hybrid baseline.
* `--muon-ns-schedule {polar-express,deepseek-v4}` switches the Muon Newton-Schulz coefficient table.
* `--no-doc-shuffle` disables per-epoch document reshuffling when using the flat-token dataset format.
* `--grad-clip <value>` enables global gradient-norm clipping before the optimizer step.
* `--gdn-use-recurrent` remains experimental and should not be treated as a stable default.

## Practical Guidance

Use the quality-best GDN run as the control for new quality-focused GDN follow-ups.

Use the GDN no-conv variant when runtime matters more than the last `0.003-0.004` of validation loss.

Treat KDA as a reference backend, not the default deployment path on this host. KDA still gives a strong quality reference, but it is materially slower.

For Hopper GDN runs, prefer `FLA_TILELANG=1`. The older non-TileLang GDN path is no longer the right frontier comparison.

## In-Flight Follow-Ups

* Test `--muon-ns-schedule deepseek-v4` on top of the current quality-best GDN baseline.
* Test `--grad-clip 1.0` on top of the current quality-best GDN baseline as a stability-only A/B.

## Brief History

1. PR `#49`: introduced the core hybrid idea. The theoretical change was to mix full softmax attention with GatedDeltaNet, so some layers use recurrent associative memory instead of recomputing all context from scratch. The negative-eigenvalue GDN state makes it easier to track changing latent state, not just accumulate information.

2. PR `#58`: kept the same GDN theory, but improved the efficiency frontier of that idea. The important point was not a new inductive bias; it was making the same recurrent-memory-plus-softmax-correction story cheaper to run, so the track could realize more of the hybrid benefit within a practical time budget.

3. KDA / FlashKDA: upgraded the recurrent memory from a single forget coefficient per head to a per-dimension forget mechanism. The theoretical effect is a more expressive state space, where one head can preserve different subspaces for different timescales instead of forcing the whole head to forget or retain together. That improved loss, but it also increased runtime.

4. Current GDN quality-best: moved the GDN frontier onto the Hopper TileLang path and added `--muon-eq-r`, improving the best validation loss to `3.230877`.

5. Current GDN no-conv speed-quality run: disabled the GDN short-convolution path with `--gdn-no-conv`, giving the current speed-quality frontier. It is faster than the quality-best GDN run while giving up only a small amount of validation loss.

Best loss so far is now the GDN + TileLang + MuonEq-R run. KDA remains an important reference backend, but GDN stays the default because it is faster and currently gives the best overall frontier on this host.

```
```
