# End-to-end experimental protocol (real distributed training)

This document proposes a practical **end-to-end** evaluation process for the `rl-memory-agent`
controller (Paper 2), beyond the included toy environment.

The core idea is to run *real* distributed training jobs while the controller:
- **observes** runtime telemetry (VRAM + timing + events),
- **acts** on a compact set of knobs at safe boundaries,
- **maintains safety** via a shield + rollback policy,
- and is evaluated on safety/performance/robustness metrics.

---

## 1) Goals and hypotheses

**Primary goal**: keep OOM near zero while improving **time-to-result** under low-resource, non-stationary conditions.

**Hypotheses**
- H1 (Safety): shield + rollback yields near-zero OOM (or fast recovery) compared to unsafe learning.
- H2 (Performance): online adaptation beats static tuning when bandwidth/I/O/interruption patterns change.
- H3 (Robustness): controller recovers quickly after shifts (bandwidth drop, I/O slowdown, preemption).

---

## 2) Experimental design overview

Each “run” is a distributed training job defined by:
- **Stack**: PyTorch DDP or PyTorch FSDP (single-node first; multi-node second).
- **Workload**: one model + dataset + sequence length + optimizer.
- **Resource regime**: VRAM budget (small GPUs), network/I/O characteristics, interruption profile.
- **Controller**: disabled (baselines) or enabled (our agent).
- **Scenario**: stationary (nominal) or with injected non-stationarity.

**Controller loop**
- Every `K` steps (or at checkpoint boundary), collect telemetry → build state → pick action.
- Apply action only at **safe boundaries** (end-of-step / pre-forward / post-checkpoint).
- If OOM happens: rollback to last safe config + resume from last checkpoint.

---

## 3) What “end-to-end” means here (minimal viable scope)

To keep runtime actuation feasible in real frameworks, split knobs into:

**Fast knobs** (change online without restart — recommended for end-to-end experiments)
- `micro_batch` (or effective micro-batch via slicing)
- `grad_accum_steps`
- `ckpt_interval_steps`
- DDP communication bucket knobs (where supported)

**Slow knobs** (often require restart / re-wrap — treat as per-run settings)
- sharding mode (DDP vs FSDP/ZeRO)
- precision mode (FP32/FP16/BF16) if it implies recasting weights/optimizer state
- activation checkpointing if your model cannot toggle it dynamically

Recommended end-to-end evaluation: **start with fast knobs only**, then add slow knobs as a second phase with controlled restarts.

---

## 4) Telemetry: what to measure (sensors)

At minimum (cheap and reliable):
- Step wall time: `time.perf_counter()`
- VRAM: `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`
- OOM event: catch CUDA OOM exception
- Restart event: true on resume-from-checkpoint

Optional (adds accuracy but more engineering):
- Compute vs comm breakdown (CUDA events + wrapped collectives or PyTorch Profiler)
- Checkpoint I/O time (timing around save/load)

Make the telemetry schema match `TelemetrySample` in `src/rl_memory_agent/telemetry.py`.

---

## 5) Actuation: how to apply knobs safely (actuators)

Practical patterns:
- **micro-batch**: either rebuild the DataLoader at boundary, or prefetch a max batch and slice it.
- **grad accumulation**: change the number of micro-steps per optimizer step.
- **checkpoint interval**: change `every_n_steps` on the checkpoint handler.
- **bucket cap** (DDP): configure when initializing DDP; for online change you typically need restart.

Safety boundaries:
- Only apply changes when gradients are cleared and optimizer state is consistent.
- For “slow knobs”, apply only on planned restart points (checkpoint → stop → relaunch with new config).

---

## 6) Baselines and ablations

**Baselines**
- Static tuned config (offline grid/manual search under nominal conditions)
- Conservative static config (high safety margin)
- Rule-based headroom controller (threshold policy)

**Ablations**
- RL without shield (expect unsafe exploration; keep short and controlled)
- Shield without learning (pure safe heuristic)
- Dual update on/off (fixed lambda vs primal–dual)

---

## 7) Scenarios (non-stationarity / low-resource stress)

Pick at least 3, with repeatable injections:

1) **Bandwidth variability**
   - Multi-node: Linux `tc netem` on NIC (rate limit + latency/jitter).
   - Single-node: emulate comm delay by adding controlled sleeps around all-reduce (wrapper).

2) **I/O variability**
   - Write checkpoints to a slower mount / throttle writes
   - Inject sleep in checkpoint save path to emulate bursty storage

3) **Interruptions**
   - Kill and resume job from checkpoint on a schedule (or random Poisson process)

4) **Memory noise**
   - Allocate transient tensors periodically to emulate fragmentation/transient buffers

---

## 8) Metrics to report

**Safety**
- OOM count and OOM rate (#/1k steps)
- Peak VRAM percentiles (p50/p95/p99), headroom margin
- Overflow ratio cost (if using an explicit overflow cost)

**Performance**
- Throughput (samples/s or tokens/s)
- Step time distribution (mean, p95)
- Time-to-result to a fixed step/token budget
- (Optional) time-to-target validation metric

**Communication / I/O**
- Communication fraction (comm time / step time), collective time (when measurable)
- Checkpoint time and checkpoint throughput (MB/s)

**Quality / energy (optional)**
- Final validation metric and accuracy vs a static-tuned baseline
- Energy (if power is available) or a proxy (mean power × step time)

**Robustness**
- Degradation after a shift (delta throughput)
- Recovery time (steps/time to return within X% of pre-shift throughput)

**Controller overhead**
- Decision latency
- Number of knob changes
- Number of shield blocks and rollbacks

---

## 9) Protocol (repeatability and comparisons)

For each workload × regime × scenario:
- Run **N seeds** (recommend N ≥ 3).
- Fix a **training budget** (steps/tokens) for comparable wall-clock measurements.
- Use a **warm-up window** (ignore first W steps for metrics).
- Log every control interval and aggregate metrics at the end.

Controller training strategy options:
- **A) Pretrain on toy env**, then online fine-tune on real job for first `T_adapt` steps, then **freeze** and evaluate.
- **B) Pure online learning** per run (simpler but more variance).

---

## 10) Minimal implementation checklist (to make this real)

You will need a small integration layer (not provided in this scaffold yet):
- A telemetry collector around the training loop.
- An actuator that can apply the selected subset of knobs at boundaries.
- A checkpoint + resume flow for rollback/restarts.
- A logger that writes JSONL/CSV for post-run analysis.

Start small:
1) DDP single-node, one workload, only `grad_accum_steps` and batch slicing.
2) Add checkpointing + interruption injection.
3) Add multi-node + bandwidth shaping.
