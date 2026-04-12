# rl-memory-agent

Research code scaffold for a **constraint-aware reinforcement learning (RL) agent** that adapts
memory-related configuration knobs online to avoid out-of-memory (OOM) events while improving
time-to-result in low-resource distributed deep learning.

This project is designed to match the Paper 2 formulation:
- **Problem**: constrained Markov decision process (CMDP)
- **Solution**: Lagrangian RL with primal–dual updates + explicit runtime safety shield/rollback

## Quick start (toy environment)

```bash
cd rl_memory_agent
python -m pip install -e .
rl-memory-agent toy --total-steps 50000
```

The toy environment simulates peak memory and step time as a function of a knob vector
(micro-batch size, gradient accumulation, checkpointing, precision, sharding, bucket cap).

## Integration idea (real training)

The intended integration is to wrap an existing training loop and provide:
- **Sensors**: VRAM usage and headroom, step-time breakdown, comm/I/O time, failures (OOM, restarts)
- **Actuators**: safe updates to a compact knob set at safe boundaries

See `src/rl_memory_agent/env.py` and `src/rl_memory_agent/safety.py` for the interfaces and the
expected data flow.

## End-to-end experiments (real training)

See `EXPERIMENTS.md` for a concrete end-to-end evaluation protocol (baselines, scenarios, metrics)
and a minimal integration checklist.

## Online training with SimGrid (simulated cluster)

If you run the SimGrid cluster environment in external-controller mode, the agent can connect and
train online from the simulator telemetry:

```bash
# Terminal A (simulator)
./simgrid_cluster_env/build/cluster_env simgrid_cluster_env/platforms/cluster_4hosts.xml \
  --mode online --controller external --control-port 5555 --control-interval 10

# Terminal B (agent)
cd rl_memory_agent
python -m pip install -e .
rl-memory-agent simgrid --host 127.0.0.1 --port 5555 --rollout-steps 256
```

Multi-constraint mode (vector costs) is supported via `--cost-components` and `--cost-limits`:

```bash
rl-memory-agent simgrid \
  --cost-components mem_overflow,comm_frac,io_frac \
  --cost-limits 0.0,0.35,0.10
```
