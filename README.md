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

