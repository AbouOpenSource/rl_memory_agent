from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass(frozen=True)
class TelemetrySample:
    step: int
    vram_allocated_mb: float
    vram_reserved_mb: float
    vram_peak_mb: float
    step_time_s: float
    compute_time_s: float
    comm_time_s: float
    io_time_s: float
    oom: bool = False
    restart: bool = False


class TelemetryWindow:
    def __init__(self, maxlen: int = 20) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")
        self._samples: Deque[TelemetrySample] = deque(maxlen=maxlen)

    def append(self, sample: TelemetrySample) -> None:
        self._samples.append(sample)

    @property
    def maxlen(self) -> int:
        return int(self._samples.maxlen or 0)

    def last(self) -> Optional[TelemetrySample]:
        if not self._samples:
            return None
        return self._samples[-1]

    def __len__(self) -> int:
        return len(self._samples)

    def mean_step_time_s(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.step_time_s for s in self._samples) / len(self._samples)

    def mean_peak_mb(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.vram_peak_mb for s in self._samples) / len(self._samples)

    def last_oom(self) -> float:
        last = self.last()
        return float(last.oom) if last else 0.0

