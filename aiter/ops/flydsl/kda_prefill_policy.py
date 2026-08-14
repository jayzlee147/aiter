# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scheduling policy shared by the FlyDSL KDA prefill route families.

The names describe the workspace algebra (W/U or C/direct-RTP) and whether
the state recurrence and output are fused or split.  Keeping this decision on
the host makes every leaf kernel independently forceable in correctness and
benchmark tests.
"""

from __future__ import annotations

from enum import Enum
import os


class KdaPrefillRoute(str, Enum):
    WF = "wf"  # W/U workspace, fused recurrence + output
    WS = "ws"  # W/U workspace, split state scan + output replay
    CF = "cf"  # C/direct-RTP workspace, fused recurrence + output
    CS = "cs"  # C/direct-RTP workspace, split scan + output replay
    RECURRENT = "recurrent"  # tokenwise correctness/very-short fallback


_ROUTE_ENV = "AITER_FLYDSL_KDA_PREFILL_ROUTE"


def _forced_route() -> KdaPrefillRoute | None:
    value = os.getenv(_ROUTE_ENV, "").strip().lower()
    if not value or value == "auto":
        return None
    try:
        return KdaPrefillRoute(value)
    except ValueError as exc:
        choices = ", ".join(("auto", *(r.value for r in KdaPrefillRoute)))
        raise ValueError(f"invalid {_ROUTE_ENV}={value!r}; expected one of {choices}") from exc


def select_kda_prefill_route(
    *, arch: str, total_tokens: int, num_sequences: int
) -> KdaPrefillRoute:
    """Select a route without inspecting tensors or synchronizing the device.

    gfx942 follows FlashKDA's production BT64 direct-RTP split pipeline.
    gfx950 uses the measured average-length crossover: short/high-grid inputs
    use register-state W/U fused work, while longer/low-grid inputs use CS.
    The recurrent leaf remains the conservative route on unknown hardware.
    """
    forced = _forced_route()
    if forced is not None:
        return forced
    n = max(1, int(num_sequences))
    avg_tokens = int(total_tokens) // n
    arch = str(arch or "").split(":", 1)[0]
    if arch.startswith("gfx942"):
        return KdaPrefillRoute.CS
    if arch.startswith("gfx950"):
        if avg_tokens >= 2048 or (n < 16 and avg_tokens >= 512):
            return KdaPrefillRoute.CS
        return KdaPrefillRoute.WF
    return KdaPrefillRoute.RECURRENT


__all__ = ["KdaPrefillRoute", "select_kda_prefill_route"]
