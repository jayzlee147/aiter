import pytest

from aiter.ops.flydsl.kda_prefill_policy import (
    KdaPrefillRoute,
    select_kda_prefill_route,
)


@pytest.mark.parametrize(
    "arch,total,n,expected",
    [
        ("gfx942", 128, 1, KdaPrefillRoute.CS),
        ("gfx942", 16384, 64, KdaPrefillRoute.CS),
        ("gfx950", 128, 1, KdaPrefillRoute.WF),
        ("gfx950", 16384, 16, KdaPrefillRoute.WF),
        ("gfx950", 16384, 8, KdaPrefillRoute.CS),
        ("gfx950", 512, 1, KdaPrefillRoute.CS),
        ("unknown", 128, 1, KdaPrefillRoute.RECURRENT),
    ],
)
def test_auto_policy(monkeypatch, arch, total, n, expected):
    monkeypatch.delenv("AITER_FLYDSL_KDA_PREFILL_ROUTE", raising=False)
    assert select_kda_prefill_route(
        arch=arch, total_tokens=total, num_sequences=n
    ) is expected


@pytest.mark.parametrize("route", list(KdaPrefillRoute))
def test_force_every_route(monkeypatch, route):
    monkeypatch.setenv("AITER_FLYDSL_KDA_PREFILL_ROUTE", route.value.upper())
    assert select_kda_prefill_route(
        arch="unknown", total_tokens=1, num_sequences=1
    ) is route


def test_invalid_force(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_KDA_PREFILL_ROUTE", "nope")
    with pytest.raises(ValueError, match="AITER_FLYDSL_KDA_PREFILL_ROUTE"):
        select_kda_prefill_route(arch="gfx942", total_tokens=1, num_sequences=1)
