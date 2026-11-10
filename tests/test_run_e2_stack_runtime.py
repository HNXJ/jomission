"""scripts/run_e2_stack.runtime_kw: unset keeps the default route; 'event' selects the event runtime; anything else stops."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_e2_stack as rs  # noqa: E402

FAKE = SimpleNamespace(RuntimeConfig=lambda **kw: ("cfg", tuple(sorted(kw.items()))))


def test_unset_is_default(monkeypatch):
    monkeypatch.delenv("JAXFNE_EDGE_DELIVERY", raising=False)
    assert rs.runtime_kw(FAKE) == {}


def test_event(monkeypatch):
    monkeypatch.setenv("JAXFNE_EDGE_DELIVERY", "event")
    assert rs.runtime_kw(FAKE) == {"runtime": ("cfg", (("edge_delivery", "event"), ("recurrent_backend", "edge_list")))}


def test_unknown_stops(monkeypatch):
    monkeypatch.setenv("JAXFNE_EDGE_DELIVERY", "dense")
    with pytest.raises(SystemExit):
        rs.runtime_kw(FAKE)
