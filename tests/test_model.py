from __future__ import annotations
import pytest
from loom.types import Decision
from tests.fakes import FakeModelClient, FailingModelClient


def test_fake_model_client_returns_decisions_in_order():
    client = FakeModelClient([Decision("a", {}), Decision("b", {})])
    assert client.decide("t", [], []).tool_name == "a"
    assert client.decide("t", [], []).tool_name == "b"
    assert client.calls == 2


def test_failing_model_client_raises():
    client = FailingModelClient(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        client.decide("t", [], [])
    assert client.calls == 1
