from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

from app.workflows.checkpointing import _checkpoint_json_default


class ExampleStatus(str, Enum):
    READY = "ready"


class ExampleState(BaseModel):
    status: ExampleStatus


def test_checkpoint_json_default_encodes_validated_graph_values() -> None:
    assert _checkpoint_json_default(ExampleState(status=ExampleStatus.READY)) == {
        "status": "ready"
    }
    assert _checkpoint_json_default(ExampleStatus.READY) == "ready"
    assert _checkpoint_json_default(datetime(2026, 8, 5, 12, 0)) == (
        "2026-08-05T12:00:00"
    )
    assert _checkpoint_json_default(Path("safe/path")) == "safe/path"
    assert _checkpoint_json_default(
        UUID("00000000-0000-0000-0000-000000000001")
    ) == "00000000-0000-0000-0000-000000000001"


def test_checkpoint_json_default_rejects_unknown_objects() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        _checkpoint_json_default(object())
