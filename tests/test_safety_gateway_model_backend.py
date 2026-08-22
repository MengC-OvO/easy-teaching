import pytest

from safety_gateway.model import ModelInferenceError, select_model_backend
from safety_gateway.settings import GatewaySettings


def test_auto_prefers_cuda_when_both_backends_are_reported() -> None:
    assert select_model_backend("auto", cuda_available=True, mps_available=True) == "cuda"


def test_auto_selects_apple_mps_without_cuda() -> None:
    assert select_model_backend("auto", cuda_available=False, mps_available=True) == "mps"


def test_explicit_unavailable_backend_fails_closed() -> None:
    with pytest.raises(ModelInferenceError, match="MPS"):
        select_model_backend("mps", cuda_available=True, mps_available=False)


def test_no_gpu_never_silently_falls_back_to_cpu() -> None:
    with pytest.raises(ModelInferenceError, match="no supported GPU"):
        select_model_backend("auto", cuda_available=False, mps_available=False)


def test_gateway_backend_defaults_to_auto_and_can_be_overridden() -> None:
    assert GatewaySettings(_env_file=None).model_backend == "auto"
    assert GatewaySettings(_env_file=None, SAFETY_MODEL_BACKEND="mps").model_backend == "mps"
