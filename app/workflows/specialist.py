"""Callable contract shared by all specialist workflows."""

from typing import Mapping, Protocol, Union

from app.schemas import SpecialistInput, SpecialistResult


SpecialistWorkflowOutput = Union[SpecialistResult, Mapping[str, object]]


class SpecialistWorkflowProtocol(Protocol):
    def invoke(self, state: SpecialistInput) -> SpecialistWorkflowOutput:
        ...
