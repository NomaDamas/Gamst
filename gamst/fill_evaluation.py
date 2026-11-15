"""Trusted opt-in recording hooks for the actual bounded fill chain.

Selected by the built-in recording runtime, never loaded from task/source JSON.
Each inspect call captures its immutable ownership before any await. Hooks do
not choose the fill predicate, claim parsing, stage order, or escalation rule.
"""
from abc import ABC, abstractmethod


class FillEvaluationContext(ABC):
    @abstractmethod
    def validate(self, driver, frame):pass
    @abstractmethod
    def measure(self, callback):
        """Invoke once inside the actual owned OpenCV thread."""
        pass
    @abstractmethod
    def published(self, role, record):pass
    @abstractmethod
    def stage_arguments(self, stage, request_id, adapter):
        """Only immutable scope/relationship/predecessor execution kwargs."""
        pass
    @abstractmethod
    def stage_finished(self, stage, request_id, response, error):pass
    @abstractmethod
    def completed(self, case, record):pass
    def partial(self, case, outcome):
        """Optional detached failed/cancelled prefix before private case retirement."""
        pass
    @abstractmethod
    def closed(self, outcome):pass
