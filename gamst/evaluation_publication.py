"""Opt-in synchronous export of an actual accepted SDK observation.

This trusted-code callback is not loadable from a recording manifest. It must
retain only its separately admitted bounded evidence and must not block or start
unowned work. Default SDK ingestion has no exporter.
"""
from abc import ABC, abstractmethod


class AcceptedObservationSink(ABC):
    @abstractmethod
    def record(self, observation_record, operation_reference):
        """Receive a detached accepted record and its actual native operation."""
        raise NotImplementedError
