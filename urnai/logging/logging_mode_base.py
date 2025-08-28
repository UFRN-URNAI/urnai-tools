from __future__ import annotations

from enum import Enum


class LoggingModeBase(Enum):
    @classmethod
    def check(cls, mode: str | LoggingModeBase) -> None:
        if isinstance(mode, str):
            try:
                mode = cls[mode.upper()]
            except KeyError:
                raise ValueError(f"Invalid mode string: {mode}") from None
        elif not isinstance(mode, cls):
            raise TypeError(f"Mode must be an instance of {cls.__name__} or a string")
        if mode not in cls:
            raise ValueError(f"Invalid mode value: {mode}")
