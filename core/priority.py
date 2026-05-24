from typing import Literal


PriorityLevel = Literal["low", "normal", "high"]

PRIORITY_RANKS: dict[PriorityLevel, int] = {
    "high": 0,
    "normal": 1,
    "low": 2,
}
PRIORITY_LEVELS: tuple[PriorityLevel, ...] = ("high", "normal", "low")
DEFAULT_PRIORITY: PriorityLevel = "normal"


def priority_rank(priority: PriorityLevel) -> int:
    return PRIORITY_RANKS[priority]


def normalize_priority(value: object) -> PriorityLevel:
    if value in PRIORITY_RANKS:
        return value  # type: ignore[return-value]
    return DEFAULT_PRIORITY
